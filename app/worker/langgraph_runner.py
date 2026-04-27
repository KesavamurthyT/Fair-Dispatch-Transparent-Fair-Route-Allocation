"""
Worker-side allocation execution logic.

This module encapsulates the CPU-heavy allocation pipeline that was
previously executed inline in the FastAPI request handler. It runs
inside a Celery worker process with synchronous DB sessions.

Key design decisions:
- Uses synchronous SQLAlchemy (psycopg2) since Celery workers are sync.
- Loads the AllocationRun payload from DB, processes it through the
  LangGraph pipeline, and persists results back to DB.
- All large intermediate artifacts (effort matrices) will be stored
  in Redis via the artifact_store (Phase 2).
"""

import json
import logging
import os
import statistics
import uuid
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_sync_session() -> Session:
    """
    Create a synchronous SQLAlchemy session for worker use.
    Workers cannot use asyncpg — they need psycopg2.
    """
    settings = get_settings()

    # Derive sync URL from async URL
    sync_url = os.environ.get("SYNC_DATABASE_URL")
    if not sync_url:
        # Convert asyncpg URL to psycopg2
        sync_url = settings.database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )

    engine = create_engine(sync_url, echo=False)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return SessionLocal()


def mark_run_status(allocation_run_id: str, status: str, error_message: str = None):
    """Update the AllocationRun status in the database."""
    session = _get_sync_session()
    try:
        from app.models.allocation_run import AllocationRun
        run = session.query(AllocationRun).filter(
            AllocationRun.id == uuid.UUID(allocation_run_id)
        ).first()
        if run:
            run.status = status
            if error_message:
                run.error_message = error_message[:500]
            if status in ("SUCCESS", "FAILED"):
                run.finished_at = datetime.utcnow()
            session.commit()
    except Exception as e:
        logger.error(f"Failed to update run status: {e}")
        session.rollback()
    finally:
        session.close()


def mark_run_failed(allocation_run_id: str, error_message: str):
    """Convenience wrapper for marking a run as failed."""
    mark_run_status(allocation_run_id, "FAILED", error_message)


def execute_allocation(allocation_run_id: str) -> dict:
    """
    Execute the full allocation pipeline for a queued AllocationRun.

    This is the core worker function that:
    1. Loads the allocation payload from the DB
    2. Runs clustering, effort computation, and the LangGraph pipeline
    3. Persists assignments and decision logs back to the DB
    4. Updates the AllocationRun status to SUCCESS

    Args:
        allocation_run_id: UUID string of the AllocationRun.

    Returns:
        dict with summary metrics (num_assignments, gini, etc.)
    """
    from app.models.allocation_run import AllocationRun, AllocationRunStatus
    from app.models.driver import Driver, DriverStatsDaily, DriverFeedback
    from app.models.driver import PreferredLanguage, VehicleType
    from app.models.package import Package, PackagePriority
    from app.models.route import Route, RoutePackage
    from app.models.assignment import Assignment
    from app.models.decision_log import DecisionLog
    from app.models.fairness_config import FairnessConfig
    from app.services.clustering import (
        cluster_packages,
        order_stops_by_nearest_neighbor,
        haversine_distance,
    )
    from app.services.workload import (
        calculate_workload,
        calculate_route_difficulty,
        estimate_route_time,
    )
    from app.services.fairness import calculate_fairness_score
    from app.models.driver_effort_model import DriverEffortModel
    from app.services.langgraph_workflow import invoke_allocation_workflow

    session = _get_sync_session()
    # Fix 3: Track all Redis artifact keys so they are cleaned up in finally,
    # even if the worker crashes mid-pipeline. Do NOT rely on TTL alone.
    _artifact_keys: List[str] = []

    try:
        # Mark as running
        mark_run_status(allocation_run_id, "RUNNING")

        # Load the allocation run and its payload
        run = session.query(AllocationRun).filter(
            AllocationRun.id == uuid.UUID(allocation_run_id)
        ).first()

        if not run:
            raise ValueError(f"AllocationRun {allocation_run_id} not found")

        if not run.payload_json:
            raise ValueError(f"AllocationRun {allocation_run_id} has no payload")

        payload = run.payload_json
        allocation_date = run.date

        # ========== UPSERT DRIVERS ==========
        driver_map = {}
        driver_models: List[Driver] = []

        for driver_input in payload["drivers"]:
            driver = session.query(Driver).filter(
                Driver.external_id == driver_input["id"]
            ).first()

            if driver:
                driver.name = driver_input["name"]
                driver.vehicle_capacity_kg = driver_input["vehicle_capacity_kg"]
                driver.preferred_language = PreferredLanguage(
                    driver_input["preferred_language"]
                )
            else:
                driver = Driver(
                    external_id=driver_input["id"],
                    name=driver_input["name"],
                    vehicle_capacity_kg=driver_input["vehicle_capacity_kg"],
                    preferred_language=PreferredLanguage(
                        driver_input["preferred_language"]
                    ),
                    vehicle_type=VehicleType.ICE,
                )
                session.add(driver)

            driver_map[driver_input["id"]] = driver

        session.flush()
        driver_models = list(driver_map.values())

        # ========== UPSERT PACKAGES ==========
        package_map = {}
        package_dicts = []

        for pkg_input in payload["packages"]:
            package = session.query(Package).filter(
                Package.external_id == pkg_input["id"]
            ).first()

            if package:
                package.weight_kg = pkg_input["weight_kg"]
                package.fragility_level = pkg_input["fragility_level"]
                package.address = pkg_input["address"]
                package.latitude = pkg_input["latitude"]
                package.longitude = pkg_input["longitude"]
                package.priority = PackagePriority(pkg_input["priority"])
            else:
                package = Package(
                    external_id=pkg_input["id"],
                    weight_kg=pkg_input["weight_kg"],
                    fragility_level=pkg_input["fragility_level"],
                    address=pkg_input["address"],
                    latitude=pkg_input["latitude"],
                    longitude=pkg_input["longitude"],
                    priority=PackagePriority(pkg_input["priority"]),
                )
                session.add(package)

            package_map[pkg_input["id"]] = package
            package_dicts.append({
                "external_id": pkg_input["id"],
                "weight_kg": pkg_input["weight_kg"],
                "fragility_level": pkg_input["fragility_level"],
                "address": pkg_input["address"],
                "latitude": pkg_input["latitude"],
                "longitude": pkg_input["longitude"],
                "priority": pkg_input["priority"],
            })

        session.flush()

        # ========== CLUSTER PACKAGES INTO ROUTES ==========
        warehouse = payload.get("warehouse", {"lat": 12.9716, "lng": 77.5946})
        clusters = cluster_packages(
            packages=package_dicts,
            num_drivers=len(payload["drivers"]),
        )

        # ========== CREATE ROUTES ==========
        route_models: List[Route] = []
        route_dicts = []

        for cluster in clusters:
            ordered_packages = order_stops_by_nearest_neighbor(
                cluster.packages,
                warehouse["lat"],
                warehouse["lng"],
            )

            # Calculate total distance
            total_dist = 0.0
            curr_lat, curr_lng = warehouse["lat"], warehouse["lng"]

            for p in ordered_packages:
                dist = haversine_distance(curr_lat, curr_lng, p["latitude"], p["longitude"])
                total_dist += dist
                curr_lat, curr_lng = p["latitude"], p["longitude"]

            total_dist += haversine_distance(
                curr_lat, curr_lng, warehouse["lat"], warehouse["lng"]
            )

            avg_fragility = sum(
                p["fragility_level"] for p in cluster.packages
            ) / max(len(cluster.packages), 1)

            difficulty = calculate_route_difficulty(
                total_weight_kg=cluster.total_weight_kg,
                num_stops=cluster.num_stops,
                avg_fragility=avg_fragility,
            )
            est_time = estimate_route_time(
                num_packages=cluster.num_packages,
                num_stops=cluster.num_stops,
            )

            route = Route(
                date=allocation_date,
                cluster_id=cluster.cluster_id,
                total_weight_kg=cluster.total_weight_kg,
                num_packages=cluster.num_packages,
                num_stops=cluster.num_stops,
                route_difficulty_score=difficulty,
                estimated_time_minutes=est_time,
                total_distance_km=total_dist,
                allocation_run_id=uuid.UUID(allocation_run_id),
            )
            session.add(route)
            route_models.append(route)

            workload = calculate_workload({
                "num_packages": cluster.num_packages,
                "total_weight_kg": cluster.total_weight_kg,
                "route_difficulty_score": difficulty,
                "estimated_time_minutes": est_time,
            })

            route_dicts.append({
                "cluster_id": cluster.cluster_id,
                "num_packages": cluster.num_packages,
                "total_weight_kg": cluster.total_weight_kg,
                "num_stops": cluster.num_stops,
                "route_difficulty_score": difficulty,
                "estimated_time_minutes": est_time,
                "workload_score": workload,
                "packages": ordered_packages,
            })

        session.flush()
        run.num_routes = len(route_models)

        # Create RoutePackage associations
        for i, route in enumerate(route_models):
            for stop_order, pkg_data in enumerate(route_dicts[i]["packages"]):
                package = package_map[pkg_data["external_id"]]
                route_package = RoutePackage(
                    route_id=route.id,
                    package_id=package.id,
                    stop_order=stop_order + 1,
                )
                session.add(route_package)

        # ========== GET FAIRNESS CONFIG ==========
        active_config = session.query(FairnessConfig).filter(
            FairnessConfig.is_active == True
        ).first()

        config_used = {}
        if active_config:
            config_used = {
                "gini_threshold": active_config.gini_threshold,
                "stddev_threshold": active_config.stddev_threshold,
                "max_gap_threshold": active_config.max_gap_threshold,
                "ev_safety_margin_pct": active_config.ev_safety_margin_pct,
                "ev_charging_penalty_weight": active_config.ev_charging_penalty_weight,
                "recovery_penalty_weight": active_config.recovery_penalty_weight,
                "recovery_lightening_factor": active_config.recovery_lightening_factor,
            }

        # ========== BUILD DRIVER CONTEXTS ==========
        driver_contexts: Dict[str, dict] = {}
        cutoff_date = allocation_date - timedelta(days=7)

        for driver in driver_models:
            driver_id_str = str(driver.id)

            recent_stats = session.query(DriverStatsDaily).filter(
                DriverStatsDaily.driver_id == driver.id,
                DriverStatsDaily.date >= cutoff_date,
            ).order_by(DriverStatsDaily.date.desc()).all()

            if recent_stats:
                recent_efforts = [
                    s.avg_workload_score for s in recent_stats
                    if s.avg_workload_score
                ]
                if recent_efforts:
                    recent_avg = statistics.mean(recent_efforts)
                    recent_std = (
                        statistics.stdev(recent_efforts)
                        if len(recent_efforts) > 1
                        else 0.0
                    )
                else:
                    recent_avg = 60.0
                    recent_std = 15.0

                hard_threshold = recent_avg + recent_std
                hard_days = sum(1 for e in recent_efforts if e > hard_threshold)
            else:
                recent_avg = 60.0
                recent_std = 15.0
                hard_days = 0

            recent_feedback = session.query(DriverFeedback).filter(
                DriverFeedback.driver_id == driver.id
            ).order_by(DriverFeedback.created_at.desc()).first()

            fatigue_score = (
                float(recent_feedback.tiredness_level) if recent_feedback else 3.0
            )
            fatigue_score = max(1.0, min(5.0, fatigue_score))

            driver_contexts[driver_id_str] = {
                "driver_id": driver_id_str,
                "recent_avg_effort": recent_avg,
                "recent_std_effort": recent_std,
                "recent_hard_days": hard_days,
                "fatigue_score": fatigue_score,
                "preferences": {},
            }

        # ========== SERIALIZE MODELS FOR LANGGRAPH ==========
        driver_model_dicts = []
        for d in driver_models:
            driver_model_dicts.append({
                "id": str(d.id),
                "external_id": d.external_id,
                "name": d.name,
                "vehicle_capacity_kg": d.vehicle_capacity_kg,
                "preferred_language": (
                    d.preferred_language.value
                    if hasattr(d.preferred_language, "value")
                    else d.preferred_language
                ),
                "vehicle_type": (
                    d.vehicle_type.value
                    if hasattr(d.vehicle_type, "value")
                    else str(d.vehicle_type)
                ),
                "battery_range_km": getattr(d, "battery_range_km", None),
                "charging_time_minutes": getattr(d, "charging_time_minutes", None),
                "is_ev": (
                    d.vehicle_type.value == "EV"
                    if hasattr(d.vehicle_type, "value")
                    else str(d.vehicle_type) == "EV"
                ),
                "experience_years": getattr(d, "experience_years", 2),
            })

        route_model_dicts = []
        for r in route_models:
            route_model_dicts.append({
                "id": str(r.id),
                "date": str(r.date),
                "cluster_id": r.cluster_id,
                "total_weight_kg": r.total_weight_kg,
                "num_packages": r.num_packages,
                "num_stops": r.num_stops,
                "route_difficulty_score": r.route_difficulty_score,
                "estimated_time_minutes": r.estimated_time_minutes,
                "total_distance_km": r.total_distance_km,
            })

        for i, rd in enumerate(route_dicts):
            rd["id"] = str(route_models[i].id)

        # ========== RECOVERY TARGETS (simplified sync version) ==========
        recovery_targets_str: Dict[str, Optional[float]] = {}
        # Recovery targets will be populated in a future integration;
        # for now pass empty dict (same behavior as no recovery config).

        # ========== FIX 2B: LOAD + RECORD MODEL VERSION ==========
        # Query the active DriverEffortModel for any driver to record the
        # current model version used in this run (for reproducibility).
        sample_driver = driver_models[0] if driver_models else None
        if sample_driver:
            try:
                active_model = session.query(DriverEffortModel).filter(
                    DriverEffortModel.driver_id == sample_driver.id,
                    DriverEffortModel.active == True,
                ).first()
                if active_model:
                    run.model_version_used = f"v{active_model.model_version}"
                    session.flush()
                    logger.info(
                        "Using model version %s for allocation %s",
                        run.model_version_used,
                        allocation_run_id,
                    )
            except Exception as mv_err:
                logger.warning("Could not record model version: %s", mv_err)

        # ========== INVOKE LANGGRAPH WORKFLOW ==========
        import asyncio

        workflow_result = asyncio.run(
            invoke_allocation_workflow(
                request_dict=payload,
                config_used=config_used,
                driver_models=driver_model_dicts,
                route_models=route_model_dicts,
                route_dicts=route_dicts,
                driver_contexts=driver_contexts,
                recovery_targets=recovery_targets_str,
                allocation_run_id=allocation_run_id,
                thread_id=allocation_run_id,
            )
        )

        # ========== PERSIST DECISION LOGS ==========
        for log_entry in workflow_result.decision_logs:
            decision_log = DecisionLog(
                allocation_run_id=uuid.UUID(allocation_run_id),
                agent_name=log_entry["agent_name"],
                step_type=log_entry["step_type"],
                input_snapshot=log_entry.get("input_snapshot", {}),
                output_snapshot=log_entry.get("output_snapshot", {}),
            )
            session.add(decision_log)

        # ========== CREATE ASSIGNMENTS ==========
        final_proposal = (
            workflow_result.final_proposal or workflow_result.route_proposal_1
        )
        final_fairness = (
            workflow_result.final_fairness or workflow_result.fairness_check_1
        )
        final_per_driver_effort = (
            workflow_result.final_per_driver_effort
            or final_proposal["per_driver_effort"]
        )

        driver_by_id = {str(d.id): d for d in driver_models}
        route_by_id = {str(r.id): r for r in route_models}

        num_assignments = 0
        for alloc_item in final_proposal["allocation"]:
            driver_id_str = str(alloc_item["driver_id"])
            route_id_str = str(alloc_item["route_id"])

            driver = driver_by_id.get(driver_id_str)
            route = route_by_id.get(route_id_str)

            if not driver or not route:
                continue

            effort = final_per_driver_effort.get(
                driver_id_str, alloc_item["effort"]
            )
            avg_effort = final_fairness["metrics"]["avg_effort"]
            fairness_score = calculate_fairness_score(effort, avg_effort)

            explanation_data = workflow_result.explanations.get(driver_id_str, {})
            driver_explanation = explanation_data.get(
                "driver_explanation", "Route assigned."
            )
            admin_explanation = explanation_data.get("admin_explanation", "")

            assignment = Assignment(
                date=allocation_date,
                driver_id=driver.id,
                route_id=route.id,
                workload_score=effort,
                fairness_score=fairness_score,
                explanation=driver_explanation,
                driver_explanation=driver_explanation,
                admin_explanation=admin_explanation,
                allocation_run_id=uuid.UUID(allocation_run_id),
            )
            session.add(assignment)
            num_assignments += 1

        # ========== FINALIZE ==========
        metrics = final_fairness["metrics"]
        run.global_gini_index = metrics["gini_index"]
        run.global_std_dev = metrics["std_dev"]
        run.global_max_gap = metrics["max_gap"]
        run.status = AllocationRunStatus.SUCCESS
        run.finished_at = datetime.utcnow()

        session.commit()

        logger.info(
            f"Allocation {allocation_run_id} completed: "
            f"{num_assignments} assignments, Gini={metrics['gini_index']:.3f}"
        )

        return {
            "num_assignments": num_assignments,
            "gini_index": metrics["gini_index"],
            "std_dev": metrics["std_dev"],
        }

    except Exception as e:
        session.rollback()
        mark_run_status(allocation_run_id, "FAILED", str(e))
        raise
    finally:
        session.close()
        # Fix 3: Explicit cleanup of all Redis artifact keys regardless of outcome.
        # This runs even if the worker process crashes after partial work.
        if _artifact_keys:
            try:
                from app.utils.artifact_store import get_artifact_store
                store = get_artifact_store()
                deleted = store.delete_keys(*_artifact_keys)
                logger.info(
                    "Cleaned up %d Redis artifact key(s) for run %s",
                    deleted,
                    allocation_run_id,
                )
            except Exception as cleanup_err:
                logger.warning(
                    "Artifact cleanup failed for run %s: %s — keys will expire via TTL",
                    allocation_run_id,
                    cleanup_err,
                )
