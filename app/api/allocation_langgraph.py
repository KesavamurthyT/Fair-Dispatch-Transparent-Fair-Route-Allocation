"""
LangGraph-enabled Allocation API endpoint.

Provides two modes:
  - POST /langgraph       → Async (HTTP 202). Enqueues a Celery task and returns job_id.
  - POST /langgraph_sync  → Deprecated sync (HTTP 200). Kept behind feature flag for migration.
  - GET  /status/{job_id} → Poll async job status.
"""

import os
import statistics
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Header, status, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models import Driver, Package, Route, RoutePackage, Assignment
from app.models.driver import PreferredLanguage, VehicleType
from app.models.package import PackagePriority
from app.models.allocation_run import AllocationRun, AllocationRunStatus
from app.models.decision_log import DecisionLog
from app.models.driver import DriverStatsDaily, DriverFeedback
from app.models.fairness_config import FairnessConfig
from app.schemas.allocation import (
    AllocationRequest,
    AllocationResponse,
    AssignmentResponse,
    GlobalFairness,
    RouteSummary,
)
from app.services.clustering import cluster_packages, order_stops_by_nearest_neighbor, haversine_distance
from app.services.workload import calculate_workload, calculate_route_difficulty, estimate_route_time
from app.services.fairness import calculate_fairness_score
from app.services.learning_agent import LearningAgent, hash_config
from app.schemas.allocation_state import AllocationState
from app.services.langgraph_workflow import invoke_allocation_workflow

router = APIRouter(prefix="/allocate", tags=["Allocation"])

settings = get_settings()


# =============================================================================
# NEW: Async Allocation Endpoint (HTTP 202)
# =============================================================================

@router.post(
    "/langgraph",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Allocate packages to drivers (async via worker)",
    description="""
    Enqueues an allocation job to be processed by a background worker.
    Returns immediately with a job_id that can be polled via GET /allocate/status/{job_id}.
    
    The worker executes the full multi-agent LangGraph pipeline:
    1. ML Effort Agent → 2. Route Planner → 3. Fairness Manager →
    4. Driver Liaison → 5. Final Resolution → 6. Explainability → 7. Learning Agent
    """,
)
async def allocate_langgraph_async(
    request: AllocationRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(
        default=None,
        alias="Idempotency-Key",
        description="Optional client-generated idempotency key (UUID). "
                    "Prevents duplicate allocations on retry.",
    ),
) -> dict:
    """Accept allocation request and enqueue to worker."""

    # Fix 4: Idempotency check — if a run with this key already exists, return it.
    if idempotency_key:
        existing_result = await db.execute(
            select(AllocationRun).where(
                AllocationRun.idempotency_key == idempotency_key
            )
        )
        existing_run = existing_result.scalar_one_or_none()
        if existing_run:
            return {
                "job_id": str(existing_run.id),
                "status": existing_run.status.value
                          if hasattr(existing_run.status, "value")
                          else str(existing_run.status),
                "message": "Duplicate request — returning existing job.",
                "idempotency_key": idempotency_key,
            }

    # Validate input
    if not request.packages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 package is required",
        )
    if not request.drivers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 driver is required",
        )

    allocation_date = request.allocation_date

    # Create allocation run with QUEUED status and store the payload
    allocation_run = AllocationRun(
        date=allocation_date,
        num_drivers=len(request.drivers),
        num_packages=len(request.packages),
        num_routes=0,
        status=AllocationRunStatus.QUEUED,
        started_at=datetime.utcnow(),
        payload_json=request.model_dump(mode="json"),
        idempotency_key=idempotency_key,  # Fix 4: store key (None if not provided)
    )
    db.add(allocation_run)
    await db.flush()

    # Enqueue the Celery task
    try:
        from app.tasks import run_allocation_job
        run_allocation_job.apply_async(
            args=[str(allocation_run.id)],
            queue="allocation",
        )
    except Exception as e:
        # If Celery/Redis is unavailable, mark failed
        allocation_run.status = AllocationRunStatus.FAILED
        allocation_run.error_message = f"Failed to enqueue task: {str(e)[:200]}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Worker queue unavailable",
                "run_id": str(allocation_run.id),
                "error": str(e)[:200],
            },
        )

    await db.commit()

    return {
        "job_id": str(allocation_run.id),
        "status": "queued",
        "message": "Allocation job accepted. Poll GET /allocate/status/{job_id} for progress.",
        "idempotency_key": idempotency_key,  # echo back so client can store it
    }


# =============================================================================
# NEW: Job Status Polling Endpoint
# =============================================================================

@router.get(
    "/status/{job_id}",
    summary="Check allocation job status",
    description="Poll the status of an async allocation job.",
)
async def allocation_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get current status of an allocation job."""

    try:
        run_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job_id format — must be a UUID",
        )

    result = await db.execute(
        select(AllocationRun).where(AllocationRun.id == run_uuid)
    )
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Allocation job {job_id} not found",
        )

    response = {
        "job_id": str(run.id),
        "status": run.status.value if hasattr(run.status, "value") else str(run.status),
        "date": str(run.date),
        "num_drivers": run.num_drivers,
        "num_routes": run.num_routes,
        "num_packages": run.num_packages,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        # Fix 2: model version used (for reproducibility)
        "model_version_used": run.model_version_used,
    }

    if run.status == AllocationRunStatus.SUCCESS:
        response["result"] = {
            "global_gini_index": run.global_gini_index,
            "global_std_dev": run.global_std_dev,
            "global_max_gap": run.global_max_gap,
            "assignments_url": f"/api/v1/runs/{run.id}/assignments",
        }
        # Fix 5: expose fairness iteration metadata from the run result_json (if populated)
        try:
            result_meta = getattr(run, "result_json", None) or {}
            if result_meta.get("fairness_metadata"):
                response["fairness_metadata"] = result_meta["fairness_metadata"]
        except Exception:
            pass

    if run.status == AllocationRunStatus.FAILED:
        response["error"] = run.error_message

    return response


# =============================================================================
# DEPRECATED: Sync Allocation Endpoint (behind feature flag)
# =============================================================================

@router.post(
    "/langgraph_sync",
    response_model=AllocationResponse,
    status_code=status.HTTP_200_OK,
    summary="[DEPRECATED] Allocate packages to drivers (synchronous)",
    description="""
    **DEPRECATED** — Use POST /allocate/langgraph (async) instead.
    
    This synchronous endpoint is kept temporarily for backward compatibility
    and A/B testing between LAP and CVRP. It will be removed after migration.
    
    Controlled by SYNC_ALLOCATION_ENABLED feature flag.
    """,
    deprecated=True,
)
async def allocate_langgraph_sync(
    request: AllocationRequest,
    db: AsyncSession = Depends(get_db),
    enable_gemini: bool = Query(False, description="Enable Gemini 1.5 Flash explanations"),
) -> AllocationResponse:
    """[DEPRECATED] Perform fair route allocation synchronously."""

    if not settings.sync_allocation_enabled:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Synchronous allocation endpoint has been disabled. Use POST /allocate/langgraph (async).",
        )

    # Validate input
    if not request.packages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 package is required",
        )
    if not request.drivers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 driver is required",
        )
    
    allocation_date = request.allocation_date
    
    # ========== START ALLOCATION RUN ==========
    allocation_run = AllocationRun(
        date=allocation_date,
        num_drivers=len(request.drivers),
        num_packages=len(request.packages),
        num_routes=0,
        status=AllocationRunStatus.PENDING,
        started_at=datetime.utcnow(),
    )
    db.add(allocation_run)
    await db.flush()
    
    try:
        # ========== PHASE 0: UPSERT DATA & CLUSTERING ==========
        # (Same as original - this is DB-dependent and must stay in endpoint)
        
        # Step 1: Upsert drivers
        driver_map = {}
        driver_models: List[Driver] = []
        
        for driver_input in request.drivers:
            result = await db.execute(
                select(Driver).where(Driver.external_id == driver_input.id)
            )
            driver = result.scalar_one_or_none()
            
            if driver:
                driver.name = driver_input.name
                driver.vehicle_capacity_kg = driver_input.vehicle_capacity_kg
                driver.preferred_language = PreferredLanguage(driver_input.preferred_language)
            else:
                driver = Driver(
                    external_id=driver_input.id,
                    name=driver_input.name,
                    vehicle_capacity_kg=driver_input.vehicle_capacity_kg,
                    preferred_language=PreferredLanguage(driver_input.preferred_language),
                    vehicle_type=VehicleType.ICE,
                )
                db.add(driver)
            
            driver_map[driver_input.id] = driver
        
        await db.flush()
        driver_models = list(driver_map.values())
        
        # Step 2: Upsert packages
        package_map = {}
        package_dicts = []
        
        for pkg_input in request.packages:
            result = await db.execute(
                select(Package).where(Package.external_id == pkg_input.id)
            )
            package = result.scalar_one_or_none()
            
            if package:
                package.weight_kg = pkg_input.weight_kg
                package.fragility_level = pkg_input.fragility_level
                package.address = pkg_input.address
                package.latitude = pkg_input.latitude
                package.longitude = pkg_input.longitude
                package.priority = PackagePriority(pkg_input.priority)
            else:
                package = Package(
                    external_id=pkg_input.id,
                    weight_kg=pkg_input.weight_kg,
                    fragility_level=pkg_input.fragility_level,
                    address=pkg_input.address,
                    latitude=pkg_input.latitude,
                    longitude=pkg_input.longitude,
                    priority=PackagePriority(pkg_input.priority),
                )
                db.add(package)
            
            package_map[pkg_input.id] = package
            package_dicts.append({
                "external_id": pkg_input.id,
                "weight_kg": pkg_input.weight_kg,
                "fragility_level": pkg_input.fragility_level,
                "address": pkg_input.address,
                "latitude": pkg_input.latitude,
                "longitude": pkg_input.longitude,
                "priority": pkg_input.priority,
            })
        
        await db.flush()
        
        # Step 3: Cluster packages into routes
        clusters = cluster_packages(
            packages=package_dicts,
            num_drivers=len(request.drivers),
        )
        
        # Step 4: Create routes
        route_models: List[Route] = []
        route_dicts = []
        
        for cluster in clusters:
            ordered_packages = order_stops_by_nearest_neighbor(
                cluster.packages,
                request.warehouse.lat,
                request.warehouse.lng,
            )
            
            # Calculate total distance
            total_dist = 0.0
            curr_lat, curr_lng = request.warehouse.lat, request.warehouse.lng
            
            for p in ordered_packages:
                dist = haversine_distance(curr_lat, curr_lng, p["latitude"], p["longitude"])
                total_dist += dist
                curr_lat, curr_lng = p["latitude"], p["longitude"]
            
            total_dist += haversine_distance(curr_lat, curr_lng, request.warehouse.lat, request.warehouse.lng)
            
            avg_fragility = sum(p["fragility_level"] for p in cluster.packages) / max(len(cluster.packages), 1)
            
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
                allocation_run_id=allocation_run.id,
            )
            db.add(route)
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
        
        await db.flush()
        
        allocation_run.num_routes = len(route_models)
        
        # Create RoutePackage associations
        for i, route in enumerate(route_models):
            for stop_order, pkg_data in enumerate(route_dicts[i]["packages"]):
                package = package_map[pkg_data["external_id"]]
                route_package = RoutePackage(
                    route_id=route.id,
                    package_id=package.id,
                    stop_order=stop_order + 1,
                )
                db.add(route_package)
        
        # ========== GET CONFIG ==========
        config_result = await db.execute(
            select(FairnessConfig).where(FairnessConfig.is_active == True).limit(1)
        )
        active_config = config_result.scalar_one_or_none()
        
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
        
        # ========== GET RECOVERY TARGETS ==========
        from app.services.recovery_service import get_driver_recovery_targets
        
        driver_ids = [d.id for d in driver_models]
        recovery_targets = await get_driver_recovery_targets(
            db, driver_ids, allocation_date, active_config
        )
        recovery_targets_str = {str(k): v for k, v in recovery_targets.items()}
        
        # ========== BUILD DRIVER CONTEXTS ==========
        driver_contexts: Dict[str, dict] = {}
        cutoff_date = allocation_date - timedelta(days=7)
        
        for driver in driver_models:
            driver_id_str = str(driver.id)
            
            stats_result = await db.execute(
                select(DriverStatsDaily)
                .where(DriverStatsDaily.driver_id == driver.id)
                .where(DriverStatsDaily.date >= cutoff_date)
                .order_by(DriverStatsDaily.date.desc())
            )
            recent_stats = stats_result.scalars().all()
            
            if recent_stats:
                recent_efforts = [s.avg_workload_score for s in recent_stats if s.avg_workload_score]
                if recent_efforts:
                    recent_avg = statistics.mean(recent_efforts)
                    recent_std = statistics.stdev(recent_efforts) if len(recent_efforts) > 1 else 0.0
                else:
                    recent_avg = 60.0
                    recent_std = 15.0
                
                hard_threshold = recent_avg + recent_std
                hard_days = sum(1 for e in recent_efforts if e > hard_threshold)
            else:
                recent_avg = 60.0
                recent_std = 15.0
                hard_days = 0
            
            feedback_result = await db.execute(
                select(DriverFeedback)
                .where(DriverFeedback.driver_id == driver.id)
                .order_by(DriverFeedback.created_at.desc())
                .limit(1)
            )
            recent_feedback = feedback_result.scalar_one_or_none()
            fatigue_score = float(recent_feedback.tiredness_level) if recent_feedback else 3.0
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
                "preferred_language": d.preferred_language.value if hasattr(d.preferred_language, 'value') else d.preferred_language,
                "vehicle_type": d.vehicle_type.value if hasattr(d.vehicle_type, 'value') else str(d.vehicle_type),
                "battery_range_km": getattr(d, 'battery_range_km', None),
                "charging_time_minutes": getattr(d, 'charging_time_minutes', None),
                "is_ev": d.vehicle_type.value == "EV" if hasattr(d.vehicle_type, 'value') else str(d.vehicle_type) == "EV",
                "experience_years": getattr(d, 'experience_years', 2),
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
        
        # Add route IDs to route_dicts
        for i, rd in enumerate(route_dicts):
            rd["id"] = str(route_models[i].id)
        
        # ========== INVOKE LANGGRAPH WORKFLOW ==========
        if enable_gemini:
            os.environ["ENABLE_GEMINI_EXPLAIN"] = "true"
        
        workflow_result = await invoke_allocation_workflow(
            request_dict=request.model_dump(mode="json"),
            config_used=config_used,
            driver_models=driver_model_dicts,
            route_models=route_model_dicts,
            route_dicts=route_dicts,
            driver_contexts=driver_contexts,
            recovery_targets=recovery_targets_str,
            allocation_run_id=str(allocation_run.id),
            thread_id=str(allocation_run.id),
        )
        
        # ========== PERSIST DECISION LOGS ==========
        for log_entry in workflow_result.decision_logs:
            decision_log = DecisionLog(
                allocation_run_id=allocation_run.id,
                agent_name=log_entry["agent_name"],
                step_type=log_entry["step_type"],
                input_snapshot=log_entry.get("input_snapshot", {}),
                output_snapshot=log_entry.get("output_snapshot", {}),
            )
            db.add(decision_log)
        
        # ========== CREATE ASSIGNMENTS ==========
        final_proposal = workflow_result.final_proposal or workflow_result.route_proposal_1
        final_fairness = workflow_result.final_fairness or workflow_result.fairness_check_1
        final_per_driver_effort = workflow_result.final_per_driver_effort or final_proposal["per_driver_effort"]
        
        driver_by_id = {str(d.id): d for d in driver_models}
        route_by_id = {str(r.id): r for r in route_models}
        
        assignments_response = []
        
        for alloc_item in final_proposal["allocation"]:
            driver_id_str = str(alloc_item["driver_id"])
            route_id_str = str(alloc_item["route_id"])
            
            driver = driver_by_id.get(driver_id_str)
            route = route_by_id.get(route_id_str)
            
            if not driver or not route:
                continue
            
            effort = final_per_driver_effort.get(driver_id_str, alloc_item["effort"])
            avg_effort = final_fairness["metrics"]["avg_effort"]
            fairness_score = calculate_fairness_score(effort, avg_effort)
            
            explanation_data = workflow_result.explanations.get(driver_id_str, {})
            driver_explanation = explanation_data.get("driver_explanation", "Route assigned.")
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
                allocation_run_id=allocation_run.id,
            )
            db.add(assignment)
            
            assignments_response.append(AssignmentResponse(
                driver_id=driver.id,
                driver_external_id=driver.external_id,
                driver_name=driver.name,
                route_id=route.id,
                workload_score=effort,
                fairness_score=fairness_score,
                route_summary=RouteSummary(
                    num_packages=route.num_packages,
                    total_weight_kg=route.total_weight_kg,
                    num_stops=route.num_stops,
                    route_difficulty_score=route.route_difficulty_score,
                    estimated_time_minutes=route.estimated_time_minutes,
                ),
                explanation=driver_explanation,
            ))
        
        # ========== UPDATE DAILY STATS ==========
        from app.services.recovery_service import update_daily_stats_for_run
        
        await update_daily_stats_for_run(
            db=db,
            allocation_run_id=allocation_run.id,
            target_date=allocation_date,
            config=active_config,
        )
        
        # ========== CREATE LEARNING EPISODE ==========
        try:
            learning_agent = LearningAgent(db)
            
            import random
            is_experimental = random.random() < 0.10
            
            await learning_agent.create_episode(
                allocation_run_id=allocation_run.id,
                fairness_config=config_used,
                num_drivers=len(driver_models),
                num_routes=len(route_models),
                is_experimental=is_experimental,
            )
        except Exception as learning_error:
            import logging
            logging.warning(f"Failed to create learning episode: {learning_error}")
        
        # ========== FINALIZE ==========
        metrics = final_fairness["metrics"]
        allocation_run.global_gini_index = metrics["gini_index"]
        allocation_run.global_std_dev = metrics["std_dev"]
        allocation_run.global_max_gap = metrics["max_gap"]
        allocation_run.status = AllocationRunStatus.SUCCESS
        allocation_run.finished_at = datetime.utcnow()
        
        await db.commit()
        
        return AllocationResponse(
            allocation_run_id=allocation_run.id,
            allocation_date=allocation_date,
            global_fairness=GlobalFairness(
                avg_workload=metrics["avg_effort"],
                std_dev=metrics["std_dev"],
                gini_index=metrics["gini_index"],
            ),
            assignments=assignments_response,
        )
        
    except Exception as e:
        allocation_run.status = AllocationRunStatus.FAILED
        allocation_run.error_message = str(e)[:500]
        allocation_run.finished_at = datetime.utcnow()
        await db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "LangGraph allocation failed",
                "run_id": str(allocation_run.id),
                "error": str(e)[:200],
            },
        )
