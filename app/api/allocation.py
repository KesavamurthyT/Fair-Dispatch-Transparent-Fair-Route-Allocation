"""
Allocation API endpoint.
Handles POST /api/v1/allocate for fair route allocation using multi-agent pipeline.

Phase 4.1: Multi-agent architecture with MLEffortAgent, RoutePlannerAgent, FairnessManagerAgent.
Phase 4.2: Added Driver Liaison Agents and Final Resolution for negotiation.
Phase 4.3: Added ExplainabilityAgent v2 for template-based explanations.
Phase 8: Learning Agent integration for bandit-based config tuning.
"""

import statistics
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Driver, Package, Route, RoutePackage, Assignment
from app.models.driver import PreferredLanguage, VehicleType
from app.models.package import PackagePriority
from app.models.allocation_run import AllocationRun, AllocationRunStatus
from app.models.decision_log import DecisionLog
from app.schemas.allocation import (
    AllocationRequest,
    AllocationResponse,
    AssignmentResponse,
    GlobalFairness,
    RouteSummary,
)
from app.schemas.agent_schemas import (
    FairnessThresholds,
    DriverAssignmentProposal,
    DriverContext,
    DriverLiaisonDecision,
)
from app.services.clustering import cluster_packages, order_stops_by_nearest_neighbor
from app.services.workload import calculate_workload, calculate_route_difficulty, estimate_route_time
from app.services.fairness import calculate_fairness_score
from app.services.explainability import ExplainabilityAgent, generate_explanation
from app.services.ml_effort_agent import MLEffortAgent
from app.services.route_planner_agent import RoutePlannerAgent
from app.services.fairness_manager_agent import FairnessManagerAgent
from app.services.driver_liaison_agent import DriverLiaisonAgent
from app.services.final_resolution import FinalResolutionAgent
from app.models.driver import DriverStatsDaily, DriverFeedback
from app.models.manual_override import ManualOverride
from app.models.fairness_config import FairnessConfig
from app.schemas.explainability import DriverExplanationInput
from app.services.learning_agent import LearningAgent, hash_config

router = APIRouter(prefix="/allocate", tags=["Allocation"])


@router.post(
    "",
    response_model=AllocationResponse,
    status_code=status.HTTP_200_OK,
    summary="Allocate packages to drivers",
    description="""
    Main allocation endpoint using multi-agent pipeline:
    1. Phase 0: Cluster packages into routes
    2. Phase 1: ML Effort Agent builds effort matrix
    3. Phase 2: Route Planner Agent generates optimal assignment (Proposal 1)
    4. Phase 3: Fairness Manager evaluates; may request re-optimization (Proposal 2)
    5. Persist AllocationRun, Assignments, and DecisionLog entries
    """,
)
async def allocate(
    request: AllocationRequest,
    db: AsyncSession = Depends(get_db),
) -> AllocationResponse:
    """Perform fair route allocation using multi-agent pipeline."""
    
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
    
    # ========== START ALLOCATION RUN ==========
    allocation_run = AllocationRun(
        date=request.allocation_date,
        num_drivers=len(request.drivers),
        num_packages=len(request.packages),
        num_routes=0,  # Updated after clustering
        status=AllocationRunStatus.PENDING,
        started_at=datetime.utcnow(),
    )
    db.add(allocation_run)
    await db.flush()  # Get allocation_run.id
    
    try:
        # ========== PHASE 0: UPSERT DATA & CLUSTERING ==========
        
        # Step 1: Upsert drivers
        driver_map = {}  # external_id -> Driver model
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
            
            # Calculate total distance (Warehouse -> Stop 1 -> ... -> Stop N -> Warehouse)
            from app.services.clustering import haversine_distance
            total_dist = 0.0
            curr_lat, curr_lng = request.warehouse.lat, request.warehouse.lng
            
            for p in ordered_packages:
                dist = haversine_distance(curr_lat, curr_lng, p["latitude"], p["longitude"])
                total_dist += dist
                curr_lat, curr_lng = p["latitude"], p["longitude"]
            
            # Add return trip to warehouse
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
                date=request.allocation_date,
                cluster_id=cluster.cluster_id,
                total_weight_kg=cluster.total_weight_kg,
                num_packages=cluster.num_packages,
                num_stops=cluster.num_stops,
                route_difficulty_score=difficulty,
                estimated_time_minutes=est_time,
                total_distance_km=total_dist,
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
        
        # Update allocation run with route count
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
        
        # ========== PHASE 1: ML EFFORT AGENT ==========
        ml_agent = MLEffortAgent()
        
        # Get active fairness config for EV settings
        config_result = await db.execute(
            select(FairnessConfig).where(FairnessConfig.is_active == True).limit(1)
        )
        active_config = config_result.scalar_one_or_none()
        
        ev_config = {
            "safety_margin_pct": active_config.ev_safety_margin_pct if active_config else 10.0,
            "charging_penalty_weight": active_config.ev_charging_penalty_weight if active_config else 0.3,
        }
        
        effort_result = ml_agent.compute_effort_matrix(
            drivers=driver_models,
            routes=route_models,
            ev_config=ev_config,
        )
        
        # Log decision
        ml_log = DecisionLog(
            allocation_run_id=allocation_run.id,
            agent_name="ML_EFFORT",
            step_type="MATRIX_GENERATION",
            input_snapshot=ml_agent.get_input_snapshot(driver_models, route_models),
            output_snapshot={
                **ml_agent.get_output_snapshot(effort_result),
                "num_infeasible_ev_pairs": len(effort_result.infeasible_pairs),
            },
        )
        db.add(ml_log)
        
        # ========== PHASE 1.5: RECOVERY TARGETS (Phase 7) ==========
        from app.services.recovery_service import get_driver_recovery_targets
        
        driver_ids = [d.id for d in driver_models]
        recovery_targets = await get_driver_recovery_targets(
            db, driver_ids, request.allocation_date, active_config
        )
        recovery_penalty_weight = active_config.recovery_penalty_weight if active_config else 3.0
        
        # ========== PHASE 2: ROUTE PLANNER AGENT - PROPOSAL 1 ==========
        planner_agent = RoutePlannerAgent()
        
        proposal1 = planner_agent.plan(
            effort_result=effort_result,
            drivers=driver_models,
            routes=route_models,
            recovery_targets=recovery_targets,
            recovery_penalty_weight=recovery_penalty_weight,
            proposal_number=1,
        )
        
        # Log proposal 1
        proposal1_log = DecisionLog(
            allocation_run_id=allocation_run.id,
            agent_name="ROUTE_PLANNER",
            step_type="PROPOSAL_1",
            input_snapshot=planner_agent.get_input_snapshot(effort_result),
            output_snapshot=planner_agent.get_output_snapshot(proposal1),
        )
        db.add(proposal1_log)
        
        # ========== PHASE 3: FAIRNESS MANAGER AGENT ==========
        fairness_agent = FairnessManagerAgent(
            thresholds=FairnessThresholds(
                gini_threshold=0.33,
                stddev_threshold=25.0,
                max_gap_threshold=25.0,
            )
        )
        
        fairness_check1 = fairness_agent.check(proposal1, proposal_number=1)
        
        # Log fairness check 1
        fairness1_log = DecisionLog(
            allocation_run_id=allocation_run.id,
            agent_name="FAIRNESS_MANAGER",
            step_type="FAIRNESS_CHECK_PROPOSAL_1",
            input_snapshot=fairness_agent.get_input_snapshot(proposal1),
            output_snapshot=fairness_agent.get_output_snapshot(fairness_check1),
        )
        db.add(fairness1_log)
        
        # Determine final allocation
        final_plan = proposal1
        final_fairness = fairness_check1
        
        if fairness_check1.status == "REOPTIMIZE" and fairness_check1.recommendations:
            # Build penalties and run Proposal 2
            penalties = planner_agent.build_penalties_from_recommendations(
                fairness_check1.recommendations,
                proposal1.per_driver_effort,
            )
            
            proposal2 = planner_agent.plan(
                effort_result=effort_result,
                drivers=driver_models,
                routes=route_models,
                fairness_penalties=penalties,
                recovery_targets=recovery_targets,
                recovery_penalty_weight=recovery_penalty_weight,
                proposal_number=2,
            )
            
            # Log proposal 2
            proposal2_log = DecisionLog(
                allocation_run_id=allocation_run.id,
                agent_name="ROUTE_PLANNER",
                step_type="PROPOSAL_2",
                input_snapshot=planner_agent.get_input_snapshot(effort_result, penalties),
                output_snapshot=planner_agent.get_output_snapshot(proposal2),
            )
            db.add(proposal2_log)
            
            # Check fairness of proposal 2
            fairness_check2 = fairness_agent.check(proposal2, proposal_number=2)
            
            # Log fairness check 2
            fairness2_log = DecisionLog(
                allocation_run_id=allocation_run.id,
                agent_name="FAIRNESS_MANAGER",
                step_type="FAIRNESS_CHECK_PROPOSAL_2",
                input_snapshot=fairness_agent.get_input_snapshot(proposal2),
                output_snapshot=fairness_agent.get_output_snapshot(fairness_check2),
            )
            db.add(fairness2_log)
            
            # Use proposal 2 if it improves fairness
            if (fairness_check2.metrics.gini_index <= fairness_check1.metrics.gini_index or
                fairness_check2.metrics.max_gap < fairness_check1.metrics.max_gap):
                final_plan = proposal2
                final_fairness = fairness_check2
        
        # ========== PHASE 4: DRIVER LIAISON AGENTS (Phase 4.2) ==========
        
        # Build DriverAssignmentProposals with ranking
        sorted_allocations = sorted(
            final_plan.allocation,
            key=lambda x: x.effort,
            reverse=True  # Highest effort = rank 1
        )
        driver_proposals: List[DriverAssignmentProposal] = []
        for rank, alloc_item in enumerate(sorted_allocations, start=1):
            driver_proposals.append(DriverAssignmentProposal(
                driver_id=str(alloc_item.driver_id),
                route_id=str(alloc_item.route_id),
                effort=alloc_item.effort,
                rank_in_team=rank,
            ))
        
        # Build DriverContexts from recent stats (last 7 days)
        driver_contexts: Dict[str, DriverContext] = {}
        cutoff_date = request.allocation_date - timedelta(days=7)
        
        for driver in driver_models:
            driver_id_str = str(driver.id)
            
            # Query recent daily stats
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
                    recent_avg = final_fairness.metrics.avg_effort
                    recent_std = final_fairness.metrics.std_dev
                
                # Count hard days (above avg + std)
                hard_threshold = recent_avg + recent_std
                hard_days = sum(1 for e in recent_efforts if e > hard_threshold)
            else:
                recent_avg = final_fairness.metrics.avg_effort
                recent_std = final_fairness.metrics.std_dev
                hard_days = 0
            
            # Get recent fatigue score from feedback
            feedback_result = await db.execute(
                select(DriverFeedback)
                .where(DriverFeedback.driver_id == driver.id)
                .order_by(DriverFeedback.created_at.desc())
                .limit(1)
            )
            recent_feedback = feedback_result.scalar_one_or_none()
            fatigue_score = float(recent_feedback.tiredness_level) if recent_feedback else 3.0
            fatigue_score = max(1.0, min(5.0, fatigue_score))  # Clamp to 1-5
            
            driver_contexts[driver_id_str] = DriverContext(
                driver_id=driver_id_str,
                recent_avg_effort=recent_avg,
                recent_std_effort=recent_std,
                recent_hard_days=hard_days,
                fatigue_score=fatigue_score,
                preferences={},  # TODO: Pull from driver preferences if available
            )
        
        # Run Driver Liaison Agent
        liaison_agent = DriverLiaisonAgent()
        negotiation_result = liaison_agent.run_for_all_drivers(
            proposals=driver_proposals,
            driver_contexts=driver_contexts,
            effort_matrix=effort_result.matrix,
            driver_ids=effort_result.driver_ids,
            route_ids=effort_result.route_ids,
            global_avg_effort=final_fairness.metrics.avg_effort,
            global_std_effort=final_fairness.metrics.std_dev,
        )
        
        # Log Driver Liaison decisions
        liaison_log = DecisionLog(
            allocation_run_id=allocation_run.id,
            agent_name="DRIVER_LIAISON",
            step_type="NEGOTIATION_DECISIONS",
            input_snapshot=liaison_agent.get_input_snapshot(
                driver_proposals,
                final_fairness.metrics.avg_effort,
                final_fairness.metrics.std_dev,
            ),
            output_snapshot=liaison_agent.get_output_snapshot(negotiation_result),
        )
        db.add(liaison_log)
        
        # ========== PHASE 5: FINAL RESOLUTION (Phase 4.2) ==========
        
        # Check if any COUNTER decisions need resolution
        counter_decisions = [
            d for d in negotiation_result.decisions if d.decision == "COUNTER"
        ]
        
        # Variables for final allocation
        final_allocation = final_plan.allocation
        final_per_driver_effort = final_plan.per_driver_effort
        
        if counter_decisions:
            # Run Final Resolution
            resolution_agent = FinalResolutionAgent()
            resolution_result = resolution_agent.resolve_counters(
                approved_proposal=final_plan,
                decisions=negotiation_result.decisions,
                effort_matrix=effort_result.matrix,
                driver_ids=effort_result.driver_ids,
                route_ids=effort_result.route_ids,
                current_metrics=final_fairness.metrics,
            )
            
            # Log Final Resolution
            resolution_log = DecisionLog(
                allocation_run_id=allocation_run.id,
                agent_name="ROUTE_PLANNER",
                step_type="FINAL_RESOLUTION",
                input_snapshot=resolution_agent.get_input_snapshot(
                    len(counter_decisions),
                    final_fairness.metrics,
                    final_fairness.metrics.avg_effort,
                ),
                output_snapshot=resolution_agent.get_output_snapshot(resolution_result),
            )
            db.add(resolution_log)
            
            # Update final metrics with resolution result
            if resolution_result.swaps_applied:
                # Use resolved allocation
                final_per_driver_effort = resolution_result.per_driver_effort
                # Update allocation_run metrics
                allocation_run.global_gini_index = resolution_result.metrics.get("gini_index", final_fairness.metrics.gini_index)
                allocation_run.global_std_dev = resolution_result.metrics.get("std_dev", final_fairness.metrics.std_dev)
                allocation_run.global_max_gap = resolution_result.metrics.get("max_gap", final_fairness.metrics.max_gap)
        
        # ========== PHASE 6: EXPLAINABILITY AGENT (Phase 4.3) ==========
        
        # Build lookup for route by ID
        route_by_id = {str(r.id): r for r in route_models}
        route_dict_by_id = {}
        for i, r in enumerate(route_models):
            route_dict_by_id[str(r.id)] = route_dicts[i]
        
        driver_by_id = {str(d.id): d for d in driver_models}
        
        # Compute per-driver ranks (1=hardest)
        sorted_efforts = sorted(
            final_per_driver_effort.items(),
            key=lambda x: x[1],
            reverse=True
        )
        rank_by_driver = {did: idx + 1 for idx, (did, _) in enumerate(sorted_efforts)}
        num_drivers = len(final_per_driver_effort)
        
        # Build liaison decisions lookup
        liaison_by_driver = {}
        if 'negotiation_result' in dir():
            for decision in negotiation_result.decisions:
                liaison_by_driver[decision.driver_id] = decision
        
        # Build swaps lookup
        swapped_drivers = set()
        if 'resolution_result' in dir() and resolution_result.swaps_applied:
            for swap in resolution_result.swaps_applied:
                swapped_drivers.add(swap.driver_a)
                swapped_drivers.add(swap.driver_b)
        
        # Initialize ExplainabilityAgent
        explain_agent = ExplainabilityAgent()
        category_counts: Dict[str, int] = {}
        avg_effort = final_fairness.metrics.avg_effort
        
        assignments_response = []
        
        for alloc_item in final_plan.allocation:
            driver_id_str = str(alloc_item.driver_id)
            driver = driver_by_id[driver_id_str]
            route = route_by_id[str(alloc_item.route_id)]
            route_dict = route_dict_by_id[str(alloc_item.route_id)]
            
            # Use resolved effort if available (after swaps), else original
            effort = final_per_driver_effort.get(driver_id_str, alloc_item.effort)
            fairness_score = calculate_fairness_score(effort, avg_effort)
            
            # Get driver context for explanation
            driver_context = driver_contexts.get(driver_id_str)
            history_efforts = []
            history_hard_days = 0
            if driver_context:
                history_efforts = [driver_context.recent_avg_effort] if driver_context.recent_avg_effort else []
                history_hard_days = driver_context.recent_hard_days
            
            # Get effort breakdown from ML agent
            breakdown_key = f"{driver_id_str}:{alloc_item.route_id}"
            effort_breakdown_obj = effort_result.breakdown.get(breakdown_key)
            effort_breakdown = {}
            if effort_breakdown_obj:
                effort_breakdown = {
                    "physical_effort": effort_breakdown_obj.physical_effort,
                    "route_complexity": effort_breakdown_obj.route_complexity,
                    "time_pressure": effort_breakdown_obj.time_pressure,
                }
            
            # Get liaison decision
            liaison_decision = liaison_by_driver.get(driver_id_str)
            
            # Check for manual override
            had_override = False
            try:
                override_result = await db.execute(
                    select(ManualOverride)
                    .where(ManualOverride.allocation_run_id == allocation_run.id)
                    .where(ManualOverride.new_driver_id == driver.id)
                    .limit(1)
                )
                had_override = override_result.scalar_one_or_none() is not None
            except Exception:
                pass  # ManualOverride may not exist yet
            
            # Determine if recovery day
            is_recovery = (
                history_hard_days >= 3 and 
                effort < avg_effort * 0.85
            )
            
            # Build explanation input
            explain_input = DriverExplanationInput(
                driver_id=driver_id_str,
                driver_name=driver.name,
                num_drivers=num_drivers,
                today_effort=effort,
                today_rank=rank_by_driver.get(driver_id_str, num_drivers),
                route_id=str(alloc_item.route_id),
                route_summary={
                    "num_packages": route.num_packages,
                    "total_weight_kg": route.total_weight_kg,
                    "num_stops": route.num_stops,
                    "difficulty_score": route.route_difficulty_score,
                    "estimated_time_minutes": route.estimated_time_minutes,
                },
                effort_breakdown=effort_breakdown,
                global_avg_effort=avg_effort,
                global_std_effort=final_fairness.metrics.std_dev,
                global_gini_index=final_fairness.metrics.gini_index,
                global_max_gap=final_fairness.metrics.max_gap,
                history_efforts_last_7_days=history_efforts,
                history_hard_days_last_7=history_hard_days,
                is_recovery_day=is_recovery,
                had_manual_override=had_override,
                liaison_decision=liaison_decision.decision if liaison_decision else None,
                swap_applied=driver_id_str in swapped_drivers,
            )
            
            # Generate explanations
            explain_output = explain_agent.build_explanation_for_driver(explain_input)
            
            # Track category counts
            category_counts[explain_output.category] = category_counts.get(explain_output.category, 0) + 1
            
            # Create assignment with both explanations
            assignment = Assignment(
                date=request.allocation_date,
                driver_id=driver.id,
                route_id=route.id,
                workload_score=effort,
                fairness_score=fairness_score,
                explanation=explain_output.driver_explanation,  # Legacy field
                driver_explanation=explain_output.driver_explanation,
                admin_explanation=explain_output.admin_explanation,
                allocation_run_id=allocation_run.id,
            )
            db.add(assignment)
            
            # Build response
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
                explanation=explain_output.driver_explanation,
            ))
        
        # Log ExplainabilityAgent step
        explain_log = DecisionLog(
            allocation_run_id=allocation_run.id,
            agent_name="EXPLAINABILITY",
            step_type="EXPLANATIONS_GENERATED",
            input_snapshot=explain_agent.get_input_snapshot(
                num_drivers=num_drivers,
                avg_effort=avg_effort,
                std_effort=final_fairness.metrics.std_dev,
                gini_index=final_fairness.metrics.gini_index,
                category_counts=category_counts,
            ),
            output_snapshot=explain_agent.get_output_snapshot(
                total_explanations=len(assignments_response),
                category_counts=category_counts,
            ),
        )
        db.add(explain_log)
        
        # ========== PHASE 7: UPDATE DAILY STATS (Phase 7) ==========
        from app.services.recovery_service import update_daily_stats_for_run
        
        await update_daily_stats_for_run(
            db=db,
            allocation_run_id=allocation_run.id,
            target_date=request.allocation_date,
            config=active_config,
        )
        
        # ========== PHASE 8: CREATE LEARNING EPISODE ==========
        # Create a learning episode for bandit feedback (reward computed later by cron)
        try:
            learning_agent = LearningAgent(db)
            
            # Build config snapshot for the episode
            config_snapshot = {}
            if active_config:
                config_snapshot = {
                    "gini_threshold": active_config.gini_threshold,
                    "stddev_threshold": active_config.stddev_threshold,
                    "recovery_lightening_factor": active_config.recovery_lightening_factor,
                    "ev_charging_penalty_weight": active_config.ev_charging_penalty_weight,
                    "max_gap_threshold": active_config.max_gap_threshold,
                }
            
            # Determine if this is experimental (10% of runs)
            import random
            is_experimental = random.random() < 0.10
            
            await learning_agent.create_episode(
                allocation_run_id=allocation_run.id,
                fairness_config=config_snapshot,
                num_drivers=len(driver_models),
                num_routes=len(route_models),
                is_experimental=is_experimental,
            )
            
            # Log learning episode creation
            learning_log = DecisionLog(
                allocation_run_id=allocation_run.id,
                agent_name="LEARNING",
                step_type="EPISODE_CREATED",
                input_snapshot={
                    "config_hash": hash_config(config_snapshot),
                    "is_experimental": is_experimental,
                },
                output_snapshot={
                    "status": "pending_reward",
                },
            )
            db.add(learning_log)
        except Exception as learning_error:
            # Learning is non-critical, log but don't fail allocation
            import logging
            logging.warning(f"Failed to create learning episode: {learning_error}")
        
        # ========== FINALIZE ALLOCATION RUN ==========
        allocation_run.global_gini_index = final_fairness.metrics.gini_index
        allocation_run.global_std_dev = final_fairness.metrics.std_dev
        allocation_run.global_max_gap = final_fairness.metrics.max_gap
        allocation_run.status = AllocationRunStatus.SUCCESS
        allocation_run.finished_at = datetime.utcnow()
        
        await db.commit()
        
        return AllocationResponse(
            allocation_run_id=allocation_run.id,
            allocation_date=request.allocation_date,
            global_fairness=GlobalFairness(
                avg_workload=final_fairness.metrics.avg_effort,
                std_dev=final_fairness.metrics.std_dev,
                gini_index=final_fairness.metrics.gini_index,
            ),
            assignments=assignments_response,
        )
        
    except Exception as e:
        # Mark allocation run as failed
        allocation_run.status = AllocationRunStatus.FAILED
        allocation_run.error_message = str(e)[:500]
        allocation_run.finished_at = datetime.utcnow()
        await db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Allocation failed",
                "run_id": str(allocation_run.id),
                "error": str(e)[:200],
            },
        )
