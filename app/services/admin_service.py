"""
Admin-facing service layer for Phase 3 API endpoints.
Provides business logic for admin operations.
"""

from datetime import date, datetime, timedelta
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, and_, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    Driver, Assignment, Route, DriverFeedback,
    Appeal, AppealStatus,
    ManualOverride,
    FairnessConfig,
    AllocationRun, AllocationRunStatus,
    DecisionLog,
)
from app.schemas.admin import (
    HealthResponse, LatestAllocationRunInfo,
    AllocationRunResponse, AllocationRunsListResponse,
    AdminAssignmentsListResponse, AdminAssignmentResponse,
    AdminDriverInfo, AdminRouteInfo, AdminFeedbackInfo,
    FairnessMetricsResponse, FairnessMetricsPoint,
    WorkloadHeatmapResponse, HeatmapDriver, HeatmapCell,
    DriverHistoryResponse, DriverHistoryDay,
    AppealsListResponse, AppealResponse, AppealDriverInfo, AppealContext,
    AppealDecisionResponse,
    ManualOverrideResponse, ManualOverrideInfo, MetricsSnapshot, UpdatedAssignment,
    FairnessConfigResponse,
    AgentTimelineResponse, DecisionLogStep,
)
from app.services.fairness import calculate_global_fairness


async def get_system_health(db: AsyncSession) -> HealthResponse:
    """
    Get system health status including latest allocation run.
    """
    # Check database connection
    try:
        await db.execute(select(func.now()))
        db_status = "up"
    except Exception:
        db_status = "down"
    
    # Get latest allocation run
    result = await db.execute(
        select(AllocationRun)
        .order_by(AllocationRun.started_at.desc())
        .limit(1)
    )
    latest_run = result.scalar_one_or_none()
    
    latest_run_info = None
    if latest_run:
        latest_run_info = LatestAllocationRunInfo(
            id=latest_run.id,
            run_date=latest_run.date,
            status=latest_run.status.value,
            started_at=latest_run.started_at,
            finished_at=latest_run.finished_at,
        )
    
    return HealthResponse(
        status="ok" if db_status == "up" else "degraded",
        database=db_status,
        latest_allocation_run=latest_run_info,
    )


async def get_allocation_runs(
    db: AsyncSession,
    target_date: date,
) -> AllocationRunsListResponse:
    """
    Get allocation runs for a specific date.
    """
    result = await db.execute(
        select(AllocationRun)
        .where(AllocationRun.date == target_date)
        .order_by(AllocationRun.started_at.desc())
    )
    runs = result.scalars().all()
    
    return AllocationRunsListResponse(
        runs=[
            AllocationRunResponse(
                id=run.id,
                run_date=run.date,
                num_drivers=run.num_drivers,
                num_routes=run.num_routes,
                num_packages=run.num_packages,
                global_gini_index=run.global_gini_index,
                global_std_dev=run.global_std_dev,
                global_max_gap=run.global_max_gap,
                status=run.status.value,
                error_message=run.error_message,
                started_at=run.started_at,
                finished_at=run.finished_at,
            )
            for run in runs
        ]
    )


async def get_assignments_paginated(
    db: AsyncSession,
    target_date: date,
    driver_id: Optional[UUID] = None,
    min_fairness: Optional[float] = None,
    max_fairness: Optional[float] = None,
    page: int = 1,
    page_size: int = 50,
) -> AdminAssignmentsListResponse:
    """
    Get paginated assignments with filters.
    """
    # Build query
    query = select(Assignment, Driver, Route).join(
        Driver, Assignment.driver_id == Driver.id
    ).join(
        Route, Assignment.route_id == Route.id
    ).where(
        Assignment.date == target_date
    )
    
    if driver_id:
        query = query.where(Assignment.driver_id == driver_id)
    if min_fairness is not None:
        query = query.where(Assignment.fairness_score >= min_fairness)
    if max_fairness is not None:
        query = query.where(Assignment.fairness_score <= max_fairness)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total_items = total_result.scalar() or 0
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(Assignment.created_at)
    
    result = await db.execute(query)
    rows = result.all()
    
    items = []
    for assignment, driver, route in rows:
        # Get feedback if exists
        feedback_result = await db.execute(
            select(DriverFeedback)
            .where(DriverFeedback.assignment_id == assignment.id)
            .limit(1)
        )
        feedback = feedback_result.scalar_one_or_none()
        
        items.append(AdminAssignmentResponse(
            assignment_id=assignment.id,
            allocation_run_id=assignment.allocation_run_id,
            driver=AdminDriverInfo(
                id=driver.id,
                name=driver.name,
                vehicle_type=driver.vehicle_type.value if driver.vehicle_type else None,
            ),
            route=AdminRouteInfo(
                id=route.id,
                num_packages=route.num_packages,
                total_weight_kg=route.total_weight_kg,
                num_stops=route.num_stops,
                route_difficulty_score=route.route_difficulty_score,
                estimated_time_minutes=route.estimated_time_minutes,
            ),
            workload_score=assignment.workload_score,
            fairness_score=assignment.fairness_score,
            explanation=assignment.explanation,
            feedback=AdminFeedbackInfo(
                fairness_rating=feedback.fairness_rating if feedback else None,
                stress_level=feedback.stress_level if feedback else None,
            ) if feedback else None,
        ))
    
    return AdminAssignmentsListResponse(
        items=items,
        page=page,
        page_size=page_size,
        total_items=total_items,
    )


async def get_fairness_metrics_series(
    db: AsyncSession,
    start_date: date,
    end_date: date,
) -> FairnessMetricsResponse:
    """
    Get fairness metrics time series.
    """
    # Get allocation runs in range
    result = await db.execute(
        select(AllocationRun)
        .where(
            and_(
                AllocationRun.date >= start_date,
                AllocationRun.date <= end_date,
                AllocationRun.status == AllocationRunStatus.SUCCESS,
            )
        )
        .order_by(AllocationRun.date)
    )
    runs = result.scalars().all()
    
    points = []
    for run in runs:
        # Get appeals count for date
        appeals_result = await db.execute(
            select(func.count(Appeal.id))
            .join(Assignment, Appeal.assignment_id == Assignment.id)
            .where(Assignment.date == run.date)
        )
        appeals_count = appeals_result.scalar() or 0
        
        # Calculate outliers (assignments > 1.5 std dev from mean)
        # For now, use a simplified calculation
        outlier_count = 0
        if run.global_std_dev > 0:
            assignments_result = await db.execute(
                select(Assignment.workload_score)
                .where(Assignment.allocation_run_id == run.id)
            )
            workloads = [w for (w,) in assignments_result.all()]
            if workloads:
                avg = sum(workloads) / len(workloads)
                threshold = run.global_std_dev * 1.5
                outlier_count = sum(1 for w in workloads if abs(w - avg) > threshold)
        
        points.append(FairnessMetricsPoint(
            point_date=run.date,
            gini_index=run.global_gini_index,
            std_dev=run.global_std_dev,
            max_gap=run.global_max_gap,
            outlier_count=outlier_count,
            appeals_count=appeals_count,
        ))
    
    return FairnessMetricsResponse(points=points)


async def get_workload_heatmap(
    db: AsyncSession,
    start_date: date,
    end_date: date,
) -> WorkloadHeatmapResponse:
    """
    Get workload heatmap data.
    """
    # Get all assignments in range
    result = await db.execute(
        select(Assignment, Driver)
        .join(Driver, Assignment.driver_id == Driver.id)
        .where(
            and_(
                Assignment.date >= start_date,
                Assignment.date <= end_date,
            )
        )
        .order_by(Assignment.date, Driver.name)
    )
    rows = result.all()
    
    # Collect unique drivers and dates
    drivers_map = {}
    dates_set = set()
    cells = []
    
    for assignment, driver in rows:
        if driver.id not in drivers_map:
            drivers_map[driver.id] = HeatmapDriver(id=driver.id, name=driver.name)
        dates_set.add(assignment.date)
        cells.append(HeatmapCell(
            driver_id=driver.id,
            cell_date=assignment.date,
            workload_score=assignment.workload_score,
            fairness_score=assignment.fairness_score,
        ))
    
    return WorkloadHeatmapResponse(
        drivers=list(drivers_map.values()),
        dates=sorted(dates_set),
        cells=cells,
    )


async def get_driver_history(
    db: AsyncSession,
    driver_id: UUID,
    window_days: int = 30,
) -> Optional[DriverHistoryResponse]:
    """
    Get detailed driver history including appeals and overrides.
    """
    # Verify driver exists
    driver_result = await db.execute(
        select(Driver).where(Driver.id == driver_id)
    )
    driver = driver_result.scalar_one_or_none()
    if not driver:
        return None
    
    end_date = date.today()
    start_date = end_date - timedelta(days=window_days)
    
    # Get assignments with feedback
    assignments_result = await db.execute(
        select(Assignment, DriverFeedback)
        .outerjoin(
            DriverFeedback,
            and_(
                DriverFeedback.assignment_id == Assignment.id,
                DriverFeedback.driver_id == Assignment.driver_id,
            )
        )
        .where(
            and_(
                Assignment.driver_id == driver_id,
                Assignment.date >= start_date,
                Assignment.date <= end_date,
            )
        )
        .order_by(Assignment.date)
    )
    
    days = []
    for assignment, feedback in assignments_result.all():
        # Count appeals for this assignment
        appeals_result = await db.execute(
            select(func.count(Appeal.id))
            .where(Appeal.assignment_id == assignment.id)
        )
        appeals_count = appeals_result.scalar() or 0
        
        # Count overrides affecting this driver
        overrides_result = await db.execute(
            select(func.count(ManualOverride.id))
            .where(
                and_(
                    ManualOverride.allocation_run_id == assignment.allocation_run_id,
                    (ManualOverride.old_driver_id == driver_id) | 
                    (ManualOverride.new_driver_id == driver_id),
                )
            )
        )
        overrides_count = overrides_result.scalar() or 0
        
        days.append(DriverHistoryDay(
            day_date=assignment.date,
            workload_score=assignment.workload_score,
            fairness_score=assignment.fairness_score,
            reported_stress_level=float(feedback.stress_level) if feedback else None,
            reported_fairness_rating=feedback.fairness_rating if feedback else None,
            appeals_count=appeals_count,
            manual_overrides_affecting_driver=overrides_count,
        ))
    
    return DriverHistoryResponse(
        driver_id=driver_id,
        window_days=window_days,
        days=days,
    )


async def list_appeals(
    db: AsyncSession,
    status_filter: Optional[str] = None,
) -> AppealsListResponse:
    """
    List appeals with optional status filter.
    """
    query = select(Appeal, Driver, Assignment).join(
        Driver, Appeal.driver_id == Driver.id
    ).join(
        Assignment, Appeal.assignment_id == Assignment.id
    )
    
    if status_filter:
        try:
            status = AppealStatus(status_filter)
            query = query.where(Appeal.status == status)
        except ValueError:
            pass  # Invalid status, ignore filter
    
    query = query.order_by(Appeal.created_at.desc())
    result = await db.execute(query)
    rows = result.all()
    
    items = []
    for appeal, driver, assignment in rows:
        items.append(AppealResponse(
            id=appeal.id,
            driver=AppealDriverInfo(id=driver.id, name=driver.name),
            assignment_id=appeal.assignment_id,
            appeal_date=assignment.date,
            reason=appeal.reason,
            status=appeal.status.value,
            admin_note=appeal.admin_note,
            created_at=appeal.created_at,
            updated_at=appeal.updated_at,
            context=AppealContext(
                workload_score=assignment.workload_score,
                fairness_score=assignment.fairness_score,
                recent_streak_hard_days=None,  # Would need more complex query
            ),
        ))
    
    return AppealsListResponse(items=items)


async def decide_appeal(
    db: AsyncSession,
    appeal_id: UUID,
    new_status: str,
    admin_note: Optional[str] = None,
) -> Optional[AppealDecisionResponse]:
    """
    Update appeal status.
    """
    # Get appeal
    result = await db.execute(
        select(Appeal).where(Appeal.id == appeal_id)
    )
    appeal = result.scalar_one_or_none()
    if not appeal:
        return None
    
    # Parse status
    try:
        status = AppealStatus(new_status)
    except ValueError:
        raise ValueError(f"Invalid status: {new_status}")
    
    # Update
    appeal.status = status
    appeal.admin_note = admin_note
    appeal.updated_at = datetime.utcnow()
    
    await db.flush()
    await db.refresh(appeal)
    
    return AppealDecisionResponse(
        id=appeal.id,
        status=appeal.status.value,
        admin_note=appeal.admin_note,
        updated_at=appeal.updated_at,
    )


async def perform_manual_override(
    db: AsyncSession,
    allocation_run_id: UUID,
    old_driver_id: UUID,
    new_driver_id: UUID,
    route_id: UUID,
    reason: Optional[str] = None,
) -> ManualOverrideResponse:
    """
    Perform manual override of route assignment.
    """
    # Get current assignment
    old_assignment_result = await db.execute(
        select(Assignment).where(
            and_(
                Assignment.allocation_run_id == allocation_run_id,
                Assignment.driver_id == old_driver_id,
                Assignment.route_id == route_id,
            )
        )
    )
    old_assignment = old_assignment_result.scalar_one_or_none()
    if not old_assignment:
        raise ValueError("Original assignment not found")
    
    # Get all workloads before override
    before_result = await db.execute(
        select(Assignment.workload_score)
        .where(Assignment.allocation_run_id == allocation_run_id)
    )
    before_workloads = [w for (w,) in before_result.all()]
    before_metrics = calculate_global_fairness(before_workloads)
    before_max_gap = max(before_workloads) - min(before_workloads) if before_workloads else 0
    
    # Update assignment to new driver
    old_assignment.driver_id = new_driver_id
    await db.flush()
    
    # Get workloads after override
    after_result = await db.execute(
        select(Assignment.workload_score)
        .where(Assignment.allocation_run_id == allocation_run_id)
    )
    after_workloads = [w for (w,) in after_result.all()]
    after_metrics = calculate_global_fairness(after_workloads)
    after_max_gap = max(after_workloads) - min(after_workloads) if after_workloads else 0
    
    # Create override record
    override = ManualOverride(
        allocation_run_id=allocation_run_id,
        old_driver_id=old_driver_id,
        new_driver_id=new_driver_id,
        route_id=route_id,
        reason=reason,
        before_metrics={
            "gini_index": before_metrics.gini_index,
            "std_dev": before_metrics.std_dev,
            "max_gap": before_max_gap,
        },
        after_metrics={
            "gini_index": after_metrics.gini_index,
            "std_dev": after_metrics.std_dev,
            "max_gap": after_max_gap,
        },
    )
    db.add(override)
    await db.flush()
    await db.refresh(override)
    
    return ManualOverrideResponse(
        manual_override=ManualOverrideInfo(
            id=override.id,
            allocation_run_id=override.allocation_run_id,
            old_driver_id=override.old_driver_id,
            new_driver_id=override.new_driver_id,
            route_id=override.route_id,
            reason=override.reason,
            before_metrics=MetricsSnapshot(
                gini_index=before_metrics.gini_index,
                std_dev=before_metrics.std_dev,
                max_gap=before_max_gap,
            ),
            after_metrics=MetricsSnapshot(
                gini_index=after_metrics.gini_index,
                std_dev=after_metrics.std_dev,
                max_gap=after_max_gap,
            ),
            created_at=override.created_at,
        ),
        updated_assignments=[
            UpdatedAssignment(
                assignment_id=old_assignment.id,
                driver_id=new_driver_id,
                route_id=route_id,
            ),
        ],
    )


async def get_active_fairness_config(db: AsyncSession) -> Optional[FairnessConfigResponse]:
    """
    Get the currently active fairness config.
    """
    result = await db.execute(
        select(FairnessConfig)
        .where(FairnessConfig.is_active == True)
        .order_by(FairnessConfig.created_at.desc())
        .limit(1)
    )
    config = result.scalar_one_or_none()
    
    if not config:
        return None
    
    return FairnessConfigResponse(
        id=config.id,
        is_active=config.is_active,
        workload_weight_packages=config.workload_weight_packages,
        workload_weight_weight_kg=config.workload_weight_weight_kg,
        workload_weight_difficulty=config.workload_weight_difficulty,
        workload_weight_time=config.workload_weight_time,
        gini_threshold=config.gini_threshold,
        stddev_threshold=config.stddev_threshold,
        max_gap_threshold=config.max_gap_threshold,
        recovery_mode_enabled=config.recovery_mode_enabled,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


async def create_fairness_config(
    db: AsyncSession,
    workload_weight_packages: float,
    workload_weight_weight_kg: float,
    workload_weight_difficulty: float,
    workload_weight_time: float,
    gini_threshold: float,
    stddev_threshold: float,
    max_gap_threshold: float,
    recovery_mode_enabled: bool,
) -> FairnessConfigResponse:
    """
    Create new fairness config and deactivate old ones.
    """
    # Deactivate existing configs
    await db.execute(
        update(FairnessConfig)
        .where(FairnessConfig.is_active == True)
        .values(is_active=False)
    )
    
    # Create new config
    config = FairnessConfig(
        is_active=True,
        workload_weight_packages=workload_weight_packages,
        workload_weight_weight_kg=workload_weight_weight_kg,
        workload_weight_difficulty=workload_weight_difficulty,
        workload_weight_time=workload_weight_time,
        gini_threshold=gini_threshold,
        stddev_threshold=stddev_threshold,
        max_gap_threshold=max_gap_threshold,
        recovery_mode_enabled=recovery_mode_enabled,
    )
    db.add(config)
    await db.flush()
    await db.refresh(config)
    
    return FairnessConfigResponse(
        id=config.id,
        is_active=config.is_active,
        workload_weight_packages=config.workload_weight_packages,
        workload_weight_weight_kg=config.workload_weight_weight_kg,
        workload_weight_difficulty=config.workload_weight_difficulty,
        workload_weight_time=config.workload_weight_time,
        gini_threshold=config.gini_threshold,
        stddev_threshold=config.stddev_threshold,
        max_gap_threshold=config.max_gap_threshold,
        recovery_mode_enabled=config.recovery_mode_enabled,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


async def get_agent_timeline(
    db: AsyncSession,
    allocation_run_id: UUID,
) -> AgentTimelineResponse:
    """
    Get enhanced agent decision timeline for an allocation run.
    """
    from app.schemas.admin import (
        AllocationRunInfo, AgentTimelineEvent,
    )
    
    # Fetch allocation run
    run_result = await db.execute(
        select(AllocationRun).where(AllocationRun.id == allocation_run_id)
    )
    allocation_run = run_result.scalar_one_or_none()
    
    if not allocation_run:
        # Return empty timeline if run not found
        return AgentTimelineResponse(
            allocation_run=AllocationRunInfo(
                id=allocation_run_id,
                date=date.today(),
                num_drivers=0,
                num_routes=0,
                num_packages=0,
                global_metrics={},
                status="NOT_FOUND",
                started_at=datetime.utcnow(),
            ),
            timeline=[],
            allocation_run_id=allocation_run_id,
            steps=[],
        )
    
    # Calculate duration
    duration = None
    if allocation_run.finished_at and allocation_run.started_at:
        duration = (allocation_run.finished_at - allocation_run.started_at).total_seconds()
    
    # Compute avg_effort from assignments if not stored
    avg_effort = 0.0
    assignments_result = await db.execute(
        select(func.avg(Assignment.workload_score))
        .where(Assignment.allocation_run_id == allocation_run_id)
    )
    avg_val = assignments_result.scalar()
    if avg_val:
        avg_effort = float(avg_val)
    
    run_info = AllocationRunInfo(
        id=allocation_run.id,
        date=allocation_run.date,
        num_drivers=allocation_run.num_drivers,
        num_routes=allocation_run.num_routes,
        num_packages=allocation_run.num_packages,
        global_metrics={
            "gini_index": allocation_run.global_gini_index,
            "std_dev": allocation_run.global_std_dev,
            "max_gap": allocation_run.global_max_gap,
            "avg_effort": round(avg_effort, 2),
        },
        status=allocation_run.status.value,
        started_at=allocation_run.started_at,
        finished_at=allocation_run.finished_at,
        duration_seconds=duration,
    )
    
    # Fetch decision logs
    result = await db.execute(
        select(DecisionLog)
        .where(DecisionLog.allocation_run_id == allocation_run_id)
        .order_by(DecisionLog.created_at, DecisionLog.id)
    )
    logs = result.scalars().all()
    
    # Build timeline events with short messages
    timeline = []
    for log in logs:
        short_message = _generate_short_message(log)
        details = _extract_details(log)
        
        timeline.append(AgentTimelineEvent(
            id=log.id,
            timestamp=log.created_at,
            agent_name=log.agent_name,
            step_type=log.step_type,
            short_message=short_message,
            details=details,
        ))
    
    # Also build legacy steps for backward compatibility
    legacy_steps = [
        DecisionLogStep(
            id=log.id,
            agent_name=log.agent_name,
            step_type=log.step_type,
            input_snapshot=log.input_snapshot,
            output_snapshot=log.output_snapshot,
            created_at=log.created_at,
        )
        for log in logs
    ]
    
    return AgentTimelineResponse(
        allocation_run=run_info,
        timeline=timeline,
        allocation_run_id=allocation_run_id,
        steps=legacy_steps,
    )


def _generate_short_message(log: DecisionLog) -> str:
    """Generate a human-readable short message for a decision log entry."""
    agent = log.agent_name
    step = log.step_type
    inp = log.input_snapshot or {}
    out = log.output_snapshot or {}
    
    if agent == "ML_EFFORT":
        num_d = inp.get("num_drivers", out.get("num_drivers", "?"))
        num_r = inp.get("num_routes", out.get("num_routes", "?"))
        return f"Computed effort matrix for {num_d} drivers × {num_r} routes"
    
    if agent == "ROUTE_PLANNER":
        if step == "PROPOSAL_1":
            return "Generated initial route assignment proposal"
        if step == "PROPOSAL_2":
            return "Generated re-optimized proposal with fairness penalties"
        if step == "FINAL_RESOLUTION":
            swaps = out.get("swaps_applied", out.get("num_swaps", 0))
            return f"Applied {swaps} swaps after negotiation"
    
    if agent == "FAIRNESS_MANAGER":
        status = out.get("status", "UNKNOWN")
        if status == "REOPTIMIZE":
            return "Fairness check requested re-optimization"
        return "Fairness check accepted proposal"
    
    if agent == "DRIVER_LIAISON":
        accept = out.get("num_accept", 0)
        counter = out.get("num_counter", 0)
        force = out.get("num_force_accept", 0)
        return f"Drivers: {accept} ACCEPT, {counter} COUNTER, {force} FORCE_ACCEPT"
    
    if agent == "EXPLAINABILITY":
        total = out.get("total_explanations", "?")
        cats = out.get("category_counts", {})
        num_cats = len(cats)
        return f"Generated {total} explanations in {num_cats} categories"
    
    return f"{agent}: {step}"


def _extract_details(log: DecisionLog) -> dict:
    """Extract relevant details from log snapshots."""
    inp = log.input_snapshot or {}
    out = log.output_snapshot or {}
    details = {}
    
    # Whitelist relevant keys
    relevant_keys = {
        "num_drivers", "num_routes", "min_effort", "max_effort", "avg_effort",
        "total_effort", "gini_index", "std_dev", "max_gap", "status",
        "num_accept", "num_counter", "num_force_accept",
        "swaps_applied", "unfulfilled_counters",
        "total_explanations", "category_counts",
        "final_gini_index", "final_std_dev", "final_max_gap",
        "matrix_shape", "num_packages",
    }
    
    for key in relevant_keys:
        if key in out:
            details[key] = out[key]
        elif key in inp:
            details[key] = inp[key]
    
    return details


async def get_driver_allocation_story(
    db: AsyncSession,
    driver_id: UUID,
    target_date: date,
) -> Optional["DriverAllocationStoryResponse"]:
    """
    Get the complete allocation story for a driver on a specific date.
    """
    from app.schemas.admin import (
        DriverAllocationStoryResponse,
        StoryDriverInfo, StoryRouteInfo, StoryRouteSummary,
        StoryEffortInfo, StoryTodayInfo,
        StoryHistoryDay, StoryRecoveryInfo,
        StoryNegotiationInfo, StoryManualOverride, StorySwapDetails,
        StoryTimelineEvent, StoryGlobalMetrics, StoryAllocationRun,
    )
    
    # Get assignment for driver on date
    assignment_result = await db.execute(
        select(Assignment, Driver, Route)
        .join(Driver, Assignment.driver_id == Driver.id)
        .join(Route, Assignment.route_id == Route.id)
        .where(
            and_(
                Assignment.driver_id == driver_id,
                Assignment.date == target_date,
            )
        )
    )
    row = assignment_result.first()
    
    if not row:
        return None
    
    assignment, driver, route = row
    
    # Get allocation run
    run_result = await db.execute(
        select(AllocationRun).where(AllocationRun.id == assignment.allocation_run_id)
    )
    allocation_run = run_result.scalar_one_or_none()
    
    if not allocation_run:
        return None
    
    # Compute avg effort and rank
    all_assignments_result = await db.execute(
        select(Assignment.driver_id, Assignment.workload_score)
        .where(Assignment.allocation_run_id == assignment.allocation_run_id)
    )
    all_efforts = list(all_assignments_result.all())
    
    efforts_by_driver = {str(did): score for did, score in all_efforts}
    sorted_efforts = sorted(efforts_by_driver.items(), key=lambda x: x[1], reverse=True)
    
    rank = 1
    for idx, (did, _) in enumerate(sorted_efforts):
        if did == str(driver_id):
            rank = idx + 1
            break
    
    num_drivers = len(all_efforts)
    avg_effort = sum(e for _, e in all_efforts) / max(num_drivers, 1)
    percent_vs_avg = ((assignment.workload_score - avg_effort) / max(avg_effort, 1)) * 100
    
    # Build today info
    today_info = StoryTodayInfo(
        assignment_id=assignment.id,
        route=StoryRouteInfo(
            id=route.id,
            summary=StoryRouteSummary(
                num_packages=route.num_packages,
                total_weight_kg=route.total_weight_kg,
                num_stops=route.num_stops,
                route_difficulty_score=route.route_difficulty_score,
                estimated_time_minutes=route.estimated_time_minutes,
            ),
        ),
        effort=StoryEffortInfo(
            value=assignment.workload_score,
            rank=rank,
            num_drivers=num_drivers,
            percent_vs_avg=round(percent_vs_avg, 1),
        ),
        fairness_score=assignment.fairness_score,
        driver_explanation=getattr(assignment, 'driver_explanation', None) or assignment.explanation,
        admin_explanation=getattr(assignment, 'admin_explanation', None),
        explainability_category=None,  # Would need to extract from DecisionLog
    )
    
    # Get 7-day history
    history_start = target_date - timedelta(days=7)
    history_result = await db.execute(
        select(Assignment, DriverFeedback)
        .outerjoin(
            DriverFeedback,
            and_(
                DriverFeedback.assignment_id == Assignment.id,
                DriverFeedback.driver_id == Assignment.driver_id,
            )
        )
        .where(
            and_(
                Assignment.driver_id == driver_id,
                Assignment.date >= history_start,
                Assignment.date < target_date,
            )
        )
        .order_by(Assignment.date)
    )
    
    history_days = []
    hard_day_count = 0
    
    for hist_assignment, feedback in history_result.all():
        # Determine tag based on effort relative to avg
        if hist_assignment.workload_score > avg_effort * 1.15:
            tag = "HARD"
            hard_day_count += 1
        elif hist_assignment.workload_score < avg_effort * 0.85:
            tag = "LIGHT"
        else:
            tag = "NORMAL"
        
        history_days.append(StoryHistoryDay(
            date=hist_assignment.date,
            effort=hist_assignment.workload_score,
            fairness_score=hist_assignment.fairness_score,
            stress_level=feedback.stress_level if feedback else None,
            fairness_rating=feedback.fairness_rating if feedback else None,
            tag=tag,
        ))
    
    # Determine if recovery day
    is_recovery = (
        hard_day_count >= 2 and
        assignment.workload_score < avg_effort * 0.9
    )
    
    recovery_info = StoryRecoveryInfo(
        is_recovery_day=is_recovery,
        recent_hard_days=hard_day_count,
    )
    
    # Get negotiation info from DecisionLog
    liaison_decision = None
    swap_applied = False
    
    liaison_log_result = await db.execute(
        select(DecisionLog)
        .where(
            and_(
                DecisionLog.allocation_run_id == assignment.allocation_run_id,
                DecisionLog.agent_name == "DRIVER_LIAISON",
            )
        )
        .limit(1)
    )
    liaison_log = liaison_log_result.scalar_one_or_none()
    
    # Check for swaps
    resolution_log_result = await db.execute(
        select(DecisionLog)
        .where(
            and_(
                DecisionLog.allocation_run_id == assignment.allocation_run_id,
                DecisionLog.agent_name == "ROUTE_PLANNER",
                DecisionLog.step_type == "FINAL_RESOLUTION",
            )
        )
        .limit(1)
    )
    resolution_log = resolution_log_result.scalar_one_or_none()
    
    if resolution_log and resolution_log.output_snapshot:
        swaps = resolution_log.output_snapshot.get("swaps_applied", [])
        if isinstance(swaps, list):
            for swap in swaps:
                if str(driver_id) in [swap.get("driver_a"), swap.get("driver_b")]:
                    swap_applied = True
                    break
        elif isinstance(swaps, int) and swaps > 0:
            # Can't determine per-driver, assume no swap for this driver
            pass
    
    # Check for manual override
    override_result = await db.execute(
        select(ManualOverride)
        .where(
            and_(
                ManualOverride.allocation_run_id == assignment.allocation_run_id,
                (ManualOverride.old_driver_id == driver_id) |
                (ManualOverride.new_driver_id == driver_id),
            )
        )
        .limit(1)
    )
    override = override_result.scalar_one_or_none()
    
    negotiation_info = StoryNegotiationInfo(
        liaison_decision=liaison_decision,
        liaison_reason=None,
        swap_applied=swap_applied,
        swap_details=None,
        manual_override=StoryManualOverride(
            affected=override is not None,
            details=override.reason if override else None,
        ),
    )
    
    # Build agent timeline slice
    timeline_slice = []
    
    all_logs_result = await db.execute(
        select(DecisionLog)
        .where(DecisionLog.allocation_run_id == assignment.allocation_run_id)
        .order_by(DecisionLog.created_at)
    )
    all_logs = all_logs_result.scalars().all()
    
    for log in all_logs:
        description = _generate_driver_specific_description(log, driver_id, driver.name)
        timeline_slice.append(StoryTimelineEvent(
            timestamp=log.created_at,
            agent_name=log.agent_name,
            step_type=log.step_type,
            description=description,
        ))
    
    return DriverAllocationStoryResponse(
        driver=StoryDriverInfo(id=driver.id, name=driver.name),
        date=target_date,
        allocation_run=StoryAllocationRun(
            id=allocation_run.id,
            global_metrics=StoryGlobalMetrics(
                gini_index=allocation_run.global_gini_index,
                std_dev=allocation_run.global_std_dev,
                max_gap=allocation_run.global_max_gap,
                avg_effort=round(avg_effort, 1),
            ),
        ),
        today=today_info,
        history_last_7_days=history_days,
        recovery=recovery_info,
        negotiation=negotiation_info,
        agent_timeline_slice=timeline_slice,
    )


def _generate_driver_specific_description(log: DecisionLog, driver_id: UUID, driver_name: str) -> str:
    """Generate driver-specific description for timeline event."""
    agent = log.agent_name
    step = log.step_type
    out = log.output_snapshot or {}
    
    if agent == "ML_EFFORT":
        return "Effort matrix computed including this driver"
    
    if agent == "ROUTE_PLANNER":
        if step == "PROPOSAL_1":
            return "Driver included in initial route proposal"
        if step == "PROPOSAL_2":
            return "Driver's assignment adjusted in re-optimized proposal"
        if step == "FINAL_RESOLUTION":
            return "Final route assignments determined after negotiation"
    
    if agent == "FAIRNESS_MANAGER":
        status = out.get("status", "UNKNOWN")
        if status == "REOPTIMIZE":
            return "Fairness check triggered re-optimization affecting all drivers"
        return "Fairness check passed for current assignments"
    
    if agent == "DRIVER_LIAISON":
        return "Negotiation decisions processed for all drivers"
    
    if agent == "EXPLAINABILITY":
        return "Explanation generated for this driver's assignment"
    
    return f"{agent}: {step}"

