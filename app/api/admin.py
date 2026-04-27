"""
Phase 3 Admin-facing API endpoints.
Handles admin operations: health, allocation runs, assignments, metrics, appeals, overrides, config.
"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.database import get_db
from app.schemas.admin import (
    HealthResponse,
    AllocationRunsListResponse,
    AdminAssignmentsListResponse,
    FairnessMetricsResponse,
    WorkloadHeatmapResponse,
    DriverHistoryResponse,
    AppealsListResponse,
    AppealDecisionRequest,
    AppealDecisionResponse,
    ManualOverrideRequest,
    ManualOverrideResponse,
    FairnessConfigRequest,
    FairnessConfigResponse,
    AgentTimelineResponse,
    DriverAllocationStoryResponse,
)
from app.services.admin_service import (
    get_system_health,
    get_allocation_runs,
    get_assignments_paginated,
    get_fairness_metrics_series,
    get_workload_heatmap,
    get_driver_history,
    list_appeals,
    decide_appeal,
    perform_manual_override,
    get_active_fairness_config,
    create_fairness_config,
    get_agent_timeline,
    get_driver_allocation_story,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Get system health status including database and latest allocation run.",
)
async def health_check(
    db: AsyncSession = Depends(get_db),
) -> HealthResponse:
    """Check system health."""
    return await get_system_health(db)


@router.get(
    "/allocation_runs",
    response_model=AllocationRunsListResponse,
    summary="List allocation runs",
    description="Get allocation runs for a specific date.",
)
async def get_allocation_runs_endpoint(
    date: date = Query(..., description="Date to query"),
    db: AsyncSession = Depends(get_db),
) -> AllocationRunsListResponse:
    """List allocation runs for date."""
    return await get_allocation_runs(db, date)


@router.get(
    "/assignments",
    response_model=AdminAssignmentsListResponse,
    summary="List assignments",
    description="Get paginated list of assignments with filters.",
)
async def get_assignments_endpoint(
    date: date = Query(..., description="Date to query"),
    driver_id: UUID = Query(default=None, description="Filter by driver"),
    min_fairness: float = Query(default=None, ge=0, le=1, description="Minimum fairness score"),
    max_fairness: float = Query(default=None, ge=0, le=1, description="Maximum fairness score"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
) -> AdminAssignmentsListResponse:
    """Get paginated assignments."""
    return await get_assignments_paginated(
        db, date, driver_id, min_fairness, max_fairness, page, page_size
    )


@router.get(
    "/metrics/fairness",
    response_model=FairnessMetricsResponse,
    summary="Fairness metrics time series",
    description="Get fairness metrics over a date range.",
)
async def get_fairness_metrics_endpoint(
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    db: AsyncSession = Depends(get_db),
) -> FairnessMetricsResponse:
    """Get fairness metrics time series."""
    return await get_fairness_metrics_series(db, start_date, end_date)


@router.get(
    "/workload_heatmap",
    response_model=WorkloadHeatmapResponse,
    summary="Workload heatmap",
    description="Get workload heatmap data for visualization.",
)
async def get_heatmap_endpoint(
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    db: AsyncSession = Depends(get_db),
) -> WorkloadHeatmapResponse:
    """Get workload heatmap data."""
    return await get_workload_heatmap(db, start_date, end_date)


@router.get(
    "/driver/{driver_id}/history",
    response_model=DriverHistoryResponse,
    summary="Driver history",
    description="Get detailed driver history including appeals and overrides.",
)
async def get_driver_history_endpoint(
    driver_id: UUID,
    window_days: int = Query(default=30, ge=1, le=365, description="Days to look back"),
    db: AsyncSession = Depends(get_db),
) -> DriverHistoryResponse:
    """Get driver history."""
    result = await get_driver_history(db, driver_id, window_days)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Driver not found",
        )
    return result


@router.get(
    "/appeals",
    response_model=AppealsListResponse,
    summary="List appeals",
    description="Get list of appeals with optional status filter.",
)
async def get_appeals_endpoint(
    status: str = Query(default=None, description="Filter by status (PENDING, APPROVED, REJECTED, RESOLVED)"),
    db: AsyncSession = Depends(get_db),
) -> AppealsListResponse:
    """List appeals."""
    return await list_appeals(db, status)


@router.post(
    "/appeals/{appeal_id}/decision",
    response_model=AppealDecisionResponse,
    summary="Decide appeal",
    description="Update appeal status with admin decision.",
)
async def decide_appeal_endpoint(
    appeal_id: UUID,
    request: AppealDecisionRequest,
    db: AsyncSession = Depends(get_db),
) -> AppealDecisionResponse:
    """Make decision on an appeal."""
    try:
        result = await decide_appeal(db, appeal_id, request.status, request.admin_note)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Appeal not found",
            )
        await db.commit()
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/manual_override",
    response_model=ManualOverrideResponse,
    summary="Manual override",
    description="Manually reassign a route from one driver to another.",
)
async def manual_override_endpoint(
    request: ManualOverrideRequest,
    db: AsyncSession = Depends(get_db),
) -> ManualOverrideResponse:
    """Perform manual route override."""
    try:
        result = await perform_manual_override(
            db=db,
            allocation_run_id=request.allocation_run_id,
            old_driver_id=request.old_driver_id,
            new_driver_id=request.new_driver_id,
            route_id=request.route_id,
            reason=request.reason,
        )
        await db.commit()
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/fairness_config",
    response_model=FairnessConfigResponse,
    summary="Get fairness config",
    description="Get the currently active fairness configuration.",
)
async def get_fairness_config_endpoint(
    db: AsyncSession = Depends(get_db),
) -> FairnessConfigResponse:
    """Get active fairness config."""
    result = await get_active_fairness_config(db)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active fairness config found",
        )
    return result


@router.post(
    "/fairness_config",
    response_model=FairnessConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create fairness config",
    description="Create new fairness config and deactivate existing ones.",
)
async def create_fairness_config_endpoint(
    request: FairnessConfigRequest,
    db: AsyncSession = Depends(get_db),
) -> FairnessConfigResponse:
    """Create new fairness config."""
    result = await create_fairness_config(
        db=db,
        workload_weight_packages=request.workload_weight_packages,
        workload_weight_weight_kg=request.workload_weight_weight_kg,
        workload_weight_difficulty=request.workload_weight_difficulty,
        workload_weight_time=request.workload_weight_time,
        gini_threshold=request.gini_threshold,
        stddev_threshold=request.stddev_threshold,
        max_gap_threshold=request.max_gap_threshold,
        recovery_mode_enabled=request.recovery_mode_enabled,
    )
    await db.commit()
    return result


@router.get(
    "/agent_timeline",
    response_model=AgentTimelineResponse,
    summary="Agent timeline",
    description="Get agent decision logs for an allocation run.",
)
async def get_agent_timeline_endpoint(
    allocation_run_id: UUID = Query(..., description="Allocation run ID"),
    db: AsyncSession = Depends(get_db),
) -> AgentTimelineResponse:
    """Get agent timeline for allocation run."""
    return await get_agent_timeline(db, allocation_run_id)


@router.get(
    "/driver_allocation_story",
    response_model=DriverAllocationStoryResponse,
    summary="Driver allocation story",
    description="Get complete allocation story for a driver on a specific date.",
)
async def get_driver_allocation_story_endpoint(
    driver_id: UUID = Query(..., description="Driver ID"),
    date: date = Query(..., description="Date to query (ISO format)"),
    db: AsyncSession = Depends(get_db),
) -> DriverAllocationStoryResponse:
    """Get driver allocation story for a specific date."""
    result = await get_driver_allocation_story(db, driver_id, date)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No assignment found for driver on given date",
        )
    return result

