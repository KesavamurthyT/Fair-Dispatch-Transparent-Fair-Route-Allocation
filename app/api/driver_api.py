"""
Phase 2 Driver-facing API endpoints.
Handles driver operations: assignments, stats, deliveries, feedback, swaps, issues.
"""

from datetime import date as date_type

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.database import get_db
from app.schemas.driver_api import (
    TodayAssignmentResponse,
    DriverStatsWindowResponse,
    DeliveryLogRequest,
    DeliveryLogResponse,
    RouteSwapRequestCreate,
    RouteSwapRequestResponse,
    StopIssueRequest,
    StopIssueResponse,
)
from app.services.driver_service import (
    get_today_assignment,
    get_driver_stats,
    log_delivery,
    create_route_swap_request,
    create_stop_issue,
)

router = APIRouter(tags=["Driver"])


@router.get(
    "/assignments/today",
    response_model=TodayAssignmentResponse,
    summary="Get today's assignment",
    description="Fetch the driver's assignment for the given date with full stop details.",
)
async def get_assignment_today(
    driver_id: UUID = Query(..., description="Driver UUID"),
    target_date: date_type = Query(default=None, description="Date (defaults to today)"),
    db: AsyncSession = Depends(get_db),
) -> TodayAssignmentResponse:
    """Get driver's assignment for today or specified date."""
    actual_date = target_date or date_type.today()
    
    result = await get_today_assignment(db, driver_id, actual_date)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No assignment found for driver on given date",
        )
    
    return result


@router.get(
    "/drivers/{driver_id}/stats",
    response_model=DriverStatsWindowResponse,
    summary="Get driver stats",
    description="Get driver statistics over a time window.",
)
async def get_driver_stats_endpoint(
    driver_id: UUID,
    window_days: int = Query(default=7, ge=1, le=90, description="Days to look back"),
    db: AsyncSession = Depends(get_db),
) -> DriverStatsWindowResponse:
    """Get driver stats over time window."""
    result = await get_driver_stats(db, driver_id, window_days)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Driver not found",
        )
    
    return result


@router.post(
    "/deliveries/log",
    response_model=DeliveryLogResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log delivery",
    description="Log a delivery attempt at a stop.",
)
async def log_delivery_endpoint(
    request: DeliveryLogRequest,
    db: AsyncSession = Depends(get_db),
) -> DeliveryLogResponse:
    """Log a delivery at a stop."""
    try:
        result = await log_delivery(
            db=db,
            assignment_id=request.assignment_id,
            route_id=request.route_id,
            driver_id=request.driver_id,
            stop_order=request.stop_order,
            status=request.status,
            issue_type=request.issue_type,
            package_id=request.package_id,
            photo_url=request.photo_url,
            signature_data=request.signature_data,
            notes=request.notes,
        )
        await db.commit()
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/route_swap_requests",
    response_model=RouteSwapRequestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Request route swap",
    description="Submit a route swap request.",
)
async def create_route_swap_endpoint(
    request: RouteSwapRequestCreate,
    db: AsyncSession = Depends(get_db),
) -> RouteSwapRequestResponse:
    """Create a route swap request."""
    try:
        result = await create_route_swap_request(
            db=db,
            from_driver_id=request.from_driver_id,
            assignment_id=request.assignment_id,
            reason=request.reason,
            to_driver_id=request.to_driver_id,
            preferred_date=request.preferred_date,
        )
        await db.commit()
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/stop_issues",
    response_model=StopIssueResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Report stop issue",
    description="Report an issue at a specific stop.",
)
async def create_stop_issue_endpoint(
    request: StopIssueRequest,
    db: AsyncSession = Depends(get_db),
) -> StopIssueResponse:
    """Create a stop issue report."""
    try:
        result = await create_stop_issue(
            db=db,
            assignment_id=request.assignment_id,
            route_id=request.route_id,
            driver_id=request.driver_id,
            stop_order=request.stop_order,
            issue_type=request.issue_type,
            notes=request.notes,
        )
        await db.commit()
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
