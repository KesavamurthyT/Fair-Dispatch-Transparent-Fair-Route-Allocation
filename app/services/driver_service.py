"""
Driver-facing service layer for Phase 2 API endpoints.
Provides business logic for driver operations.
"""

from datetime import date, timedelta
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    Driver, Assignment, Route, RoutePackage, Package, DriverFeedback,
    DeliveryLog, DeliveryStatus, DeliveryIssueType,
    RouteSwapRequest, SwapRequestStatus,
    StopIssue, StopIssueType,
)
from app.schemas.driver_api import (
    TodayAssignmentResponse, DriverDetail, AssignmentDetail,
    RouteSummaryDetail, StopDetail, PackageDetail,
    DriverStatsWindowResponse, DayStats, StatsAggregates,
    DeliveryLogResponse,
    RouteSwapRequestResponse,
    StopIssueResponse,
)


async def get_today_assignment(
    db: AsyncSession,
    driver_id: UUID,
    target_date: date,
) -> Optional[TodayAssignmentResponse]:
    """
    Fetch the driver's assignment for a given date with full stop details.
    
    Args:
        db: Database session
        driver_id: Driver UUID
        target_date: Date to fetch assignment for
    
    Returns:
        TodayAssignmentResponse if found, None otherwise
    """
    # Get driver
    driver_result = await db.execute(
        select(Driver).where(Driver.id == driver_id)
    )
    driver = driver_result.scalar_one_or_none()
    if not driver:
        return None
    
    # Get latest assignment for driver on date
    assignment_result = await db.execute(
        select(Assignment)
        .where(
            and_(
                Assignment.driver_id == driver_id,
                Assignment.date == target_date,
            )
        )
        .order_by(Assignment.created_at.desc())
        .limit(1)
    )
    assignment = assignment_result.scalar_one_or_none()
    if not assignment:
        return None
    
    # Get route
    route_result = await db.execute(
        select(Route).where(Route.id == assignment.route_id)
    )
    route = route_result.scalar_one_or_none()
    if not route:
        return None
    
    # Get packages with stop order
    route_packages_result = await db.execute(
        select(RoutePackage, Package)
        .join(Package, RoutePackage.package_id == Package.id)
        .where(RoutePackage.route_id == route.id)
        .order_by(RoutePackage.stop_order)
    )
    route_packages = route_packages_result.all()
    
    # Group packages by stop (address)
    stops_map: dict[str, StopDetail] = {}
    for rp, pkg in route_packages:
        addr_key = pkg.address.strip().lower()
        if addr_key not in stops_map:
            stops_map[addr_key] = StopDetail(
                stop_order=rp.stop_order,
                address=pkg.address,
                latitude=pkg.latitude,
                longitude=pkg.longitude,
                packages=[],
                building_type=None,
                floor_number=None,
                stairs_likelihood=None,
            )
        stops_map[addr_key].packages.append(PackageDetail(
            id=pkg.id,
            external_id=pkg.external_id,
            weight_kg=pkg.weight_kg,
            fragility_level=pkg.fragility_level,
            priority=pkg.priority.value,
        ))
    
    # Order stops by stop_order
    stops = sorted(stops_map.values(), key=lambda s: s.stop_order)
    
    return TodayAssignmentResponse(
        assignment_date=target_date,
        driver=DriverDetail(
            id=driver.id,
            external_id=driver.external_id,
            name=driver.name,
            preferred_language=driver.preferred_language.value,
        ),
        assignment=AssignmentDetail(
            assignment_id=assignment.id,
            route_id=route.id,
            workload_score=assignment.workload_score,
            fairness_score=assignment.fairness_score,
            explanation=assignment.explanation or "",
            route_summary=RouteSummaryDetail(
                num_packages=route.num_packages,
                total_weight_kg=route.total_weight_kg,
                num_stops=route.num_stops,
                route_difficulty_score=route.route_difficulty_score,
                estimated_time_minutes=route.estimated_time_minutes,
            ),
            stops=stops,
        ),
    )


async def get_driver_stats(
    db: AsyncSession,
    driver_id: UUID,
    window_days: int = 7,
) -> Optional[DriverStatsWindowResponse]:
    """
    Get driver stats over a time window.
    
    Args:
        db: Database session
        driver_id: Driver UUID
        window_days: Number of days to look back
    
    Returns:
        DriverStatsWindowResponse if driver exists, None otherwise
    """
    # Verify driver exists
    driver_result = await db.execute(
        select(Driver).where(Driver.id == driver_id)
    )
    driver = driver_result.scalar_one_or_none()
    if not driver:
        return None
    
    # Calculate date range
    end_date = date.today()
    start_date = end_date - timedelta(days=window_days)
    
    # Get assignments in window with optional feedback
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
    
    days: List[DayStats] = []
    total_workload = 0.0
    total_fairness = 0.0
    stress_levels: List[int] = []
    
    for assignment, feedback in assignments_result.all():
        stress_level = feedback.stress_level if feedback else None
        fairness_rating = feedback.fairness_rating if feedback else None
        
        days.append(DayStats(
            stats_date=assignment.date,
            workload_score=assignment.workload_score,
            fairness_score=assignment.fairness_score,
            reported_stress_level=float(stress_level) if stress_level else None,
            reported_fairness_rating=fairness_rating,
        ))
        
        total_workload += assignment.workload_score
        total_fairness += assignment.fairness_score
        if stress_level:
            stress_levels.append(stress_level)
    
    num_days = len(days)
    avg_stress = sum(stress_levels) / len(stress_levels) if stress_levels else None
    
    return DriverStatsWindowResponse(
        driver_id=driver_id,
        window_days=window_days,
        days=days,
        aggregates=StatsAggregates(
            avg_workload=round(total_workload / num_days, 2) if num_days else 0.0,
            avg_fairness_score=round(total_fairness / num_days, 2) if num_days else 0.0,
            avg_stress_level=round(avg_stress, 2) if avg_stress else None,
        ),
    )


async def log_delivery(
    db: AsyncSession,
    assignment_id: UUID,
    route_id: UUID,
    driver_id: UUID,
    stop_order: int,
    status: str,
    issue_type: str = "NONE",
    package_id: Optional[UUID] = None,
    photo_url: Optional[str] = None,
    signature_data: Optional[str] = None,
    notes: Optional[str] = None,
) -> DeliveryLogResponse:
    """
    Create a delivery log entry.
    
    Args:
        db: Database session
        assignment_id: Assignment UUID
        route_id: Route UUID
        driver_id: Driver UUID
        stop_order: Stop order number
        status: Delivery status string
        issue_type: Issue type string
        package_id: Optional package UUID
        photo_url: Optional photo URL
        signature_data: Optional signature data
        notes: Optional notes
    
    Returns:
        DeliveryLogResponse with created log
    
    Raises:
        ValueError: If validation fails
    """
    # Validate assignment belongs to driver and route
    assignment_result = await db.execute(
        select(Assignment).where(Assignment.id == assignment_id)
    )
    assignment = assignment_result.scalar_one_or_none()
    
    if not assignment:
        raise ValueError("Assignment not found")
    if assignment.driver_id != driver_id:
        raise ValueError("Assignment does not belong to given driver")
    if assignment.route_id != route_id:
        raise ValueError("Assignment does not belong to given route")
    
    # Parse enums
    try:
        delivery_status = DeliveryStatus(status)
    except ValueError:
        raise ValueError(f"Invalid status: {status}")
    
    try:
        delivery_issue = DeliveryIssueType(issue_type)
    except ValueError:
        raise ValueError(f"Invalid issue_type: {issue_type}")
    
    # Create log
    log = DeliveryLog(
        assignment_id=assignment_id,
        route_id=route_id,
        driver_id=driver_id,
        stop_order=stop_order,
        package_id=package_id,
        status=delivery_status,
        issue_type=delivery_issue,
        photo_url=photo_url,
        signature_data=signature_data,
        notes=notes,
    )
    
    db.add(log)
    await db.flush()
    await db.refresh(log)
    
    return DeliveryLogResponse(
        id=log.id,
        assignment_id=log.assignment_id,
        route_id=log.route_id,
        driver_id=log.driver_id,
        stop_order=log.stop_order,
        package_id=log.package_id,
        status=log.status.value,
        issue_type=log.issue_type.value,
        photo_url=log.photo_url,
        signature_data=log.signature_data,
        notes=log.notes,
        timestamp=log.timestamp,
    )


async def create_route_swap_request(
    db: AsyncSession,
    from_driver_id: UUID,
    assignment_id: UUID,
    reason: str,
    to_driver_id: Optional[UUID] = None,
    preferred_date: Optional[date] = None,
) -> RouteSwapRequestResponse:
    """
    Create a route swap request.
    
    Args:
        db: Database session
        from_driver_id: Driver requesting swap
        assignment_id: Assignment to swap
        reason: Reason for swap
        to_driver_id: Optional target driver
        preferred_date: Optional preferred date
    
    Returns:
        RouteSwapRequestResponse
    
    Raises:
        ValueError: If validation fails
    """
    # Validate assignment belongs to driver
    assignment_result = await db.execute(
        select(Assignment).where(Assignment.id == assignment_id)
    )
    assignment = assignment_result.scalar_one_or_none()
    
    if not assignment:
        raise ValueError("Assignment not found")
    if assignment.driver_id != from_driver_id:
        raise ValueError("Assignment does not belong to the requesting driver")
    
    # Create request
    swap_request = RouteSwapRequest(
        from_driver_id=from_driver_id,
        to_driver_id=to_driver_id,
        assignment_id=assignment_id,
        reason=reason,
        preferred_date=preferred_date,
        status=SwapRequestStatus.PENDING,
    )
    
    db.add(swap_request)
    await db.flush()
    await db.refresh(swap_request)
    
    return RouteSwapRequestResponse(
        id=swap_request.id,
        from_driver_id=swap_request.from_driver_id,
        to_driver_id=swap_request.to_driver_id,
        assignment_id=swap_request.assignment_id,
        reason=swap_request.reason,
        preferred_date=swap_request.preferred_date,
        status=swap_request.status.value,
        created_at=swap_request.created_at,
        updated_at=swap_request.updated_at,
    )


async def create_stop_issue(
    db: AsyncSession,
    assignment_id: UUID,
    route_id: UUID,
    driver_id: UUID,
    stop_order: int,
    issue_type: str,
    notes: str,
) -> StopIssueResponse:
    """
    Create a stop issue report.
    
    Args:
        db: Database session
        assignment_id: Assignment UUID
        route_id: Route UUID
        driver_id: Driver UUID
        stop_order: Stop order number
        issue_type: Issue type string
        notes: Issue notes
    
    Returns:
        StopIssueResponse
    
    Raises:
        ValueError: If validation fails
    """
    # Validate assignment
    assignment_result = await db.execute(
        select(Assignment).where(Assignment.id == assignment_id)
    )
    assignment = assignment_result.scalar_one_or_none()
    
    if not assignment:
        raise ValueError("Assignment not found")
    if assignment.driver_id != driver_id:
        raise ValueError("Assignment does not belong to given driver")
    if assignment.route_id != route_id:
        raise ValueError("Assignment does not belong to given route")
    
    # Parse issue type
    try:
        issue = StopIssueType(issue_type)
    except ValueError:
        raise ValueError(f"Invalid issue_type: {issue_type}")
    
    # Create issue
    stop_issue = StopIssue(
        assignment_id=assignment_id,
        route_id=route_id,
        driver_id=driver_id,
        stop_order=stop_order,
        issue_type=issue,
        notes=notes,
    )
    
    db.add(stop_issue)
    await db.flush()
    await db.refresh(stop_issue)
    
    return StopIssueResponse(
        id=stop_issue.id,
        assignment_id=stop_issue.assignment_id,
        route_id=stop_issue.route_id,
        driver_id=stop_issue.driver_id,
        stop_order=stop_issue.stop_order,
        issue_type=stop_issue.issue_type.value,
        notes=stop_issue.notes,
        created_at=stop_issue.created_at,
    )
