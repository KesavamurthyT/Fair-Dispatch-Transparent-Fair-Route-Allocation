"""
Routes API endpoint.
Handles GET /api/v1/routes/{id} for route details.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Route, Assignment, Driver, RoutePackage
from app.schemas.route import RouteResponse, RouteAssignmentInfo, RouteStopInfo

router = APIRouter(prefix="/routes", tags=["Routes"])


@router.get(
    "/{route_id}",
    response_model=RouteResponse,
    summary="Get route details",
    description="Returns route details including assignment information and stops.",
)
async def get_route(
    route_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> RouteResponse:
    """Get route details by ID."""
    
    # Fetch route with assignments and packages
    result = await db.execute(
        select(Route)
        .where(Route.id == route_id)
        .options(
            selectinload(Route.assignments).selectinload(Assignment.driver),
            selectinload(Route.route_packages).selectinload(RoutePackage.package)
        )
    )
    route = result.scalar_one_or_none()
    
    if not route:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Route with ID {route_id} not found",
        )
    
    # Get assignment info if exists
    assignment_info = None
    if route.assignments:
        # Get the most recent assignment
        assignment = max(route.assignments, key=lambda a: a.created_at)
        assignment_info = RouteAssignmentInfo(
            assignment_id=assignment.id,
            driver_id=assignment.driver_id,
            driver_name=assignment.driver.name if assignment.driver else "Unknown",
            workload_score=assignment.workload_score,
            fairness_score=assignment.fairness_score,
            explanation=assignment.explanation,
        )
        
    # Get stops info
    stops = []
    if route.route_packages:
        # Sort by stop order
        sorted_packages = sorted(route.route_packages, key=lambda rp: rp.stop_order)
        for rp in sorted_packages:
            if rp.package:
                stops.append(RouteStopInfo(
                    package_id=rp.package.id,
                    stop_order=rp.stop_order,
                    address=rp.package.address,
                    latitude=rp.package.latitude,
                    longitude=rp.package.longitude,
                    weight_kg=rp.package.weight_kg,
                    priority=rp.package.priority.value if hasattr(rp.package.priority, 'value') else str(rp.package.priority),
                    fragility_level=rp.package.fragility_level
                ))
    
    return RouteResponse(
        id=route.id,
        date=route.date,
        cluster_id=route.cluster_id,
        total_weight_kg=route.total_weight_kg,
        num_packages=route.num_packages,
        num_stops=route.num_stops,
        route_difficulty_score=route.route_difficulty_score,
        estimated_time_minutes=route.estimated_time_minutes,
        created_at=route.created_at,
        assignment=assignment_info,
        stops=stops,
    )
