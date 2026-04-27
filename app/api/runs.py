"""
Run-scoped API endpoints.
Provides endpoints for fetching data specific to an allocation run.
"""

import json
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Route, RoutePackage, Package, Assignment, Driver
from app.models.allocation_run import AllocationRun, AllocationRunStatus
from app.core.events import agent_event_bus


router = APIRouter(prefix="/runs", tags=["Runs"])


# =============================================================================
# Pydantic Schemas
# =============================================================================

class StopInfo(BaseModel):
    """Stop information for map display."""
    lat: float
    lng: float
    package_id: str
    stop_order: int
    address: Optional[str] = None


class RouteOnMap(BaseModel):
    """Route information for map display."""
    route_id: str
    driver_id: Optional[str] = None
    driver_name: Optional[str] = None
    stops: List[StopInfo]
    total_weight_kg: float
    num_packages: int
    estimated_time_minutes: int


class RoutesOnMapResponse(BaseModel):
    """Response for routes-on-map endpoint."""
    allocation_run_id: str
    routes: List[RouteOnMap]


class RunSummaryResponse(BaseModel):
    """Summary information for an allocation run."""
    allocation_run_id: str
    date: str
    status: str
    num_drivers: int
    num_routes: int
    num_packages: int
    global_gini_index: float
    global_std_dev: float
    global_max_gap: float
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/{run_id}/routes-on-map",
    response_model=RoutesOnMapResponse,
    summary="Get routes for map visualization",
    description="""
    Returns all routes and their stop coordinates for a given allocation run.
    Used exclusively by the 8090 visualization map.
    Each route includes driver info and ordered stops with coordinates.
    """,
)
async def get_routes_for_map(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> RoutesOnMapResponse:
    """Return routes and stops for map visualization, scoped to run_id."""
    
    # Verify run exists
    run_result = await db.execute(
        select(AllocationRun).where(AllocationRun.id == run_id)
    )
    allocation_run = run_result.scalar_one_or_none()
    
    if not allocation_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Allocation run {run_id} not found",
        )
    
    # Get all routes for this run
    routes_result = await db.execute(
        select(Route).where(Route.allocation_run_id == run_id)
    )
    routes = routes_result.scalars().all()
    
    # Get assignments to map routes to drivers
    assignments_result = await db.execute(
        select(Assignment, Driver)
        .join(Driver, Assignment.driver_id == Driver.id)
        .where(Assignment.allocation_run_id == run_id)
    )
    assignments = assignments_result.all()
    
    # Build driver lookup by route_id
    driver_by_route: dict[uuid.UUID, tuple] = {}
    for assignment, driver in assignments:
        driver_by_route[assignment.route_id] = (str(driver.id), driver.name)
    
    route_objs: List[RouteOnMap] = []
    
    for route in routes:
        # Get stops for this route ordered by stop_order
        stops_result = await db.execute(
            select(RoutePackage, Package)
            .join(Package, RoutePackage.package_id == Package.id)
            .where(RoutePackage.route_id == route.id)
            .order_by(RoutePackage.stop_order)
        )
        stops_data = stops_result.all()
        
        stops = [
            StopInfo(
                lat=pkg.latitude,
                lng=pkg.longitude,
                package_id=str(pkg.id),
                stop_order=rp.stop_order,
                address=pkg.address,
            )
            for rp, pkg in stops_data
        ]
        
        driver_info = driver_by_route.get(route.id)
        
        route_objs.append(RouteOnMap(
            route_id=str(route.id),
            driver_id=driver_info[0] if driver_info else None,
            driver_name=driver_info[1] if driver_info else None,
            stops=stops,
            total_weight_kg=route.total_weight_kg,
            num_packages=route.num_packages,
            estimated_time_minutes=route.estimated_time_minutes,
        ))
    
    return RoutesOnMapResponse(
        allocation_run_id=str(run_id),
        routes=route_objs,
    )


@router.get(
    "/{run_id}/summary",
    response_model=RunSummaryResponse,
    summary="Get allocation run summary",
    description="Returns metadata and metrics for a specific allocation run.",
)
async def get_run_summary(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> RunSummaryResponse:
    """Return summary information for an allocation run."""
    
    result = await db.execute(
        select(AllocationRun).where(AllocationRun.id == run_id)
    )
    allocation_run = result.scalar_one_or_none()
    
    if not allocation_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Allocation run {run_id} not found",
        )
    
    return RunSummaryResponse(
        allocation_run_id=str(allocation_run.id),
        date=str(allocation_run.date),
        status=allocation_run.status.value,
        num_drivers=allocation_run.num_drivers,
        num_routes=allocation_run.num_routes,
        num_packages=allocation_run.num_packages,
        global_gini_index=allocation_run.global_gini_index,
        global_std_dev=allocation_run.global_std_dev,
        global_max_gap=allocation_run.global_max_gap,
        started_at=allocation_run.started_at.isoformat() if allocation_run.started_at else None,
        finished_at=allocation_run.finished_at.isoformat() if allocation_run.finished_at else None,
    )


@router.get(
    "/{run_id}/recent-events",
    summary="Get recent agent events for a run",
    description="Returns recent agent events from the in-memory event bus for a specific run.",
)
async def get_recent_events_for_run(run_id: uuid.UUID):
    """Return cached recent events for a run (for late joiners)."""
    run_id_str = str(run_id)
    events = agent_event_bus.get_recent_events(allocation_run_id=run_id_str, limit=50)
    return {"allocation_run_id": run_id_str, "events": events}


@router.get(
    "/{run_id}/agent-events",
    summary="SSE stream for agent events (run-scoped)",
    description="""
    Server-Sent Events stream of agent events filtered by allocation_run_id.
    Only events matching the specified run_id will be streamed.
    Connect using EventSource in browser.
    """,
)
async def agent_events_for_run(run_id: uuid.UUID):
    """SSE endpoint filtered by allocation_run_id."""
    
    run_id_str = str(run_id)
    
    async def event_generator():
        # Send initial connection event
        connected_event = {
            "type": "connected",
            "allocation_run_id": run_id_str,
            "message": f"Subscribed to events for run {run_id_str[:8]}...",
            "timestamp": datetime.utcnow().isoformat(),
        }
        yield f"data: {json.dumps(connected_event)}\n\n"
        
        # Subscribe to event bus and filter by run_id
        async for event in agent_event_bus.subscribe():
            # Only emit events for this specific run
            if event.get("allocation_run_id") == run_id_str:
                yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
