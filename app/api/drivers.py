"""
Drivers API endpoint.
Handles GET /api/v1/drivers/{id} for driver details.
"""

from datetime import date, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Driver, DriverStatsDaily
from app.schemas.driver import DriverResponse, DriverStatsResponse

router = APIRouter(prefix="/drivers", tags=["Drivers"])


@router.get(
    "/{driver_id}",
    response_model=DriverResponse,
    summary="Get driver details",
    description="Returns driver details including recent fairness statistics (last 7 days).",
)
async def get_driver(
    driver_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DriverResponse:
    """Get driver details by ID."""
    
    # Fetch driver with recent stats
    result = await db.execute(
        select(Driver)
        .where(Driver.id == driver_id)
        .options(selectinload(Driver.daily_stats))
    )
    driver = result.scalar_one_or_none()
    
    if not driver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Driver with ID {driver_id} not found",
        )
    
    # Get stats for last 7 days
    seven_days_ago = date.today() - timedelta(days=7)
    recent_stats = [
        DriverStatsResponse(
            date=stat.date,
            avg_workload_score=stat.avg_workload_score,
            total_routes=stat.total_routes,
            gini_contribution=stat.gini_contribution,
            reported_stress_level=stat.reported_stress_level,
            reported_fairness_score=stat.reported_fairness_score,
        )
        for stat in driver.daily_stats
        if stat.date >= seven_days_ago
    ]
    
    # Sort by date descending
    recent_stats.sort(key=lambda x: x.date, reverse=True)
    
    return DriverResponse(
        id=driver.id,
        external_id=driver.external_id,
        name=driver.name,
        phone=driver.phone,
        whatsapp_number=driver.whatsapp_number,
        preferred_language=driver.preferred_language.value,
        vehicle_type=driver.vehicle_type.value,
        vehicle_capacity_kg=driver.vehicle_capacity_kg,
        license_number=driver.license_number,
        created_at=driver.created_at,
        updated_at=driver.updated_at,
        recent_stats=recent_stats,
    )


@router.get(
    "/external/{external_id}",
    response_model=DriverResponse,
    summary="Get driver by external ID",
    description="Returns driver details by external ID (from integration system).",
)
async def get_driver_by_external_id(
    external_id: str,
    db: AsyncSession = Depends(get_db),
) -> DriverResponse:
    """Get driver details by external ID."""
    
    result = await db.execute(
        select(Driver)
        .where(Driver.external_id == external_id)
        .options(selectinload(Driver.daily_stats))
    )
    driver = result.scalar_one_or_none()
    
    if not driver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Driver with external ID {external_id} not found",
        )
    
    # Get stats for last 7 days
    seven_days_ago = date.today() - timedelta(days=7)
    recent_stats = [
        DriverStatsResponse(
            date=stat.date,
            avg_workload_score=stat.avg_workload_score,
            total_routes=stat.total_routes,
            gini_contribution=stat.gini_contribution,
            reported_stress_level=stat.reported_stress_level,
            reported_fairness_score=stat.reported_fairness_score,
        )
        for stat in driver.daily_stats
        if stat.date >= seven_days_ago
    ]
    
    recent_stats.sort(key=lambda x: x.date, reverse=True)
    
    return DriverResponse(
        id=driver.id,
        external_id=driver.external_id,
        name=driver.name,
        phone=driver.phone,
        whatsapp_number=driver.whatsapp_number,
        preferred_language=driver.preferred_language.value,
        vehicle_type=driver.vehicle_type.value,
        vehicle_capacity_kg=driver.vehicle_capacity_kg,
        license_number=driver.license_number,
        created_at=driver.created_at,
        updated_at=driver.updated_at,
        recent_stats=recent_stats,
    )
