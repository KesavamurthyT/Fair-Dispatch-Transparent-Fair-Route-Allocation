"""
Pydantic schemas for driver API.
Response models for GET /api/v1/drivers/{id} endpoint.
"""

from datetime import date, datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DriverStatsResponse(BaseModel):
    """Daily statistics for a driver."""
    date: date
    avg_workload_score: float
    total_routes: int
    gini_contribution: Optional[float] = None
    reported_stress_level: Optional[float] = None
    reported_fairness_score: Optional[float] = None


class DriverResponse(BaseModel):
    """Response schema for driver details with recent stats."""
    id: UUID
    external_id: Optional[str] = None
    name: str
    phone: Optional[str] = None
    whatsapp_number: Optional[str] = None
    preferred_language: str
    vehicle_type: str
    vehicle_capacity_kg: float
    license_number: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    recent_stats: List[DriverStatsResponse] = Field(
        default_factory=list,
        description="Recent daily stats (last 7 days)",
    )
    
    model_config = {"from_attributes": True}
