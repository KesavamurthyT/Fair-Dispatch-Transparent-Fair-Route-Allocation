"""
Pydantic schemas for route API.
Response models for GET /api/v1/routes/{id} endpoint.
"""

from datetime import date, datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel


class RouteAssignmentInfo(BaseModel):
    """Assignment information for a route."""
    assignment_id: UUID
    driver_id: UUID
    driver_name: str
    workload_score: float
    fairness_score: float
    explanation: Optional[str] = None


class RouteStopInfo(BaseModel):
    """Information about a stop on the route."""
    package_id: UUID
    stop_order: int
    address: str
    latitude: float
    longitude: float
    weight_kg: float
    priority: str
    fragility_level: int


class RouteResponse(BaseModel):
    """Response schema for route details."""
    id: UUID
    date: date
    cluster_id: int
    total_weight_kg: float
    num_packages: int
    num_stops: int
    route_difficulty_score: float
    estimated_time_minutes: int
    created_at: datetime
    assignment: Optional[RouteAssignmentInfo] = None
    stops: List[RouteStopInfo] = []
    
    model_config = {"from_attributes": True}
