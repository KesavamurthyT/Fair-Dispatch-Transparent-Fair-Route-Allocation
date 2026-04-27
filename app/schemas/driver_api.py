"""
Pydantic schemas for Phase 2 Driver-facing API endpoints.
"""

import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ==================== Today's Assignment Schemas ====================

class PackageDetail(BaseModel):
    """Package details for a stop."""
    id: UUID
    external_id: str
    weight_kg: float
    fragility_level: int
    priority: str


class StopDetail(BaseModel):
    """Details of a delivery stop."""
    stop_order: int
    address: str
    latitude: float
    longitude: float
    packages: List[PackageDetail]
    building_type: Optional[str] = None
    floor_number: Optional[int] = None
    stairs_likelihood: Optional[float] = None


class RouteSummaryDetail(BaseModel):
    """Summary of route characteristics."""
    num_packages: int
    total_weight_kg: float
    num_stops: int
    route_difficulty_score: float
    estimated_time_minutes: int


class AssignmentDetail(BaseModel):
    """Full assignment details with stops."""
    assignment_id: UUID
    route_id: UUID
    workload_score: float
    fairness_score: float
    explanation: str
    route_summary: RouteSummaryDetail
    stops: List[StopDetail]


class DriverDetail(BaseModel):
    """Driver info in assignment response."""
    id: UUID
    external_id: Optional[str]
    name: str
    preferred_language: str


class TodayAssignmentResponse(BaseModel):
    """Response for GET /api/v1/assignments/today."""
    assignment_date: datetime.date
    driver: DriverDetail
    assignment: AssignmentDetail


# ==================== Driver Stats Schemas ====================

class DayStats(BaseModel):
    """Stats for a single day."""
    stats_date: datetime.date
    workload_score: float
    fairness_score: float
    reported_stress_level: Optional[float] = None
    reported_fairness_rating: Optional[int] = None


class StatsAggregates(BaseModel):
    """Aggregated stats over window."""
    avg_workload: float
    avg_fairness_score: float
    avg_stress_level: Optional[float] = None


class DriverStatsWindowResponse(BaseModel):
    """Response for GET /api/v1/drivers/{id}/stats."""
    driver_id: UUID
    window_days: int
    days: List[DayStats]
    aggregates: StatsAggregates


# ==================== Delivery Log Schemas ====================

class DeliveryLogRequest(BaseModel):
    """Request for POST /api/v1/deliveries/log."""
    assignment_id: UUID
    route_id: UUID
    driver_id: UUID
    stop_order: int = Field(..., ge=1)
    package_id: Optional[UUID] = None
    status: str = Field(..., description="DELIVERED, FAILED, or PARTIAL")
    issue_type: str = Field(default="NONE", description="Issue type if any")
    photo_url: Optional[str] = None
    signature_data: Optional[str] = None
    notes: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "assignment_id": "550e8400-e29b-41d4-a716-446655440001",
                "route_id": "550e8400-e29b-41d4-a716-446655440002",
                "driver_id": "550e8400-e29b-41d4-a716-446655440003",
                "stop_order": 1,
                "package_id": "550e8400-e29b-41d4-a716-446655440004",
                "status": "DELIVERED",
                "issue_type": "NONE",
                "notes": "Left with security"
            }
        }
    }


class DeliveryLogResponse(BaseModel):
    """Response for POST /api/v1/deliveries/log."""
    id: UUID
    assignment_id: UUID
    route_id: UUID
    driver_id: UUID
    stop_order: int
    package_id: Optional[UUID]
    status: str
    issue_type: str
    photo_url: Optional[str]
    signature_data: Optional[str]
    notes: Optional[str]
    timestamp: datetime.datetime


# ==================== Extended Feedback Schemas ====================

class ExtendedFeedbackRequest(BaseModel):
    """Extended feedback request for Phase 2."""
    driver_id: UUID
    assignment_id: UUID
    fairness_rating: int = Field(..., ge=1, le=5)
    stress_level: int = Field(..., ge=1, le=10)
    tiredness_level: int = Field(..., ge=1, le=5)
    hardest_aspect: Optional[str] = None
    route_difficulty_self_report: Optional[int] = Field(None, ge=1, le=5)
    would_take_similar_route_again: Optional[bool] = None
    most_unfair_aspect: Optional[str] = None
    comments: Optional[str] = Field(None, max_length=1000)

    model_config = {
        "json_schema_extra": {
            "example": {
                "driver_id": "550e8400-e29b-41d4-a716-446655440001",
                "assignment_id": "550e8400-e29b-41d4-a716-446655440002",
                "fairness_rating": 4,
                "stress_level": 5,
                "tiredness_level": 3,
                "hardest_aspect": "stairs",
                "route_difficulty_self_report": 4,
                "would_take_similar_route_again": True,
                "most_unfair_aspect": "parking",
                "comments": "Too many apartments with no lift"
            }
        }
    }


class ExtendedFeedbackResponse(BaseModel):
    """Response for extended feedback submission."""
    id: UUID
    driver_id: UUID
    assignment_id: UUID
    fairness_rating: int
    stress_level: int
    tiredness_level: int
    hardest_aspect: Optional[str]
    route_difficulty_self_report: Optional[int]
    would_take_similar_route_again: Optional[bool]
    most_unfair_aspect: Optional[str]
    comments: Optional[str]
    created_at: datetime.datetime


# ==================== Route Swap Request Schemas ====================

class RouteSwapRequestCreate(BaseModel):
    """Request for POST /api/v1/route_swap_requests."""
    from_driver_id: UUID
    to_driver_id: Optional[UUID] = None
    assignment_id: UUID
    reason: str = Field(..., min_length=1)
    preferred_date: Optional[datetime.date] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "from_driver_id": "550e8400-e29b-41d4-a716-446655440001",
                "to_driver_id": None,
                "assignment_id": "550e8400-e29b-41d4-a716-446655440002",
                "reason": "Had 3 hard days, want a lighter one tomorrow",
                "preferred_date": "2026-02-11"
            }
        }
    }


class RouteSwapRequestResponse(BaseModel):
    """Response for route swap request."""
    id: UUID
    from_driver_id: UUID
    to_driver_id: Optional[UUID]
    assignment_id: UUID
    reason: str
    preferred_date: Optional[datetime.date]
    status: str
    created_at: datetime.datetime
    updated_at: datetime.datetime


# ==================== Stop Issue Schemas ====================

class StopIssueRequest(BaseModel):
    """Request for POST /api/v1/stop_issues."""
    assignment_id: UUID
    route_id: UUID
    driver_id: UUID
    stop_order: int = Field(..., ge=1)
    issue_type: str = Field(..., description="NAVIGATION, SAFETY, TIME_WINDOW, CUSTOMER_UNAVAILABLE, OTHER")
    notes: str = Field(..., min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "assignment_id": "550e8400-e29b-41d4-a716-446655440001",
                "route_id": "550e8400-e29b-41d4-a716-446655440002",
                "driver_id": "550e8400-e29b-41d4-a716-446655440003",
                "stop_order": 5,
                "issue_type": "SAFETY",
                "notes": "Street too dark, dogs on road"
            }
        }
    }


class StopIssueResponse(BaseModel):
    """Response for stop issue creation."""
    id: UUID
    assignment_id: UUID
    route_id: UUID
    driver_id: UUID
    stop_order: int
    issue_type: str
    notes: str
    created_at: datetime.datetime
