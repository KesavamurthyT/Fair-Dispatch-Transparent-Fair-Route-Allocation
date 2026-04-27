"""
Pydantic schemas for allocation API.
Request and response models for POST /api/v1/allocate endpoint.
"""

import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class WarehouseInput(BaseModel):
    """Warehouse location for the allocation."""
    lat: float = Field(..., description="Latitude of the warehouse")
    lng: float = Field(..., description="Longitude of the warehouse")


class PackageInput(BaseModel):
    """Input schema for a package in the allocation request."""
    id: str = Field(..., description="External package ID")
    weight_kg: float = Field(..., ge=0.01, description="Package weight in kg")
    fragility_level: int = Field(1, ge=1, le=5, description="Fragility level 1-5")
    address: str = Field(..., min_length=1, description="Delivery address")
    latitude: float = Field(..., description="Delivery latitude")
    longitude: float = Field(..., description="Delivery longitude")
    priority: str = Field("NORMAL", description="Priority: NORMAL, HIGH, EXPRESS")
    
    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        allowed = {"NORMAL", "HIGH", "EXPRESS"}
        if v.upper() not in allowed:
            raise ValueError(f"Priority must be one of: {allowed}")
        return v.upper()


class DriverInput(BaseModel):
    """Input schema for a driver in the allocation request."""
    id: str = Field(..., description="External driver ID")
    name: str = Field(..., min_length=1, description="Driver name")
    vehicle_capacity_kg: float = Field(100.0, ge=0, description="Vehicle capacity in kg")
    preferred_language: str = Field("en", description="Preferred language: en, ta, hi, te, kn")

    # ──────────────────────────────────────────────────────────────────────
    # ⚠ HUMAN-FACTOR EXTENDED FIELDS — NON-ACTIVE
    #
    # These fields are accepted by the API for future research, logging,
    # audit, and explainability enrichment. They are:
    #   • Optional with safe defaults (None or 0)
    #   • Validated with strict Pydantic v2 range constraints
    #   • Backward-compatible (existing JSON without these fields works fine)
    #
    # ⚠ These human-factor fields are currently NON-ACTIVE and do NOT
    #    influence allocation. They are NOT used in:
    #   - MLEffortAgent / effort matrix
    #   - RoutePlannerAgent / solver cost matrix
    #   - FairnessManagerAgent / Gini / std-dev / max-gap
    #   - DriverLiaisonAgent / negotiation comfort band
    #   - LearningAgent / bandit or XGBoost features
    #   - Clustering / workload scoring
    # ──────────────────────────────────────────────────────────────────────
    experience_years: Optional[int] = Field(
        None, ge=0, le=40,
        description="Years of driving experience (NON-ACTIVE, audit only)",
    )
    health_score: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Self-reported health score 0-10 (NON-ACTIVE, audit only)",
    )
    night_vision_impairment: Optional[bool] = Field(
        None,
        description="Whether driver has night vision impairment (NON-ACTIVE, audit only)",
    )
    fatigue_level: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Current fatigue level 0-10 (NON-ACTIVE, audit only)",
    )
    recent_stress_level: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Recent stress level 0-10 (NON-ACTIVE, audit only)",
    )
    age: Optional[int] = Field(
        None, ge=18, le=75,
        description="Driver age in years (NON-ACTIVE, audit only)",
    )
    vehicle_type_extended: Optional[Literal["EV", "diesel", "petrol"]] = Field(
        None,
        description="Extended vehicle fuel type (NON-ACTIVE, audit only)",
    )
    shift_type: Optional[Literal["day", "night"]] = Field(
        None,
        description="Shift type preference (NON-ACTIVE, audit only)",
    )
    preferred_location_zone: Optional[str] = Field(
        None,
        description="Preferred delivery zone name (NON-ACTIVE, audit only)",
    )
    circadian_penalty_index: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Circadian rhythm penalty 0-1 (NON-ACTIVE, audit only)",
    )
    monotony_index: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Route monotony index 0-1 (NON-ACTIVE, audit only)",
    )
    recovery_deficit_score: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Recovery deficit score 0-10 (NON-ACTIVE, audit only)",
    )
    gender: Optional[Literal["male", "female", "other", "prefer_not_to_say"]] = Field(
        None,
        description="Driver gender (NON-ACTIVE, audit only)",
    )

    @field_validator("preferred_language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        allowed = {"en", "ta", "hi", "te", "kn"}
        if v.lower() not in allowed:
            raise ValueError(f"Language must be one of: {allowed}")
        return v.lower()


class AllocationRequest(BaseModel):
    """Request schema for POST /api/v1/allocate."""
    allocation_date: datetime.date = Field(..., description="Date for the allocation")
    warehouse: WarehouseInput
    packages: List[PackageInput] = Field(..., min_length=1)
    drivers: List[DriverInput] = Field(..., min_length=1)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2026-02-10",
                "warehouse": {"lat": 12.9716, "lng": 77.5946},
                "packages": [
                    {
                        "id": "external_pkg_001",
                        "weight_kg": 2.5,
                        "fragility_level": 3,
                        "address": "Some street, Area, City",
                        "latitude": 12.97,
                        "longitude": 77.60,
                        "priority": "NORMAL",
                    }
                ],
                "drivers": [
                    {
                        "id": "external_driver_001",
                        "name": "Raju",
                        "vehicle_capacity_kg": 150,
                        "preferred_language": "en",
                    }
                ],
            }
        }
    }


class RouteSummary(BaseModel):
    """Summary of a route's characteristics."""
    num_packages: int
    total_weight_kg: float
    num_stops: int
    route_difficulty_score: float
    estimated_time_minutes: int


class AssignmentResponse(BaseModel):
    """Response schema for a single driver assignment."""
    driver_id: UUID
    driver_external_id: str
    driver_name: str
    route_id: UUID
    workload_score: float
    fairness_score: float
    route_summary: RouteSummary
    explanation: str


class GlobalFairness(BaseModel):
    """Global fairness metrics for the allocation."""
    avg_workload: float
    std_dev: float
    gini_index: float


class AllocationResponse(BaseModel):
    """Response schema for POST /api/v1/allocate."""
    allocation_run_id: UUID
    allocation_date: datetime.date
    global_fairness: GlobalFairness
    assignments: List[AssignmentResponse]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "allocation_run_id": "550e8400-e29b-41d4-a716-446655440000",
                "date": "2026-02-10",
                "global_fairness": {
                    "avg_workload": 63.2,
                    "std_dev": 18.4,
                    "gini_index": 0.29,
                },
                "assignments": [
                    {
                        "driver_id": "550e8400-e29b-41d4-a716-446655440001",
                        "driver_external_id": "external_driver_001",
                        "driver_name": "Raju",
                        "route_id": "550e8400-e29b-41d4-a716-446655440002",
                        "workload_score": 65.3,
                        "fairness_score": 0.82,
                        "route_summary": {
                            "num_packages": 22,
                            "total_weight_kg": 48.5,
                            "num_stops": 14,
                            "route_difficulty_score": 2.1,
                            "estimated_time_minutes": 145,
                        },
                        "explanation": "You received a moderate route...",
                    }
                ],
            }
        }
    }
