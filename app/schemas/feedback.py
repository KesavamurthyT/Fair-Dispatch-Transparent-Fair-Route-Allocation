"""
Pydantic schemas for feedback API.
Request and response models for POST /api/v1/feedback endpoint.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class FeedbackRequest(BaseModel):
    """Request schema for POST /api/v1/feedback."""
    driver_id: UUID = Field(..., description="Driver UUID")
    assignment_id: UUID = Field(..., description="Assignment UUID")
    fairness_rating: int = Field(..., ge=1, le=5, description="Fairness rating 1-5")
    stress_level: int = Field(..., ge=1, le=10, description="Stress level 1-10")
    tiredness_level: int = Field(..., ge=1, le=5, description="Tiredness level 1-5")
    hardest_aspect: Optional[str] = Field(
        None,
        description="Hardest aspect: traffic, parking, stairs, weather, heavy_load, customer, navigation, other",
    )
    comments: Optional[str] = Field(None, max_length=1000, description="Additional comments")
    
    @field_validator("hardest_aspect")
    @classmethod
    def validate_hardest_aspect(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {
            "traffic", "parking", "stairs", "weather",
            "heavy_load", "customer", "navigation", "other",
        }
        if v.lower() not in allowed:
            raise ValueError(f"hardest_aspect must be one of: {allowed}")
        return v.lower()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "driver_id": "550e8400-e29b-41d4-a716-446655440001",
                "assignment_id": "550e8400-e29b-41d4-a716-446655440002",
                "fairness_rating": 4,
                "stress_level": 5,
                "tiredness_level": 3,
                "hardest_aspect": "traffic",
                "comments": "Route was good overall",
            }
        }
    }


class FeedbackResponse(BaseModel):
    """Response schema for successfully created feedback."""
    id: UUID
    driver_id: UUID
    assignment_id: UUID
    fairness_rating: int
    stress_level: int
    tiredness_level: int
    hardest_aspect: Optional[str] = None
    comments: Optional[str] = None
    created_at: datetime
    message: str = "Feedback submitted successfully"
    
    model_config = {"from_attributes": True}
