"""
Pydantic schemas for Explainability Agent v2 (Phase 4.3).
Defines input/output types for generating driver and admin explanations.

Phase 8: Added personalized model version for learning-based explanations.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DriverExplanationInput(BaseModel):
    """Input data for generating explanations for a single driver."""
    
    # Driver identification
    driver_id: str
    driver_name: str
    num_drivers: int = Field(..., description="Total number of drivers in this allocation")
    
    # Today's assignment
    today_effort: float = Field(..., description="Effort score for today's route")
    today_rank: int = Field(..., description="Rank among drivers (1=hardest, N=easiest)")
    route_id: str = Field(..., description="ID of assigned route")
    route_summary: Dict = Field(
        ...,
        description="Route details: num_packages, total_weight_kg, num_stops, difficulty_score, estimated_time_minutes"
    )
    
    # Effort breakdown from ML Effort Agent
    effort_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown: physical_effort, route_complexity, time_pressure"
    )
    
    # Global fairness metrics
    global_avg_effort: float
    global_std_effort: float
    global_gini_index: float
    global_max_gap: float
    
    # Recent history (last 7 days)
    history_efforts_last_7_days: List[float] = Field(
        default_factory=list,
        description="Effort scores for the last 7 days"
    )
    history_hard_days_last_7: int = Field(
        default=0,
        description="Count of days with above-threshold effort in last 7 days"
    )
    
    # Recovery / fairness extras
    is_recovery_day: bool = Field(
        default=False,
        description="True if driver is on scheduled recovery"
    )
    had_manual_override: bool = Field(
        default=False,
        description="True if admin manually overrode this assignment"
    )
    complexity_debt: float = Field(
        default=0.0,
        description="Cumulative complexity debt from recent hard days"
    )
    
    # EV context (Phase 7)
    is_ev_driver: bool = Field(
        default=False,
        description="True if driver uses an electric vehicle"
    )
    ev_charging_overhead: float = Field(
        default=0.0,
        description="Effort overhead from EV charging/range constraints"
    )
    
    # Negotiation info from Phase 4.2
    liaison_decision: Optional[str] = Field(
        default=None,
        description="ACCEPT, COUNTER, or FORCE_ACCEPT from Driver Liaison"
    )
    swap_applied: bool = Field(
        default=False,
        description="True if a swap was applied during Final Resolution"
    )
    
    # Learning Agent context (Phase 8)
    personalized_model_version: Optional[int] = Field(
        default=None,
        description="Version of personalized XGBoost model used for this driver"
    )
    personalized_model_mse: Optional[float] = Field(
        default=None,
        description="Current MSE of the personalized model"
    )


class DriverExplanationOutput(BaseModel):
    """Output from ExplainabilityAgent for a single driver."""
    
    driver_explanation: str = Field(
        ...,
        description="Short explanation for driver-facing views (1-3 sentences)"
    )
    admin_explanation: str = Field(
        ...,
        description="Detailed explanation for admin views (multi-sentence with metrics)"
    )
    category: str = Field(
        ...,
        description="Classification category: NEAR_AVG, HEAVY_WITH_SWAP, HEAVY_NO_SWAP, RECOVERY, LIGHT_RECOVERY, LIGHT, HEAVY, LEARNING_OPTIMIZED"
    )

