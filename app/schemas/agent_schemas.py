"""
Pydantic schemas for multi-agent allocation pipeline.
Used by MLEffortAgent, RoutePlannerAgent, and FairnessManagerAgent.
"""

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ==================== ML Effort Agent Schemas ====================

class EffortBreakdown(BaseModel):
    """Breakdown of effort components for a driver-route pair."""
    physical_effort: float = Field(..., description="Effort from packages, weight, stairs")
    route_complexity: float = Field(..., description="Effort from stops, distance, difficulty")
    time_pressure: float = Field(..., description="Effort from time constraints")
    capacity_penalty: float = Field(default=0.0, description="Penalty for capacity mismatch")
    total: float = Field(..., description="Total effort score")


class EffortMatrixResult(BaseModel):
    """Result from ML Effort Agent."""
    matrix: List[List[float]] = Field(..., description="Effort matrix [num_drivers][num_routes]")
    breakdown: Dict[str, EffortBreakdown] = Field(
        ..., 
        description="Breakdown by 'driver_id:route_id' key"
    )
    stats: Dict[str, float] = Field(..., description="Matrix statistics: min, max, avg")
    driver_ids: List[str] = Field(..., description="Driver IDs in matrix row order")
    route_ids: List[str] = Field(..., description="Route IDs in matrix column order")
    # Phase 7: EV infeasibility tracking
    infeasible_pairs: List[str] = Field(
        default_factory=list,
        description="List of 'driver_id:route_id' pairs that are infeasible for EV drivers"
    )


# ==================== Route Planner Agent Schemas ====================

class AllocationItem(BaseModel):
    """Single driver-route assignment."""
    driver_id: UUID
    route_id: UUID
    effort: float


class RoutePlanResult(BaseModel):
    """Result from Route Planner Agent."""
    allocation: List[AllocationItem] = Field(..., description="Driver-route assignments")
    total_effort: float = Field(..., description="Sum of all effort scores")
    avg_effort: float = Field(..., description="Average effort per driver")
    per_driver_effort: Dict[str, float] = Field(
        ..., 
        description="Effort by driver_id"
    )
    proposal_number: int = Field(default=1, description="Proposal 1 or 2")
    solver_status: str = Field(default="OPTIMAL", description="Solver status: OPTIMAL, FEASIBLE, or FALLBACK")


# ==================== Fairness Manager Agent Schemas ====================

class FairnessMetrics(BaseModel):
    """Fairness metrics computed by Fairness Manager."""
    avg_effort: float
    std_dev: float
    max_gap: float
    gini_index: float
    min_effort: float
    max_effort: float
    outlier_count: int = Field(default=0, description="Drivers above avg + 2*std_dev")
    pct_above_avg: float = Field(default=0.0, description="% drivers above average")


class FairnessRecommendations(BaseModel):
    """Recommendations for re-optimization."""
    penalize_high_effort_drivers: bool = False
    high_effort_driver_ids: List[str] = Field(default_factory=list)
    penalty_factor: float = Field(default=1.5, description="Multiplier for high-effort driver costs")
    target_max_gap: Optional[float] = None


class FairnessCheckResult(BaseModel):
    """Result from Fairness Manager Agent."""
    status: Literal["ACCEPT", "REOPTIMIZE"]
    metrics: FairnessMetrics
    recommendations: Optional[FairnessRecommendations] = None
    proposal_number: int = Field(default=1, description="Proposal number checked")
    thresholds_used: Dict[str, float] = Field(
        default_factory=dict,
        description="Threshold values used for decision"
    )


# ==================== Agent Configuration Schemas ====================

class EffortWeights(BaseModel):
    """Configurable weights for effort calculation."""
    alpha_packages: float = Field(default=1.0, ge=0, description="Weight for num_packages")
    beta_weight: float = Field(default=0.5, ge=0, description="Weight for total_weight_kg")
    gamma_difficulty: float = Field(default=10.0, ge=0, description="Weight for difficulty score")
    delta_time: float = Field(default=0.2, ge=0, description="Weight for estimated time")
    epsilon_mismatch: float = Field(default=15.0, ge=0, description="Penalty for capacity mismatch")


class FairnessThresholds(BaseModel):
    """Thresholds for fairness decision."""
    gini_threshold: float = Field(default=0.33, ge=0, le=1)
    stddev_threshold: float = Field(default=25.0, ge=0)
    max_gap_threshold: float = Field(default=25.0, ge=0)


# ==================== Driver Liaison Agent Schemas (Phase 4.2) ====================

DecisionType = Literal["ACCEPT", "COUNTER", "FORCE_ACCEPT"]


class DriverContext(BaseModel):
    """Historical context for driver-specific negotiation decisions."""
    driver_id: str
    recent_avg_effort: float = Field(..., description="Average effort over recent period")
    recent_std_effort: float = Field(..., description="Std dev of recent effort")
    recent_hard_days: int = Field(default=0, description="Days where effort > avg + std")
    fatigue_score: float = Field(default=3.0, ge=1, le=5, description="Fatigue 1-5, 5=exhausted")
    preferences: Dict[str, bool] = Field(default_factory=dict, description="e.g. avoids_stairs")


class DriverAssignmentProposal(BaseModel):
    """Proposal sent to driver's liaison agent for review."""
    driver_id: str
    route_id: str
    effort: float = Field(..., description="Computed effort for this assignment")
    rank_in_team: int = Field(..., ge=1, description="1=hardest, N=easiest for this run")


class DriverLiaisonDecision(BaseModel):
    """Decision from a driver's liaison agent."""
    driver_id: str
    decision: DecisionType
    preferred_route_id: Optional[str] = None
    reason: str = Field(..., description="Explanation for the decision")


class NegotiationResult(BaseModel):
    """Container for all driver liaison decisions."""
    decisions: List[DriverLiaisonDecision] = Field(default_factory=list)
    num_accept: int = Field(default=0)
    num_counter: int = Field(default=0)
    num_force_accept: int = Field(default=0)


# ==================== Final Resolution Schemas (Phase 4.2) ====================

class SwapRecord(BaseModel):
    """Record of a swap applied during final resolution."""
    driver_a: str
    driver_b: str
    route_a: str  # Original route of driver A
    route_b: str  # Original route of driver B (swapped to driver A)
    effort_a_before: float
    effort_a_after: float
    effort_b_before: float
    effort_b_after: float


class FinalResolutionResult(BaseModel):
    """Result from final resolution after negotiation."""
    allocation: List[Dict[str, Any]] = Field(..., description="Final driver->route mapping with effort")
    per_driver_effort: Dict[str, float] = Field(..., description="Final effort per driver")
    metrics: Dict[str, float] = Field(..., description="Final fairness metrics")
    swaps_applied: List[SwapRecord] = Field(default_factory=list)
    unfulfilled_counters: List[str] = Field(default_factory=list, description="Driver IDs")
