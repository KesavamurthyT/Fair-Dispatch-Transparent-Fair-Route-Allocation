"""
AllocationState schema for LangGraph workflow.
Preserves all existing data flow while enabling LangGraph orchestration.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field


class AllocationState(BaseModel):
    """
    LangGraph state for the allocation workflow.
    
    This schema preserves all existing data flow patterns from the
    original multi-agent pipeline while enabling LangGraph orchestration,
    checkpointing, and LangSmith tracing.
    """
    
    # === Input (from /allocate POST) ===
    request: Dict[str, Any] = Field(
        default_factory=dict,
        description="AllocationRequest.dict() - original request payload"
    )
    config_used: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Active FairnessConfig snapshot"
    )
    
    # === Database Context (populated during workflow) ===
    allocation_run_id: Optional[str] = Field(
        default=None,
        description="UUID string of AllocationRun for persistence"
    )
    driver_models: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Driver model data for agents"
    )
    route_models: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Route model data for agents"
    )
    route_dicts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Route dictionaries with packages"
    )
    driver_contexts: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="DriverContext per driver for liaison agent"
    )
    
    # === Phase 1: ML Effort Agent Output ===
    effort_matrix: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Inline effort matrix dict: matrix, driver_ids, route_ids, stats, breakdown, infeasible_pairs"
    )
    effort_matrix_ref: Optional[str] = Field(
        default=None,
        description="Redis artifact key for the effort matrix (stored via artifact_store)"
    )
    effort_matrix_meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Lightweight metadata: shape, driver_ids, route_ids, avg_effort"
    )

    
    # === Phase 2-3: Route Planner + Fairness Agent Outputs ===
    route_proposal_1: Optional[Dict[str, Any]] = Field(
        default=None,
        description="RoutePlanResult proposal 1 (pure effort-based)"
    )
    fairness_check_1: Optional[Dict[str, Any]] = Field(
        default=None,
        description="FairnessCheckResult for proposal 1"
    )
    route_proposal_2: Optional[Dict[str, Any]] = Field(
        default=None,
        description="RoutePlanResult proposal 2 (with fairness penalties)"
    )
    fairness_check_2: Optional[Dict[str, Any]] = Field(
        default=None,
        description="FairnessCheckResult for proposal 2"
    )
    final_proposal: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Selected proposal after fairness comparison"
    )
    final_fairness: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Final fairness metrics"
    )
    
    # === Phase 4-5: Driver Liaison + Final Resolution Outputs ===
    liaison_feedback: Optional[Dict[str, Any]] = Field(
        default=None,
        description="NegotiationResult from DriverLiaisonAgent"
    )
    resolution_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="FinalResolutionAgent swap results"
    )
    final_per_driver_effort: Dict[str, float] = Field(
        default_factory=dict,
        description="Final effort per driver after resolution"
    )
    
    # === Phase 6: Explainability Agent Output ===
    explanations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="driver_id -> {driver_explanation, admin_explanation, category}"
    )
    
    # === Phase 7: Recovery (from RecoveryService) ===
    recovery_targets: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="driver_id -> max effort target (None if no recovery needed)"
    )
    
    # === Phase 8: Learning ===
    learning_episode_created: bool = Field(
        default=False,
        description="Whether learning episode was created"
    )
    learning_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Learning agent summary: history stats, fatigue, recommendations"
    )
    
    # === Observability (Phase 5 compatible) ===
    decision_logs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Agent step logs for DecisionLog persistence"
    )
    workflow_start: datetime = Field(
        default_factory=datetime.utcnow,
        description="Workflow start timestamp"
    )
    
    # === Final Response Data ===
    assignments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="AssignmentResponse data for API response"
    )
    global_fairness: Optional[Dict[str, Any]] = Field(
        default=None,
        description="GlobalFairness metrics for API response"
    )
    
    class Config:
        arbitrary_types_allowed = True


# Helper function to serialize state for checkpointing
def serialize_state(state: AllocationState) -> Dict[str, Any]:
    """Serialize AllocationState to dict for checkpointing."""
    return state.model_dump(mode="json")


# Helper function to deserialize state from checkpoint
def deserialize_state(data: Dict[str, Any]) -> AllocationState:
    """Deserialize dict to AllocationState from checkpoint."""
    return AllocationState.model_validate(data)
