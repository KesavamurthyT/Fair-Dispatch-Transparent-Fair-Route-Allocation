"""
Pydantic schemas for Phase 3 Admin-facing API endpoints.
"""

import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ==================== Health Schemas ====================

class LatestAllocationRunInfo(BaseModel):
    """Basic info about the latest allocation run."""
    id: UUID
    run_date: datetime.date
    status: str
    started_at: datetime.datetime
    finished_at: Optional[datetime.datetime]


class HealthResponse(BaseModel):
    """Response for GET /api/v1/admin/health."""
    status: str
    database: str
    latest_allocation_run: Optional[LatestAllocationRunInfo] = None


# ==================== Allocation Run Schemas ====================

class AllocationRunResponse(BaseModel):
    """Single allocation run response."""
    id: UUID
    run_date: datetime.date
    num_drivers: int
    num_routes: int
    num_packages: int
    global_gini_index: float
    global_std_dev: float
    global_max_gap: float
    status: str
    error_message: Optional[str]
    started_at: datetime.datetime
    finished_at: Optional[datetime.datetime]


class AllocationRunsListResponse(BaseModel):
    """Response for GET /api/v1/admin/allocation_runs."""
    runs: List[AllocationRunResponse]


# ==================== Admin Assignment Schemas ====================

class AdminDriverInfo(BaseModel):
    """Driver info for admin assignment view."""
    id: UUID
    name: str
    vehicle_type: Optional[str] = None


class AdminRouteInfo(BaseModel):
    """Route info for admin assignment view."""
    id: UUID
    num_packages: int
    total_weight_kg: float
    num_stops: int
    route_difficulty_score: float
    estimated_time_minutes: int


class AdminFeedbackInfo(BaseModel):
    """Feedback info for admin assignment view."""
    fairness_rating: Optional[int]
    stress_level: Optional[int]


class AdminAssignmentResponse(BaseModel):
    """Single assignment for admin view."""
    assignment_id: UUID
    allocation_run_id: UUID
    driver: AdminDriverInfo
    route: AdminRouteInfo
    workload_score: float
    fairness_score: float
    explanation: Optional[str]
    driver_explanation: Optional[str] = None
    admin_explanation: Optional[str] = None
    feedback: Optional[AdminFeedbackInfo] = None


class AdminAssignmentsListResponse(BaseModel):
    """Response for GET /api/v1/admin/assignments."""
    items: List[AdminAssignmentResponse]
    page: int
    page_size: int
    total_items: int


# ==================== Fairness Metrics Schemas ====================

class FairnessMetricsPoint(BaseModel):
    """A single point in fairness metrics time series."""
    point_date: datetime.date
    gini_index: float
    std_dev: float
    max_gap: float
    outlier_count: int
    appeals_count: int


class FairnessMetricsResponse(BaseModel):
    """Response for GET /api/v1/admin/metrics/fairness."""
    points: List[FairnessMetricsPoint]


# ==================== Workload Heatmap Schemas ====================

class HeatmapDriver(BaseModel):
    """Driver in heatmap response."""
    id: UUID
    name: str


class HeatmapCell(BaseModel):
    """Single cell in workload heatmap."""
    driver_id: UUID
    cell_date: datetime.date
    workload_score: float
    fairness_score: float


class WorkloadHeatmapResponse(BaseModel):
    """Response for GET /api/v1/admin/workload_heatmap."""
    drivers: List[HeatmapDriver]
    dates: List[datetime.date]
    cells: List[HeatmapCell]


# ==================== Driver History Schemas ====================

class DriverHistoryDay(BaseModel):
    """Single day in driver history."""
    day_date: datetime.date
    workload_score: float
    fairness_score: float
    reported_stress_level: Optional[float]
    reported_fairness_rating: Optional[int]
    appeals_count: int = 0
    manual_overrides_affecting_driver: int = 0


class DriverHistoryResponse(BaseModel):
    """Response for GET /api/v1/admin/driver/{id}/history."""
    driver_id: UUID
    window_days: int
    days: List[DriverHistoryDay]


# ==================== Appeals Schemas ====================

class AppealContext(BaseModel):
    """Context info for an appeal."""
    workload_score: float
    fairness_score: float
    recent_streak_hard_days: Optional[int] = None


class AppealDriverInfo(BaseModel):
    """Driver info in appeal response."""
    id: UUID
    name: str


class AppealResponse(BaseModel):
    """Single appeal response."""
    id: UUID
    driver: AppealDriverInfo
    assignment_id: UUID
    appeal_date: Optional[datetime.date]
    reason: str
    status: str
    admin_note: Optional[str]
    created_at: datetime.datetime
    updated_at: datetime.datetime
    context: Optional[AppealContext] = None


class AppealsListResponse(BaseModel):
    """Response for GET /api/v1/admin/appeals."""
    items: List[AppealResponse]


class AppealDecisionRequest(BaseModel):
    """Request for POST /api/v1/admin/appeals/{id}/decision."""
    status: str = Field(..., description="APPROVED, REJECTED, or RESOLVED")
    admin_note: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "APPROVED",
                "admin_note": "Agreed – workload above streak threshold, will assign lighter route tomorrow."
            }
        }
    }


class AppealDecisionResponse(BaseModel):
    """Response for appeal decision."""
    id: UUID
    status: str
    admin_note: Optional[str]
    updated_at: datetime.datetime


# ==================== Manual Override Schemas ====================

class ManualOverrideRequest(BaseModel):
    """Request for POST /api/v1/admin/manual_override."""
    allocation_run_id: UUID
    old_driver_id: UUID
    new_driver_id: UUID
    route_id: UUID
    reason: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "allocation_run_id": "550e8400-e29b-41d4-a716-446655440001",
                "old_driver_id": "550e8400-e29b-41d4-a716-446655440002",
                "new_driver_id": "550e8400-e29b-41d4-a716-446655440003",
                "route_id": "550e8400-e29b-41d4-a716-446655440004",
                "reason": "Driver A reported injury"
            }
        }
    }


class MetricsSnapshot(BaseModel):
    """Before/after metrics snapshot."""
    gini_index: float
    std_dev: float
    max_gap: float


class ManualOverrideInfo(BaseModel):
    """Manual override details in response."""
    id: UUID
    allocation_run_id: UUID
    old_driver_id: Optional[UUID]
    new_driver_id: Optional[UUID]
    route_id: Optional[UUID]
    reason: Optional[str]
    before_metrics: MetricsSnapshot
    after_metrics: MetricsSnapshot
    created_at: datetime.datetime


class UpdatedAssignment(BaseModel):
    """Assignment affected by manual override."""
    assignment_id: UUID
    driver_id: UUID
    route_id: Optional[UUID]


class ManualOverrideResponse(BaseModel):
    """Response for POST /api/v1/admin/manual_override."""
    manual_override: ManualOverrideInfo
    updated_assignments: List[UpdatedAssignment]


# ==================== Fairness Config Schemas ====================

class FairnessConfigRequest(BaseModel):
    """Request for POST /api/v1/admin/fairness_config."""
    workload_weight_packages: float = Field(default=1.0, ge=0)
    workload_weight_weight_kg: float = Field(default=0.5, ge=0)
    workload_weight_difficulty: float = Field(default=10.0, ge=0)
    workload_weight_time: float = Field(default=0.2, ge=0)
    gini_threshold: float = Field(default=0.33, ge=0, le=1)
    stddev_threshold: float = Field(default=25.0, ge=0)
    max_gap_threshold: float = Field(default=25.0, ge=0)
    recovery_mode_enabled: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "workload_weight_packages": 1.0,
                "workload_weight_weight_kg": 0.5,
                "workload_weight_difficulty": 10.0,
                "workload_weight_time": 0.2,
                "gini_threshold": 0.33,
                "stddev_threshold": 25.0,
                "max_gap_threshold": 25.0,
                "recovery_mode_enabled": True
            }
        }
    }


class FairnessConfigResponse(BaseModel):
    """Response for fairness config endpoints."""
    id: UUID
    is_active: bool
    workload_weight_packages: float
    workload_weight_weight_kg: float
    workload_weight_difficulty: float
    workload_weight_time: float
    gini_threshold: float
    stddev_threshold: float
    max_gap_threshold: float
    recovery_mode_enabled: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime


# ==================== Agent Timeline Schemas ====================

class DecisionLogStep(BaseModel):
    """Single step in agent timeline (legacy)."""
    id: UUID
    agent_name: str
    step_type: str
    input_snapshot: Optional[dict] = None
    output_snapshot: Optional[dict] = None
    created_at: datetime.datetime


class AllocationRunInfo(BaseModel):
    """Allocation run info for timeline."""
    id: UUID
    date: datetime.date
    num_drivers: int
    num_routes: int
    num_packages: int
    global_metrics: dict  # gini_index, std_dev, max_gap, avg_effort
    status: str
    started_at: datetime.datetime
    finished_at: Optional[datetime.datetime] = None
    duration_seconds: Optional[float] = None


class AgentTimelineEvent(BaseModel):
    """Enhanced timeline event with short_message and details."""
    id: UUID
    timestamp: datetime.datetime
    agent_name: str
    step_type: str
    short_message: str
    details: dict = Field(default_factory=dict)


class AgentTimelineResponse(BaseModel):
    """Response for GET /api/v1/admin/agent_timeline."""
    allocation_run: AllocationRunInfo
    timeline: List[AgentTimelineEvent]
    # Legacy field for backward compatibility
    allocation_run_id: Optional[UUID] = None
    steps: Optional[List[DecisionLogStep]] = None


# ==================== Driver Allocation Story Schemas ====================

class StoryDriverInfo(BaseModel):
    """Driver info in story response."""
    id: UUID
    name: str


class StoryRouteSummary(BaseModel):
    """Route summary in story response."""
    num_packages: int
    total_weight_kg: float
    num_stops: int
    route_difficulty_score: float
    estimated_time_minutes: int


class StoryRouteInfo(BaseModel):
    """Route info in story response."""
    id: UUID
    summary: StoryRouteSummary


class StoryEffortInfo(BaseModel):
    """Effort metrics for today."""
    value: float
    rank: int
    num_drivers: int
    percent_vs_avg: float


class StoryTodayInfo(BaseModel):
    """Today's assignment info."""
    assignment_id: UUID
    route: StoryRouteInfo
    effort: StoryEffortInfo
    fairness_score: float
    driver_explanation: Optional[str] = None
    admin_explanation: Optional[str] = None
    explainability_category: Optional[str] = None


class StoryHistoryDay(BaseModel):
    """Single day in history."""
    date: datetime.date
    effort: float
    fairness_score: float
    stress_level: Optional[int] = None
    fairness_rating: Optional[int] = None
    tag: str = "NORMAL"  # HARD, NORMAL, LIGHT


class StoryRecoveryInfo(BaseModel):
    """Recovery info."""
    is_recovery_day: bool
    recent_hard_days: int


class StorySwapDetails(BaseModel):
    """Swap details if applied."""
    original_route_id: Optional[UUID] = None
    swapped_with_driver_id: Optional[UUID] = None
    swapped_with_driver_name: Optional[str] = None


class StoryManualOverride(BaseModel):
    """Manual override info."""
    affected: bool
    details: Optional[str] = None


class StoryNegotiationInfo(BaseModel):
    """Negotiation info from Phase 4.2."""
    liaison_decision: Optional[str] = None  # ACCEPT, COUNTER, FORCE_ACCEPT
    liaison_reason: Optional[str] = None
    swap_applied: bool = False
    swap_details: Optional[StorySwapDetails] = None
    manual_override: StoryManualOverride


class StoryTimelineEvent(BaseModel):
    """Agent timeline event for driver story."""
    timestamp: datetime.datetime
    agent_name: str
    step_type: str
    description: str


class StoryGlobalMetrics(BaseModel):
    """Global metrics for the allocation run."""
    gini_index: float
    std_dev: float
    max_gap: float
    avg_effort: float


class StoryAllocationRun(BaseModel):
    """Allocation run for story."""
    id: UUID
    global_metrics: StoryGlobalMetrics


class DriverAllocationStoryResponse(BaseModel):
    """Response for GET /api/v1/admin/driver_allocation_story."""
    driver: StoryDriverInfo
    date: datetime.date
    allocation_run: StoryAllocationRun
    today: StoryTodayInfo
    history_last_7_days: List[StoryHistoryDay]
    recovery: StoryRecoveryInfo
    negotiation: StoryNegotiationInfo
    agent_timeline_slice: List[StoryTimelineEvent]

