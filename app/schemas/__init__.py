"""Schemas package initialization."""

from app.schemas.allocation import (
    PackageInput,
    DriverInput,
    WarehouseInput,
    AllocationRequest,
    RouteSummary,
    AssignmentResponse,
    GlobalFairness,
    AllocationResponse,
)
from app.schemas.driver import DriverResponse, DriverStatsResponse
from app.schemas.route import RouteResponse
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

# Phase 2 schemas
from app.schemas.driver_api import (
    TodayAssignmentResponse,
    DriverStatsWindowResponse,
    DeliveryLogRequest,
    DeliveryLogResponse,
    ExtendedFeedbackRequest,
    ExtendedFeedbackResponse,
    RouteSwapRequestCreate,
    RouteSwapRequestResponse,
    StopIssueRequest,
    StopIssueResponse,
)

# Phase 3 schemas
from app.schemas.admin import (
    HealthResponse,
    AllocationRunResponse,
    AllocationRunsListResponse,
    AdminAssignmentsListResponse,
    FairnessMetricsResponse,
    WorkloadHeatmapResponse,
    DriverHistoryResponse,
    AppealsListResponse,
    AppealDecisionRequest,
    AppealDecisionResponse,
    ManualOverrideRequest,
    ManualOverrideResponse,
    FairnessConfigRequest,
    FairnessConfigResponse,
    AgentTimelineResponse,
)

# Phase 4.1 - Agent schemas
from app.schemas.agent_schemas import (
    EffortMatrixResult,
    RoutePlanResult,
    FairnessCheckResult,
    EffortWeights,
    FairnessThresholds,
)

__all__ = [
    # Phase 1
    "PackageInput",
    "DriverInput",
    "WarehouseInput",
    "AllocationRequest",
    "RouteSummary",
    "AssignmentResponse",
    "GlobalFairness",
    "AllocationResponse",
    "DriverResponse",
    "DriverStatsResponse",
    "RouteResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    # Phase 2
    "TodayAssignmentResponse",
    "DriverStatsWindowResponse",
    "DeliveryLogRequest",
    "DeliveryLogResponse",
    "ExtendedFeedbackRequest",
    "ExtendedFeedbackResponse",
    "RouteSwapRequestCreate",
    "RouteSwapRequestResponse",
    "StopIssueRequest",
    "StopIssueResponse",
    # Phase 3
    "HealthResponse",
    "AllocationRunResponse",
    "AllocationRunsListResponse",
    "AdminAssignmentsListResponse",
    "FairnessMetricsResponse",
    "WorkloadHeatmapResponse",
    "DriverHistoryResponse",
    "AppealsListResponse",
    "AppealDecisionRequest",
    "AppealDecisionResponse",
    "ManualOverrideRequest",
    "ManualOverrideResponse",
    "FairnessConfigRequest",
    "FairnessConfigResponse",
    "AgentTimelineResponse",
    # Phase 4.1 - Agent schemas
    "EffortMatrixResult",
    "RoutePlanResult",
    "FairnessCheckResult",
    "EffortWeights",
    "FairnessThresholds",
]
