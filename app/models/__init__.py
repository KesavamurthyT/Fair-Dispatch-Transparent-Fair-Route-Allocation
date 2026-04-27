"""Models package initialization - imports all models for easy access."""

from app.models.driver import Driver, DriverStatsDaily, DriverFeedback, VehicleType, PreferredLanguage
from app.models.package import Package
from app.models.route import Route, RoutePackage
from app.models.assignment import Assignment
from app.models.delivery_log import DeliveryLog, DeliveryStatus, DeliveryIssueType
from app.models.route_swap import RouteSwapRequest, SwapRequestStatus
from app.models.stop_issue import StopIssue, StopIssueType
from app.models.appeal import Appeal, AppealStatus
from app.models.manual_override import ManualOverride
from app.models.fairness_config import FairnessConfig
from app.models.allocation_run import AllocationRun, AllocationRunStatus
from app.models.decision_log import DecisionLog
from app.models.learning_episode import LearningEpisode
from app.models.driver_effort_model import DriverEffortModel
from app.models.api_key import ApiKey, ApiKeyUsageLog

__all__ = [
    # Phase 1 models
    "Driver",
    "DriverStatsDaily",
    "DriverFeedback",
    "VehicleType",
    "PreferredLanguage",
    "Package",
    "Route",
    "RoutePackage",
    "Assignment",
    # Phase 2 models
    "DeliveryLog",
    "DeliveryStatus",
    "DeliveryIssueType",
    "RouteSwapRequest",
    "SwapRequestStatus",
    "StopIssue",
    "StopIssueType",
    # Phase 3 models
    "Appeal",
    "AppealStatus",
    "ManualOverride",
    "FairnessConfig",
    "AllocationRun",
    "AllocationRunStatus",
    "DecisionLog",
    # Phase 8 models
    "LearningEpisode",
    "DriverEffortModel",
    # API Key Management
    "ApiKey",
    "ApiKeyUsageLog",
]

