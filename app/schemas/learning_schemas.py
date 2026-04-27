"""
Pydantic schemas for Phase 8 Learning Agent APIs.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field


class BanditArmStatistics(BaseModel):
    """Statistics for a single bandit arm."""
    arm_idx: int
    config_hash: str
    config: Dict[str, Any]
    alpha: float
    beta: float
    samples: int
    mean_reward: float


class BanditStatistics(BaseModel):
    """Overall bandit statistics."""
    total_samples: int
    explored_arms: int
    total_arms: int = 81


class LearningStatusResponse(BaseModel):
    """Response for GET /api/v1/admin/learning_status."""
    current_config: Optional[Dict[str, Any]] = None
    top_performing_configs: List[BanditArmStatistics] = []
    driver_models_active: int = 0
    avg_prediction_mse: float = 0.0
    recent_episodes_7d: int = 0
    total_arms: int = 81
    bandit_statistics: BanditStatistics


class LearningEpisodeResponse(BaseModel):
    """Response schema for a learning episode."""
    id: UUID
    allocation_run_id: UUID
    config_hash: str
    fairness_config: Dict[str, Any]
    arm_idx: int
    num_drivers: int
    num_routes: int
    episode_reward: Optional[float] = None
    reward_computed_at: Optional[datetime] = None
    alpha_prior: float
    beta_prior: float
    samples_count: int
    is_experimental: bool
    avg_fairness_rating: Optional[float] = None
    avg_stress_level: Optional[float] = None
    completion_rate: Optional[float] = None
    feedback_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class LearningEpisodesListResponse(BaseModel):
    """Paginated list of learning episodes."""
    episodes: List[LearningEpisodeResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class DriverModelStatusResponse(BaseModel):
    """Status of a driver's effort model."""
    driver_id: UUID
    model_version: int
    training_samples: int
    current_mse: Optional[float] = None
    r2_score: Optional[float] = None
    mse_history: Optional[Dict[str, Any]] = None
    active: bool
    last_trained_at: Optional[datetime] = None


class ForceConfigRequest(BaseModel):
    """Request body for forcing a specific config."""
    gini_threshold: float = Field(ge=0.1, le=0.5, default=0.33)
    stddev_threshold: float = Field(ge=10.0, le=50.0, default=25.0)
    recovery_lightening_factor: float = Field(ge=0.3, le=1.0, default=0.7)
    ev_charging_penalty_weight: float = Field(ge=0.0, le=1.0, default=0.3)
    max_gap_threshold: float = Field(ge=10.0, le=50.0, default=25.0)
    reason: str = Field(min_length=1, max_length=500)


class ForceConfigResponse(BaseModel):
    """Response after forcing a config."""
    status: str
    message: str
    config_applied: Dict[str, Any]
    previous_config: Optional[Dict[str, Any]] = None


class TriggerLearningRequest(BaseModel):
    """Request to manually trigger learning pipeline."""
    process_episodes: bool = True
    update_models: bool = True
    select_config: bool = True


class TriggerLearningResponse(BaseModel):
    """Response after triggering learning pipeline."""
    status: str
    episodes_processed: int = 0
    rewards_computed: int = 0
    models_updated: int = 0
    config_selection: Optional[str] = None
    duration_seconds: float = 0.0
    errors: List[str] = []


class DriverModelUpdateRequest(BaseModel):
    """Request to update a specific driver's model."""
    driver_id: UUID


class DriverModelUpdateResponse(BaseModel):
    """Response after updating a driver's model."""
    status: str
    driver_id: UUID
    model_version: Optional[int] = None
    training_samples: Optional[int] = None
    mse: Optional[float] = None
    r2_score: Optional[float] = None
    reason: Optional[str] = None


class AllDriverModelsListResponse(BaseModel):
    """List of all driver models."""
    models: List[DriverModelStatusResponse]
    total: int
    active_count: int
    avg_mse: Optional[float] = None
