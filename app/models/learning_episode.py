"""
LearningEpisode database model.
Tracks bandit learning episodes per allocation run for Thompson Sampling.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Float, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base, GUID


class LearningEpisode(Base):
    """
    LearningEpisode model for tracking bandit learning episodes.
    Each allocation run creates one episode; reward is computed 24h+ later.
    """
    __tablename__ = "learning_episodes"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    allocation_run_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("allocation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,  # One episode per allocation run
    )
    
    # Bandit arm identification
    config_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    fairness_config: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )
    arm_idx: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    
    # Episode context
    num_drivers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_routes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Reward (computed later by cron job)
    episode_reward: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    reward_computed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
    )
    
    # Thompson Sampling priors at time of selection
    alpha_prior: Mapped[float] = mapped_column(Float, default=1.0)
    beta_prior: Mapped[float] = mapped_column(Float, default=1.0)
    samples_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # A/B testing flag
    is_experimental: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        index=True,
    )
    
    # Feedback aggregation (stored for debugging)
    avg_fairness_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_stress_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    completion_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feedback_count: Mapped[int] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<LearningEpisode(id={self.id}, reward={self.episode_reward})>"
