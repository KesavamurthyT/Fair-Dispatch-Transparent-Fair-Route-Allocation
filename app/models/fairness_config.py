"""
FairnessConfig database model.
Stores fairness engine weights and thresholds.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base, GUID


class FairnessConfig(Base):
    """
    FairnessConfig model for storing fairness engine weights and thresholds.
    Only one config should be active at a time.
    """
    __tablename__ = "fairness_configs"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )
    
    # Workload weights
    workload_weight_packages: Mapped[float] = mapped_column(Float, default=1.0)
    workload_weight_weight_kg: Mapped[float] = mapped_column(Float, default=0.5)
    workload_weight_difficulty: Mapped[float] = mapped_column(Float, default=10.0)
    workload_weight_time: Mapped[float] = mapped_column(Float, default=0.2)
    
    # Fairness thresholds
    gini_threshold: Mapped[float] = mapped_column(Float, default=0.33)
    stddev_threshold: Mapped[float] = mapped_column(Float, default=25.0)
    max_gap_threshold: Mapped[float] = mapped_column(Float, default=25.0)
    
    # Recovery mode (Phase 7)
    recovery_mode_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    complexity_debt_hard_threshold: Mapped[float] = mapped_column(Float, default=2.0)
    recovery_lightening_factor: Mapped[float] = mapped_column(Float, default=0.7)
    recovery_penalty_weight: Mapped[float] = mapped_column(Float, default=3.0)
    
    # EV config (Phase 7)
    ev_charging_penalty_weight: Mapped[float] = mapped_column(Float, default=0.3)
    ev_safety_margin_pct: Mapped[float] = mapped_column(Float, default=10.0)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<FairnessConfig(id={self.id}, is_active={self.is_active})>"

