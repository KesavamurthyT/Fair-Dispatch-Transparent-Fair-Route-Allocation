"""
DriverEffortModel database model.
Stores per-driver XGBoost models for personalized effort prediction.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Float, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base, GUID


class DriverEffortModel(Base):
    """
    DriverEffortModel stores per-driver XGBoost model metadata.
    Models are saved as safe JSON files on disk via xgb.save_model().
    NO PICKLE storage is used.
    """
    __tablename__ = "driver_effort_models"

    driver_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("drivers.id", ondelete="CASCADE"),
        primary_key=True,
    )
    
    model_version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )
    
    # Safe model storage: filesystem path to JSON model artifact
    model_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    
    # SHA-256 checksum for model integrity verification
    model_checksum: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )
    
    # Training metadata
    training_samples: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )
    feature_names: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
    )
    
    # Performance tracking
    mse_history: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        default=list,
    )  # List of last 10 MSE values
    current_mse: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    r2_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    
    # Model state
    active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        index=True,
    )
    
    last_trained_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
    )
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
        return f"<DriverEffortModel(driver_id={self.driver_id}, v{self.model_version})>"
