"""
AllocationRun database model.
High-level metadata per /allocate execution.
"""

import enum
import uuid
from datetime import datetime, date
from typing import Optional

from sqlalchemy import Integer, Float, Text, String, Date, DateTime, Enum, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base, GUID


class AllocationRunStatus(str, enum.Enum):
    """Status of an allocation run."""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class AllocationRun(Base):
    """
    AllocationRun model for high-level allocation metadata.
    Tracks each allocation execution with global metrics.
    """
    __tablename__ = "allocation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    num_drivers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_routes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_packages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Global fairness metrics
    global_gini_index: Mapped[float] = mapped_column(Float, default=0.0)
    global_std_dev: Mapped[float] = mapped_column(Float, default=0.0)
    global_max_gap: Mapped[float] = mapped_column(Float, default=0.0)
    
    status: Mapped[AllocationRunStatus] = mapped_column(
        Enum(AllocationRunStatus, native_enum=False),
        nullable=False,
        default=AllocationRunStatus.SUCCESS,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Serialized request payload for async worker processing
    payload_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Fix 2: model version used during this allocation (reproducibility)
    model_version_used: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )
    
    # Fix 4: idempotency key — prevents duplicate jobs on client retry
    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String(100), unique=True, nullable=True, index=True
    )
    
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
    )

    def __repr__(self) -> str:
        return f"<AllocationRun(id={self.id}, date={self.date}, status={self.status})>"
