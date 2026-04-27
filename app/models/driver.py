"""
Driver-related database models.
Includes Driver, DriverStatsDaily, and DriverFeedback models.
"""

import enum
import uuid
from datetime import datetime, date
from typing import Optional, List, TYPE_CHECKING

from sqlalchemy import String, Float, Integer, Text, Date, DateTime, ForeignKey, Enum, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, GUID

if TYPE_CHECKING:
    from app.models.assignment import Assignment


class PreferredLanguage(str, enum.Enum):
    """Supported languages for driver communication."""
    EN = "en"
    TA = "ta"
    HI = "hi"
    TE = "te"
    KN = "kn"


class VehicleType(str, enum.Enum):
    """Types of vehicles used for delivery."""
    ICE = "ICE"
    EV = "EV"
    BICYCLE = "BICYCLE"


class Driver(Base):
    """
    Driver model representing delivery personnel.
    Stores personal info, vehicle details, and preferences.
    """
    __tablename__ = "drivers"
    
    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    external_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        unique=True,
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    whatsapp_number: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    preferred_language: Mapped[PreferredLanguage] = mapped_column(
        Enum(PreferredLanguage, native_enum=False),
        default=PreferredLanguage.EN,
    )
    vehicle_type: Mapped[VehicleType] = mapped_column(
        Enum(VehicleType, native_enum=False),
        default=VehicleType.ICE,
    )
    vehicle_capacity_kg: Mapped[float] = mapped_column(Float, default=100.0)
    license_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    ev_charging_pref: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # EV-specific fields (Phase 7)
    battery_range_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    charging_time_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    
    # Relationships
    daily_stats: Mapped[List["DriverStatsDaily"]] = relationship(
        "DriverStatsDaily",
        back_populates="driver",
        cascade="all, delete-orphan",
    )
    assignments: Mapped[List["Assignment"]] = relationship(
        "Assignment",
        back_populates="driver",
        cascade="all, delete-orphan",
    )
    feedback: Mapped[List["DriverFeedback"]] = relationship(
        "DriverFeedback",
        back_populates="driver",
        cascade="all, delete-orphan",
    )
    
    @property
    def is_ev(self) -> bool:
        """Check if driver uses an electric vehicle."""
        return self.vehicle_type == VehicleType.EV
    
    def __repr__(self) -> str:
        return f"<Driver(id={self.id}, name={self.name})>"


class DriverStatsDaily(Base):
    """
    Daily statistics for each driver.
    Tracks workload, routes, fairness metrics, and recovery state per day.
    """
    __tablename__ = "driver_stats_daily"
    
    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    driver_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("drivers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    avg_workload_score: Mapped[float] = mapped_column(Float, default=0.0)
    total_routes: Mapped[int] = mapped_column(Integer, default=0)
    gini_contribution: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reported_stress_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reported_fairness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Recovery tracking fields (Phase 7)
    is_hard_day: Mapped[bool] = mapped_column(Boolean, default=False)
    complexity_debt: Mapped[float] = mapped_column(Float, default=0.0)
    is_recovery_day: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Learning fields (Phase 8)
    predicted_effort: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_effort: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_error: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    allocation_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("allocation_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    # Relationships
    driver: Mapped["Driver"] = relationship("Driver", back_populates="daily_stats")
    
    def __repr__(self) -> str:
        return f"<DriverStatsDaily(driver_id={self.driver_id}, date={self.date})>"


class HardestAspect(str, enum.Enum):
    """Common difficult aspects of delivery work."""
    TRAFFIC = "traffic"
    PARKING = "parking"
    STAIRS = "stairs"
    WEATHER = "weather"
    HEAVY_LOAD = "heavy_load"
    CUSTOMER = "customer"
    NAVIGATION = "navigation"
    OTHER = "other"


class DriverFeedback(Base):
    """
    Feedback submitted by drivers after completing assignments.
    Used for learning and improving future allocations.
    """
    __tablename__ = "driver_feedback"
    
    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    driver_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("drivers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    assignment_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("assignments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    fairness_rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    stress_level: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-10
    tiredness_level: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    hardest_aspect: Mapped[Optional[HardestAspect]] = mapped_column(
        Enum(HardestAspect, native_enum=False),
        nullable=True,
    )
    comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Extended feedback fields (Phase 2)
    route_difficulty_self_report: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )  # 1-5 scale
    would_take_similar_route_again: Mapped[Optional[bool]] = mapped_column(
        nullable=True,
    )
    most_unfair_aspect: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    
    # Relationships
    driver: Mapped["Driver"] = relationship("Driver", back_populates="feedback")
    assignment: Mapped["Assignment"] = relationship("Assignment", back_populates="feedback")
    
    def __repr__(self) -> str:
        return f"<DriverFeedback(id={self.id}, driver_id={self.driver_id})>"
