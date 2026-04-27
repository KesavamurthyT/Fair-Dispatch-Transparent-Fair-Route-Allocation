"""
Assignment database model.
Represents the allocation of a route to a driver.
"""

import uuid
from datetime import datetime, date
from typing import Optional, List, TYPE_CHECKING

from sqlalchemy import Float, Text, Date, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, GUID

if TYPE_CHECKING:
    from app.models.driver import Driver, DriverFeedback
    from app.models.route import Route


class Assignment(Base):
    """
    Assignment model representing the allocation of a route to a driver.
    Contains workload and fairness scores plus explanation.
    """
    __tablename__ = "assignments"
    
    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    driver_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("drivers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    route_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("routes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workload_score: Mapped[float] = mapped_column(Float, default=0.0)
    fairness_score: Mapped[float] = mapped_column(Float, default=1.0)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    driver_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    admin_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    allocation_run_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        nullable=False,
        index=True,
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    
    # Relationships
    driver: Mapped["Driver"] = relationship("Driver", back_populates="assignments")
    route: Mapped["Route"] = relationship("Route", back_populates="assignments")
    feedback: Mapped[List["DriverFeedback"]] = relationship(
        "DriverFeedback",
        back_populates="assignment",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Assignment(id={self.id}, driver_id={self.driver_id}, route_id={self.route_id})>"
