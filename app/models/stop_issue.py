"""
StopIssue database model.
Represents an issue at a specific stop.
"""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Integer, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, GUID

if TYPE_CHECKING:
    from app.models.assignment import Assignment
    from app.models.route import Route
    from app.models.driver import Driver


class StopIssueType(str, enum.Enum):
    """Type of issue at a stop."""
    NAVIGATION = "NAVIGATION"
    SAFETY = "SAFETY"
    TIME_WINDOW = "TIME_WINDOW"
    CUSTOMER_UNAVAILABLE = "CUSTOMER_UNAVAILABLE"
    OTHER = "OTHER"


class StopIssue(Base):
    """
    StopIssue model representing an issue at a specific stop.
    Used to track and report problems during deliveries.
    """
    __tablename__ = "stop_issues"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    assignment_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("assignments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    route_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("routes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    driver_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("drivers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stop_order: Mapped[int] = mapped_column(Integer, nullable=False)
    issue_type: Mapped[StopIssueType] = mapped_column(
        Enum(StopIssueType, native_enum=False),
        nullable=False,
    )
    notes: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    assignment: Mapped["Assignment"] = relationship("Assignment")
    route: Mapped["Route"] = relationship("Route")
    driver: Mapped["Driver"] = relationship("Driver")

    def __repr__(self) -> str:
        return f"<StopIssue(id={self.id}, stop_order={self.stop_order}, type={self.issue_type})>"
