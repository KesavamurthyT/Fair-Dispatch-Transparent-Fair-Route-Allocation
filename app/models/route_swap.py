"""
RouteSwapRequest database model.
Represents a driver requesting to swap or change a route.
"""

import enum
import uuid
from datetime import datetime, date
from typing import Optional, TYPE_CHECKING

from sqlalchemy import Text, Date, DateTime, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, GUID

if TYPE_CHECKING:
    from app.models.driver import Driver
    from app.models.assignment import Assignment


class SwapRequestStatus(str, enum.Enum):
    """Status of a route swap request."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class RouteSwapRequest(Base):
    """
    RouteSwapRequest model for driver route swap/change requests.
    Tracks request status and allows drivers to request lighter routes.
    """
    __tablename__ = "route_swap_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    from_driver_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("drivers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    to_driver_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("drivers.id", ondelete="SET NULL"),
        nullable=True,
    )
    assignment_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("assignments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    preferred_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[SwapRequestStatus] = mapped_column(
        Enum(SwapRequestStatus, native_enum=False),
        nullable=False,
        default=SwapRequestStatus.PENDING,
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

    # Relationships
    from_driver: Mapped["Driver"] = relationship(
        "Driver",
        foreign_keys=[from_driver_id],
    )
    to_driver: Mapped[Optional["Driver"]] = relationship(
        "Driver",
        foreign_keys=[to_driver_id],
    )
    assignment: Mapped["Assignment"] = relationship("Assignment")

    def __repr__(self) -> str:
        return f"<RouteSwapRequest(id={self.id}, from_driver={self.from_driver_id}, status={self.status})>"
