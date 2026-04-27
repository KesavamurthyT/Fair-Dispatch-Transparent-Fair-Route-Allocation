"""
DeliveryLog database model.
Represents a delivery attempt at a given stop.
"""

import enum
import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import String, Integer, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, GUID

if TYPE_CHECKING:
    from app.models.assignment import Assignment
    from app.models.route import Route
    from app.models.driver import Driver
    from app.models.package import Package


class DeliveryStatus(str, enum.Enum):
    """Status of a delivery attempt."""
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class DeliveryIssueType(str, enum.Enum):
    """Type of issue encountered during delivery."""
    NONE = "NONE"
    NOT_AT_HOME = "NOT_AT_HOME"
    WRONG_ADDRESS = "WRONG_ADDRESS"
    SAFETY = "SAFETY"
    ACCESS_DENIED = "ACCESS_DENIED"
    OTHER = "OTHER"


class DeliveryLog(Base):
    """
    DeliveryLog model representing a delivery attempt at a given stop.
    Tracks delivery status, issues, and proof of delivery.
    """
    __tablename__ = "delivery_logs"

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
    package_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("packages.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[DeliveryStatus] = mapped_column(
        Enum(DeliveryStatus, native_enum=False),
        nullable=False,
        default=DeliveryStatus.DELIVERED,
    )
    issue_type: Mapped[DeliveryIssueType] = mapped_column(
        Enum(DeliveryIssueType, native_enum=False),
        nullable=False,
        default=DeliveryIssueType.NONE,
    )
    photo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    signature_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    assignment: Mapped["Assignment"] = relationship("Assignment")
    route: Mapped["Route"] = relationship("Route")
    driver: Mapped["Driver"] = relationship("Driver")
    package: Mapped[Optional["Package"]] = relationship("Package")

    def __repr__(self) -> str:
        return f"<DeliveryLog(id={self.id}, stop_order={self.stop_order}, status={self.status})>"
