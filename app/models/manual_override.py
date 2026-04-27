"""
ManualOverride database model.
Captures manual admin interventions in allocations.
"""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, GUID

if TYPE_CHECKING:
    from app.models.driver import Driver
    from app.models.route import Route


class ManualOverride(Base):
    """
    ManualOverride model for admin manual interventions.
    Records route reassignments with before/after fairness metrics.
    """
    __tablename__ = "manual_overrides"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    allocation_run_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        nullable=False,
        index=True,
    )
    old_driver_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("drivers.id", ondelete="SET NULL"),
        nullable=True,
    )
    new_driver_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("drivers.id", ondelete="SET NULL"),
        nullable=True,
    )
    route_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("routes.id", ondelete="SET NULL"),
        nullable=True,
    )
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    before_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    after_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    old_driver: Mapped[Optional["Driver"]] = relationship(
        "Driver",
        foreign_keys=[old_driver_id],
    )
    new_driver: Mapped[Optional["Driver"]] = relationship(
        "Driver",
        foreign_keys=[new_driver_id],
    )
    route: Mapped[Optional["Route"]] = relationship("Route")

    def __repr__(self) -> str:
        return f"<ManualOverride(id={self.id}, old_driver={self.old_driver_id}, new_driver={self.new_driver_id})>"
