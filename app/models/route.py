"""
Route database models.
Includes Route and RoutePackage (association table) models.
"""

import uuid
from datetime import datetime, date
from typing import Optional, List, TYPE_CHECKING

from sqlalchemy import Integer, Float, Date, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, GUID

if TYPE_CHECKING:
    from app.models.package import Package
    from app.models.assignment import Assignment


class Route(Base):
    """
    Route model representing a delivery route (cluster of packages).
    Contains aggregated metrics about the route.
    """
    __tablename__ = "routes"
    
    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    total_weight_kg: Mapped[float] = mapped_column(Float, default=0.0)
    num_packages: Mapped[int] = mapped_column(Integer, default=0)
    num_stops: Mapped[int] = mapped_column(Integer, default=0)
    route_difficulty_score: Mapped[float] = mapped_column(Float, default=1.0)
    estimated_time_minutes: Mapped[int] = mapped_column(Integer, default=60)
    
    # Distance for EV range calculations (Phase 7)
    total_distance_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Run scoping - links route to specific allocation run
    # Nullable for backward compatibility with existing routes
    allocation_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("allocation_runs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    
    # Relationships
    route_packages: Mapped[List["RoutePackage"]] = relationship(
        "RoutePackage",
        back_populates="route",
        cascade="all, delete-orphan",
    )
    assignments: Mapped[List["Assignment"]] = relationship(
        "Assignment",
        back_populates="route",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Route(id={self.id}, cluster_id={self.cluster_id}, packages={self.num_packages})>"


class RoutePackage(Base):
    """
    Association table linking routes to packages with stop order.
    Represents which packages belong to which route and in what order.
    """
    __tablename__ = "route_packages"
    
    route_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("routes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    package_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("packages.id", ondelete="CASCADE"),
        primary_key=True,
    )
    stop_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Relationships
    route: Mapped["Route"] = relationship("Route", back_populates="route_packages")
    package: Mapped["Package"] = relationship("Package", back_populates="route_packages")
    
    def __repr__(self) -> str:
        return f"<RoutePackage(route_id={self.route_id}, package_id={self.package_id}, order={self.stop_order})>"
