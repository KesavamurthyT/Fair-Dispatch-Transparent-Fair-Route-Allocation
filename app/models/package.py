"""
Package database model.
Represents delivery packages with location and priority info.
"""

import enum
import uuid
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

from sqlalchemy import String, Float, Integer, Text, DateTime, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, GUID

if TYPE_CHECKING:
    from app.models.route import RoutePackage


class PackagePriority(str, enum.Enum):
    """Priority levels for package delivery."""
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXPRESS = "EXPRESS"


class Package(Base):
    """
    Package model representing items to be delivered.
    Contains weight, fragility, location, and priority information.
    """
    __tablename__ = "packages"
    
    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    external_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    fragility_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)  # 1-5
    address: Mapped[str] = mapped_column(Text, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    priority: Mapped[PackagePriority] = mapped_column(
        Enum(PackagePriority, native_enum=False),
        default=PackagePriority.NORMAL,
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    
    # Relationships
    route_packages: Mapped[List["RoutePackage"]] = relationship(
        "RoutePackage",
        back_populates="package",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Package(id={self.id}, external_id={self.external_id})>"
