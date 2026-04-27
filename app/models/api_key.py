"""
API Key models for key management and usage tracking.
"""

import uuid
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

from sqlalchemy import String, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, GUID


class ApiKey(Base):
    """
    API Key model for managing access credentials.
    Stores a hashed version of the key for secure lookup.
    """
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_prefix: Mapped[str] = mapped_column(
        String(12), nullable=False, index=True,
    )
    key_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    total_allocations: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow,
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    # Relationships
    usage_logs: Mapped[List["ApiKeyUsageLog"]] = relationship(
        "ApiKeyUsageLog",
        back_populates="api_key",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ApiKey(id={self.id}, name={self.name}, prefix={self.key_prefix})>"


class ApiKeyUsageLog(Base):
    """
    Per-request usage log for API keys.
    """
    __tablename__ = "api_key_usage_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    response_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True,
    )

    # Relationships
    api_key: Mapped["ApiKey"] = relationship(
        "ApiKey", back_populates="usage_logs",
    )

    def __repr__(self) -> str:
        return f"<ApiKeyUsageLog(key_id={self.api_key_id}, endpoint={self.endpoint})>"
