"""
DecisionLog database model.
Stores per-agent step logs for workflow visualization.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, GUID


class DecisionLog(Base):
    """
    DecisionLog model for agent workflow visualization.
    Records per-agent step logs during allocation with input/output snapshots.
    """
    __tablename__ = "decision_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    allocation_run_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("allocation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    step_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    input_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    allocation_run = relationship("AllocationRun")

    def __repr__(self) -> str:
        return f"<DecisionLog(id={self.id}, agent={self.agent_name}, step={self.step_type})>"
