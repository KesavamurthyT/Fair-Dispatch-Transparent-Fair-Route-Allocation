"""Alembic migration: add hardening columns to allocation_runs and driver_effort_models.

Revision ID: a1b2c3d4e5f6
Revises: (head)
Create Date: 2026-02-20

Changes:
  1. allocation_runs.model_version_used  (VARCHAR 100, nullable) - Fix 2
  2. allocation_runs.idempotency_key     (VARCHAR 100, unique, nullable, indexed) - Fix 4
  3. driver_effort_models.model_path     (VARCHAR 500, nullable) - previously added
  4. driver_effort_models.model_checksum (VARCHAR 64,  nullable) - previously added
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "a1b2c3d4e5f6"
down_revision = None  # set to the actual previous revision in your project
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── allocation_runs ──────────────────────────────────────────────────────
    op.add_column(
        "allocation_runs",
        sa.Column("model_version_used", sa.String(100), nullable=True),
    )
    op.add_column(
        "allocation_runs",
        sa.Column("idempotency_key", sa.String(100), nullable=True),
    )
    op.create_unique_constraint(
        "uq_allocation_runs_idempotency_key",
        "allocation_runs",
        ["idempotency_key"],
    )
    op.create_index(
        "ix_allocation_runs_idempotency_key",
        "allocation_runs",
        ["idempotency_key"],
        unique=True,
    )

    # ── allocation_runs.status enum: add QUEUED / RUNNING (idempotent) ───────
    # Note: PostgreSQL requires ALTER TYPE to add new enum values.
    # SQLite (dev) handles this automatically.
    # For Postgres, run manually if not already present:
    #   ALTER TYPE allocationrunstatus ADD VALUE IF NOT EXISTS 'QUEUED';
    #   ALTER TYPE allocationrunstatus ADD VALUE IF NOT EXISTS 'RUNNING';

    # ── driver_effort_models: replace model_pickle with safe storage ─────────
    # Only run if model_pickle still exists (idempotent guard)
    conn = op.get_bind()
    insp = sa.inspect(conn)
    cols = {c["name"] for c in insp.get_columns("driver_effort_models")}

    if "model_pickle" in cols:
        op.drop_column("driver_effort_models", "model_pickle")

    if "model_path" not in cols:
        op.add_column(
            "driver_effort_models",
            sa.Column("model_path", sa.String(500), nullable=True),
        )
    if "model_checksum" not in cols:
        op.add_column(
            "driver_effort_models",
            sa.Column("model_checksum", sa.String(64), nullable=True),
        )


def downgrade() -> None:
    # ── allocation_runs ──────────────────────────────────────────────────────
    op.drop_index("ix_allocation_runs_idempotency_key", table_name="allocation_runs")
    op.drop_constraint(
        "uq_allocation_runs_idempotency_key",
        "allocation_runs",
        type_="unique",
    )
    op.drop_column("allocation_runs", "idempotency_key")
    op.drop_column("allocation_runs", "model_version_used")

    # ── driver_effort_models: restore pickle column (dev only) ───────────────
    op.drop_column("driver_effort_models", "model_checksum")
    op.drop_column("driver_effort_models", "model_path")
    op.add_column(
        "driver_effort_models",
        sa.Column("model_pickle", sa.LargeBinary(), nullable=True),
    )
