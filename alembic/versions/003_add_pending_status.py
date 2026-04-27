"""Add PENDING status to allocation_run_status enum.

Revision ID: 003_add_pending_status
Revises: 002_phase2_phase3_models
Create Date: 2026-02-04
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003_add_pending_status'
down_revision = '002_phase2_phase3_models'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add PENDING to allocationrunstatus enum."""
    # For PostgreSQL, we need to add the new value to the enum type
    op.execute("ALTER TYPE allocationrunstatus ADD VALUE IF NOT EXISTS 'PENDING' BEFORE 'SUCCESS'")


def downgrade() -> None:
    """Note: PostgreSQL does not support removing enum values directly.
    
    To fully downgrade, you would need to:
    1. Create a new enum without PENDING
    2. Update all PENDING rows to another status
    3. Alter the column to use the new enum
    4. Drop the old enum
    
    For simplicity, this downgrade is a no-op.
    """
    pass
