"""Add driver_explanation and admin_explanation columns to assignments.

Revision ID: 004
Revises: 003_add_pending_status
Create Date: 2026-02-04
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003_add_pending_status'
branch_labels = None
depends_on = None


def upgrade():
    # Add driver_explanation column
    op.add_column(
        'assignments',
        sa.Column('driver_explanation', sa.Text(), nullable=True)
    )
    
    # Add admin_explanation column
    op.add_column(
        'assignments',
        sa.Column('admin_explanation', sa.Text(), nullable=True)
    )
    
    # Backfill: copy existing explanation to driver_explanation
    op.execute("""
        UPDATE assignments 
        SET driver_explanation = explanation 
        WHERE explanation IS NOT NULL
    """)


def downgrade():
    op.drop_column('assignments', 'admin_explanation')
    op.drop_column('assignments', 'driver_explanation')
