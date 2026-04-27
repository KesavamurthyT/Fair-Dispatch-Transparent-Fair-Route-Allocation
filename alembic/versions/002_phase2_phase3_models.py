"""Add Phase 2 and Phase 3 models

Revision ID: 002_phase2_phase3_models
Revises: 001_initial_schema
Create Date: 2026-02-04

Creates tables:
- delivery_logs
- route_swap_requests
- stop_issues
- appeals
- manual_overrides
- fairness_configs
- allocation_runs
- decision_logs

Modifies tables:
- driver_feedback (adds new columns)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision = '002_phase2_phase3_models'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create allocation_runs table first (referenced by decision_logs)
    op.create_table(
        'allocation_runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('date', sa.Date(), nullable=False, index=True),
        sa.Column('num_drivers', sa.Integer(), nullable=False, default=0),
        sa.Column('num_routes', sa.Integer(), nullable=False, default=0),
        sa.Column('num_packages', sa.Integer(), nullable=False, default=0),
        sa.Column('global_gini_index', sa.Float(), default=0.0),
        sa.Column('global_std_dev', sa.Float(), default=0.0),
        sa.Column('global_max_gap', sa.Float(), default=0.0),
        sa.Column('status', sa.Enum('SUCCESS', 'FAILED', name='allocationrunstatus'), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
    )

    # Create delivery_logs table
    op.create_table(
        'delivery_logs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('assignment_id', UUID(as_uuid=True), sa.ForeignKey('assignments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('route_id', UUID(as_uuid=True), sa.ForeignKey('routes.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('driver_id', UUID(as_uuid=True), sa.ForeignKey('drivers.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('stop_order', sa.Integer(), nullable=False),
        sa.Column('package_id', UUID(as_uuid=True), sa.ForeignKey('packages.id', ondelete='SET NULL'), nullable=True),
        sa.Column('status', sa.Enum('DELIVERED', 'FAILED', 'PARTIAL', name='deliverystatus'), nullable=False),
        sa.Column('issue_type', sa.Enum('NONE', 'NOT_AT_HOME', 'WRONG_ADDRESS', 'SAFETY', 'ACCESS_DENIED', 'OTHER', name='deliveryissuetype'), nullable=False),
        sa.Column('photo_url', sa.String(500), nullable=True),
        sa.Column('signature_data', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
    )

    # Create route_swap_requests table
    op.create_table(
        'route_swap_requests',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('from_driver_id', UUID(as_uuid=True), sa.ForeignKey('drivers.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('to_driver_id', UUID(as_uuid=True), sa.ForeignKey('drivers.id', ondelete='SET NULL'), nullable=True),
        sa.Column('assignment_id', UUID(as_uuid=True), sa.ForeignKey('assignments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('preferred_date', sa.Date(), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'APPROVED', 'REJECTED', 'CANCELLED', name='swaprequeststatus'), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    # Create stop_issues table
    op.create_table(
        'stop_issues',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('assignment_id', UUID(as_uuid=True), sa.ForeignKey('assignments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('route_id', UUID(as_uuid=True), sa.ForeignKey('routes.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('driver_id', UUID(as_uuid=True), sa.ForeignKey('drivers.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('stop_order', sa.Integer(), nullable=False),
        sa.Column('issue_type', sa.Enum('NAVIGATION', 'SAFETY', 'TIME_WINDOW', 'CUSTOMER_UNAVAILABLE', 'OTHER', name='stopissuetype'), nullable=False),
        sa.Column('notes', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # Create appeals table
    op.create_table(
        'appeals',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('driver_id', UUID(as_uuid=True), sa.ForeignKey('drivers.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('assignment_id', UUID(as_uuid=True), sa.ForeignKey('assignments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'APPROVED', 'REJECTED', 'RESOLVED', name='appealstatus'), nullable=False),
        sa.Column('admin_note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    # Create manual_overrides table
    op.create_table(
        'manual_overrides',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('allocation_run_id', UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('old_driver_id', UUID(as_uuid=True), sa.ForeignKey('drivers.id', ondelete='SET NULL'), nullable=True),
        sa.Column('new_driver_id', UUID(as_uuid=True), sa.ForeignKey('drivers.id', ondelete='SET NULL'), nullable=True),
        sa.Column('route_id', UUID(as_uuid=True), sa.ForeignKey('routes.id', ondelete='SET NULL'), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('before_metrics', sa.JSON(), nullable=True),
        sa.Column('after_metrics', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # Create fairness_configs table
    op.create_table(
        'fairness_configs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, index=True),
        sa.Column('workload_weight_packages', sa.Float(), default=1.0),
        sa.Column('workload_weight_weight_kg', sa.Float(), default=0.5),
        sa.Column('workload_weight_difficulty', sa.Float(), default=10.0),
        sa.Column('workload_weight_time', sa.Float(), default=0.2),
        sa.Column('gini_threshold', sa.Float(), default=0.33),
        sa.Column('stddev_threshold', sa.Float(), default=25.0),
        sa.Column('max_gap_threshold', sa.Float(), default=25.0),
        sa.Column('recovery_mode_enabled', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    # Create decision_logs table
    op.create_table(
        'decision_logs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('allocation_run_id', UUID(as_uuid=True), sa.ForeignKey('allocation_runs.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('agent_name', sa.String(100), nullable=False, index=True),
        sa.Column('step_type', sa.String(100), nullable=False),
        sa.Column('input_snapshot', sa.JSON(), nullable=True),
        sa.Column('output_snapshot', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # Add new columns to driver_feedback table for Phase 2 extended feedback
    op.add_column('driver_feedback', sa.Column('route_difficulty_self_report', sa.Integer(), nullable=True))
    op.add_column('driver_feedback', sa.Column('would_take_similar_route_again', sa.Boolean(), nullable=True))
    op.add_column('driver_feedback', sa.Column('most_unfair_aspect', sa.String(100), nullable=True))


def downgrade() -> None:
    # Remove new columns from driver_feedback
    op.drop_column('driver_feedback', 'most_unfair_aspect')
    op.drop_column('driver_feedback', 'would_take_similar_route_again')
    op.drop_column('driver_feedback', 'route_difficulty_self_report')

    # Drop tables in reverse order
    op.drop_table('decision_logs')
    op.drop_table('fairness_configs')
    op.drop_table('manual_overrides')
    op.drop_table('appeals')
    op.drop_table('stop_issues')
    op.drop_table('route_swap_requests')
    op.drop_table('delivery_logs')
    op.drop_table('allocation_runs')

    # Drop enums
    op.execute("DROP TYPE IF EXISTS allocationrunstatus")
    op.execute("DROP TYPE IF EXISTS deliverystatus")
    op.execute("DROP TYPE IF EXISTS deliveryissuetype")
    op.execute("DROP TYPE IF EXISTS swaprequeststatus")
    op.execute("DROP TYPE IF EXISTS stopissuetype")
    op.execute("DROP TYPE IF EXISTS appealstatus")
