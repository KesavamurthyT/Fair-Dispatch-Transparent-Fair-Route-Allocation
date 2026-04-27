"""Initial schema - Create all tables

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-02-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE preferredlanguage AS ENUM ('en', 'ta', 'hi', 'te', 'kn')")
    op.execute("CREATE TYPE vehicletype AS ENUM ('ICE', 'EV', 'BICYCLE')")
    op.execute("CREATE TYPE packagepriority AS ENUM ('NORMAL', 'HIGH', 'EXPRESS')")
    op.execute("CREATE TYPE hardestaspect AS ENUM ('traffic', 'parking', 'stairs', 'weather', 'heavy_load', 'customer', 'navigation', 'other')")
    
    # Create drivers table
    op.create_table(
        'drivers',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('external_id', sa.String(100), unique=True, nullable=True, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('phone', sa.String(20), nullable=True),
        sa.Column('whatsapp_number', sa.String(20), nullable=True),
        sa.Column('preferred_language', postgresql.ENUM('en', 'ta', 'hi', 'te', 'kn', name='preferredlanguage', create_type=False), nullable=False, server_default='en'),
        sa.Column('vehicle_type', postgresql.ENUM('ICE', 'EV', 'BICYCLE', name='vehicletype', create_type=False), nullable=False, server_default='ICE'),
        sa.Column('vehicle_capacity_kg', sa.Float(), nullable=False, server_default='100.0'),
        sa.Column('license_number', sa.String(50), nullable=True),
        sa.Column('ev_charging_pref', postgresql.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    
    # Create packages table
    op.create_table(
        'packages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('external_id', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('weight_kg', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('fragility_level', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('address', sa.Text(), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('priority', postgresql.ENUM('NORMAL', 'HIGH', 'EXPRESS', name='packagepriority', create_type=False), nullable=False, server_default='NORMAL'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    
    # Create routes table
    op.create_table(
        'routes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('date', sa.Date(), nullable=False, index=True),
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('total_weight_kg', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('num_packages', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('num_stops', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('route_difficulty_score', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('estimated_time_minutes', sa.Integer(), nullable=False, server_default='60'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    
    # Create route_packages association table
    op.create_table(
        'route_packages',
        sa.Column('route_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('routes.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('package_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('packages.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('stop_order', sa.Integer(), nullable=False, server_default='0'),
    )
    
    # Create assignments table
    op.create_table(
        'assignments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('date', sa.Date(), nullable=False, index=True),
        sa.Column('driver_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('drivers.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('route_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('routes.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('workload_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('fairness_score', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('allocation_run_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    
    # Create driver_stats_daily table
    op.create_table(
        'driver_stats_daily',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('driver_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('drivers.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('date', sa.Date(), nullable=False, index=True),
        sa.Column('avg_workload_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('total_routes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('gini_contribution', sa.Float(), nullable=True),
        sa.Column('reported_stress_level', sa.Float(), nullable=True),
        sa.Column('reported_fairness_score', sa.Float(), nullable=True),
    )
    
    # Create driver_feedback table
    op.create_table(
        'driver_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('driver_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('drivers.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('assignment_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('assignments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('fairness_rating', sa.Integer(), nullable=False),
        sa.Column('stress_level', sa.Integer(), nullable=False),
        sa.Column('tiredness_level', sa.Integer(), nullable=False),
        sa.Column('hardest_aspect', postgresql.ENUM('traffic', 'parking', 'stairs', 'weather', 'heavy_load', 'customer', 'navigation', 'other', name='hardestaspect', create_type=False), nullable=True),
        sa.Column('comments', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    
    # Create indexes for common queries
    op.create_index('ix_assignments_allocation_run', 'assignments', ['allocation_run_id'])
    op.create_index('ix_driver_stats_driver_date', 'driver_stats_daily', ['driver_id', 'date'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_driver_stats_driver_date', table_name='driver_stats_daily')
    op.drop_index('ix_assignments_allocation_run', table_name='assignments')
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('driver_feedback')
    op.drop_table('driver_stats_daily')
    op.drop_table('assignments')
    op.drop_table('route_packages')
    op.drop_table('routes')
    op.drop_table('packages')
    op.drop_table('drivers')
    
    # Drop enum types
    op.execute("DROP TYPE IF EXISTS hardestaspect")
    op.execute("DROP TYPE IF EXISTS packagepriority")
    op.execute("DROP TYPE IF EXISTS vehicletype")
    op.execute("DROP TYPE IF EXISTS preferredlanguage")
