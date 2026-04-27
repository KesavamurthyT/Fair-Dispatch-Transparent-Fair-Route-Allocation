"""
Phase 7: EV-aware routing and Recovery Mode

Revision ID: 005_phase7_ev_recovery
Revises: 004_add_explanation_fields
Create Date: 2026-02-04
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '005_phase7_ev_recovery'
down_revision = '004_add_explanation_fields'
branch_labels = None
depends_on = None


def upgrade():
    # Driver EV fields
    op.add_column('drivers', sa.Column('battery_range_km', sa.Float(), nullable=True))
    op.add_column('drivers', sa.Column('charging_time_minutes', sa.Integer(), nullable=True))
    
    # Route distance field
    op.add_column('routes', sa.Column('total_distance_km', sa.Float(), nullable=True))
    
    # DriverStatsDaily recovery fields
    op.add_column('driver_stats_daily', sa.Column('is_hard_day', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('driver_stats_daily', sa.Column('complexity_debt', sa.Float(), nullable=False, server_default='0.0'))
    op.add_column('driver_stats_daily', sa.Column('is_recovery_day', sa.Boolean(), nullable=False, server_default='false'))
    
    # FairnessConfig recovery/EV fields
    op.add_column('fairness_configs', sa.Column('complexity_debt_hard_threshold', sa.Float(), nullable=False, server_default='2.0'))
    op.add_column('fairness_configs', sa.Column('recovery_lightening_factor', sa.Float(), nullable=False, server_default='0.7'))
    op.add_column('fairness_configs', sa.Column('recovery_penalty_weight', sa.Float(), nullable=False, server_default='3.0'))
    op.add_column('fairness_configs', sa.Column('ev_charging_penalty_weight', sa.Float(), nullable=False, server_default='0.3'))
    op.add_column('fairness_configs', sa.Column('ev_safety_margin_pct', sa.Float(), nullable=False, server_default='10.0'))


def downgrade():
    # FairnessConfig
    op.drop_column('fairness_configs', 'ev_safety_margin_pct')
    op.drop_column('fairness_configs', 'ev_charging_penalty_weight')
    op.drop_column('fairness_configs', 'recovery_penalty_weight')
    op.drop_column('fairness_configs', 'recovery_lightening_factor')
    op.drop_column('fairness_configs', 'complexity_debt_hard_threshold')
    
    # DriverStatsDaily
    op.drop_column('driver_stats_daily', 'is_recovery_day')
    op.drop_column('driver_stats_daily', 'complexity_debt')
    op.drop_column('driver_stats_daily', 'is_hard_day')
    
    # Route
    op.drop_column('routes', 'total_distance_km')
    
    # Driver
    op.drop_column('drivers', 'charging_time_minutes')
    op.drop_column('drivers', 'battery_range_km')
