"""Phase 8: Learning Agent tables and DriverStatsDaily extensions.

Revision ID: 006_phase8_learning
Revises: 005_phase7_ev_recovery
Create Date: 2026-02-04

Creates:
- learning_episodes table for bandit learning
- driver_effort_models table for per-driver XGBoost models
- Extends driver_stats_daily with learning fields
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '006_phase8_learning'
down_revision = '005_phase7_ev_recovery'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create learning_episodes table
    op.create_table(
        'learning_episodes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('allocation_run_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('allocation_runs.id', ondelete='CASCADE'),
                  nullable=False, unique=True),
        sa.Column('config_hash', sa.String(64), nullable=False),
        sa.Column('fairness_config', postgresql.JSONB, nullable=False),
        sa.Column('arm_idx', sa.Integer, nullable=False, default=0),
        sa.Column('num_drivers', sa.Integer, nullable=False, default=0),
        sa.Column('num_routes', sa.Integer, nullable=False, default=0),
        sa.Column('episode_reward', sa.Float, nullable=True),
        sa.Column('reward_computed_at', sa.DateTime, nullable=True),
        sa.Column('alpha_prior', sa.Float, default=1.0),
        sa.Column('beta_prior', sa.Float, default=1.0),
        sa.Column('samples_count', sa.Integer, default=0),
        sa.Column('is_experimental', sa.Boolean, default=False),
        sa.Column('avg_fairness_rating', sa.Float, nullable=True),
        sa.Column('avg_stress_level', sa.Float, nullable=True),
        sa.Column('completion_rate', sa.Float, nullable=True),
        sa.Column('feedback_count', sa.Integer, default=0),
        sa.Column('created_at', sa.DateTime, nullable=False, 
                  server_default=sa.func.now()),
    )
    
    # Create indexes for learning_episodes
    op.create_index('ix_learning_episodes_config_hash', 'learning_episodes', ['config_hash'])
    op.create_index('ix_learning_episodes_created_at', 'learning_episodes', ['created_at'])
    op.create_index('ix_learning_episodes_is_experimental', 'learning_episodes', ['is_experimental'])
    op.create_index('ix_learning_episodes_allocation_run_id', 'learning_episodes', ['allocation_run_id'])
    
    # Create driver_effort_models table
    op.create_table(
        'driver_effort_models',
        sa.Column('driver_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('drivers.id', ondelete='CASCADE'),
                  primary_key=True),
        sa.Column('model_version', sa.Integer, nullable=False, default=1),
        sa.Column('model_pickle', sa.LargeBinary, nullable=True),
        sa.Column('training_samples', sa.Integer, default=0),
        sa.Column('feature_names', postgresql.JSONB, nullable=True),
        sa.Column('mse_history', postgresql.JSONB, nullable=True),
        sa.Column('current_mse', sa.Float, nullable=True),
        sa.Column('r2_score', sa.Float, nullable=True),
        sa.Column('active', sa.Boolean, default=True),
        sa.Column('last_trained_at', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False,
                  server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Create index for driver_effort_models
    op.create_index('ix_driver_effort_models_active', 'driver_effort_models', ['active'])
    
    # Extend driver_stats_daily with learning fields
    op.add_column('driver_stats_daily',
                  sa.Column('predicted_effort', sa.Float, nullable=True))
    op.add_column('driver_stats_daily',
                  sa.Column('actual_effort', sa.Float, nullable=True))
    op.add_column('driver_stats_daily',
                  sa.Column('prediction_error', sa.Float, nullable=True))
    op.add_column('driver_stats_daily',
                  sa.Column('model_version_used', sa.Integer, nullable=True))
    op.add_column('driver_stats_daily',
                  sa.Column('allocation_run_id', postgresql.UUID(as_uuid=True),
                           sa.ForeignKey('allocation_runs.id', ondelete='SET NULL'),
                           nullable=True))
    
    # Create index for allocation_run_id
    op.create_index('ix_driver_stats_daily_allocation_run_id', 
                    'driver_stats_daily', ['allocation_run_id'])


def downgrade() -> None:
    # Remove indexes
    op.drop_index('ix_driver_stats_daily_allocation_run_id', 'driver_stats_daily')
    op.drop_index('ix_driver_effort_models_active', 'driver_effort_models')
    op.drop_index('ix_learning_episodes_allocation_run_id', 'learning_episodes')
    op.drop_index('ix_learning_episodes_is_experimental', 'learning_episodes')
    op.drop_index('ix_learning_episodes_created_at', 'learning_episodes')
    op.drop_index('ix_learning_episodes_config_hash', 'learning_episodes')
    
    # Remove columns from driver_stats_daily
    op.drop_column('driver_stats_daily', 'allocation_run_id')
    op.drop_column('driver_stats_daily', 'model_version_used')
    op.drop_column('driver_stats_daily', 'prediction_error')
    op.drop_column('driver_stats_daily', 'actual_effort')
    op.drop_column('driver_stats_daily', 'predicted_effort')
    
    # Drop tables
    op.drop_table('driver_effort_models')
    op.drop_table('learning_episodes')
