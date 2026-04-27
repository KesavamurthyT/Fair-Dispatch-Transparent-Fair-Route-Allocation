#!/usr/bin/env python3
"""
Daily Learning Pipeline for Phase 8.

Run daily at 6 AM (recommended) via cron:
    0 6 * * * cd /path/to/fair-dispatch && python -m cron.daily_learning

This script:
1. Processes completed allocation runs from yesterday
2. Computes episode rewards from driver feedback
3. Updates bandit posteriors
4. Selects new fairness config for today
5. Retrains per-driver XGBoost models
6. Logs metrics for monitoring
"""

import asyncio
import logging
import sys
from datetime import datetime, date, timedelta
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/cron", 1)[0])

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_maker
from app.models import (
    AllocationRun,
    AllocationRunStatus,
    LearningEpisode,
    FairnessConfig,
    Driver,
    DriverEffortModel,
)
from app.services.learning_agent import (
    LearningAgent,
    FairnessBandit,
    hash_config,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("daily_learning")


class DailyLearningPipeline:
    """
    Daily learning pipeline that processes feedback, updates models,
    and deploys new configurations.
    """
    
    # Safety rails
    MIN_EPISODES_FOR_BANDIT = 10
    MIN_FEEDBACK_RATE = 0.3  # At least 30% of drivers must give feedback
    EXPERIMENTAL_COHORT_PCT = 0.10  # 10% get experimental config
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.learning_agent = LearningAgent(db)
        self.metrics = {
            "episodes_processed": 0,
            "rewards_computed": 0,
            "models_updated": 0,
            "errors": [],
        }
    
    async def run(self) -> dict:
        """Run the full daily learning pipeline."""
        logger.info("Starting daily learning pipeline...")
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Process yesterday's episodes
            await self._process_pending_episodes()
            
            # Step 2: Select today's config
            await self._select_todays_config()
            
            # Step 3: Update per-driver models
            await self._update_driver_models()
            
            # Step 4: Log metrics
            self.metrics["duration_seconds"] = (datetime.utcnow() - start_time).total_seconds()
            self.metrics["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Daily learning completed: {self.metrics}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Daily learning failed: {e}")
            self.metrics["error"] = str(e)
            self.metrics["status"] = "failed"
            raise
    
    async def _process_pending_episodes(self) -> None:
        """Process all pending episodes with sufficient feedback time (24h+)."""
        logger.info("Processing pending episodes...")
        
        # Find episodes created more than 24h ago without computed reward
        cutoff = datetime.utcnow() - timedelta(hours=24)
        
        result = await self.db.execute(
            select(LearningEpisode)
            .where(LearningEpisode.created_at <= cutoff)
            .where(LearningEpisode.episode_reward.is_(None))
            .order_by(LearningEpisode.created_at.asc())
            .limit(100)  # Process in batches
        )
        pending_episodes = result.scalars().all()
        
        logger.info(f"Found {len(pending_episodes)} pending episodes")
        
        for episode in pending_episodes:
            try:
                result = await self.learning_agent.process_episode_reward(episode.id)
                if result["status"] == "success":
                    self.metrics["rewards_computed"] += 1
                    logger.debug(f"Computed reward for episode {episode.id}: {result['reward']:.3f}")
                self.metrics["episodes_processed"] += 1
            except Exception as e:
                logger.warning(f"Failed to process episode {episode.id}: {e}")
                self.metrics["errors"].append(f"episode_{episode.id}: {str(e)}")
        
        await self.db.commit()
    
    async def _select_todays_config(self) -> None:
        """Select and deploy today's fairness configuration."""
        logger.info("Selecting today's configuration...")
        
        # Check if we have enough data for bandit selection
        await self.learning_agent.bandit.load_priors()
        total_samples = int(sum(self.learning_agent.bandit.samples))
        
        if total_samples < self.MIN_EPISODES_FOR_BANDIT:
            logger.info(f"Insufficient bandit data ({total_samples}/{self.MIN_EPISODES_FOR_BANDIT}). Using default config.")
            self.metrics["config_selection"] = "default_insufficient_data"
            return
        
        # Select stable config (non-experimental)
        stable_selection = await self.learning_agent.select_config(experimental=False)
        stable_config = stable_selection["config"]
        
        logger.info(f"Selected stable config (arm {stable_selection['arm_idx']}): "
                   f"gini={stable_config['gini_threshold']}, "
                   f"stddev={stable_config['stddev_threshold']}")
        
        # Update active FairnessConfig in database
        await self._update_active_config(stable_config)
        
        self.metrics["config_selection"] = "bandit_selected"
        self.metrics["selected_arm"] = stable_selection["arm_idx"]
        self.metrics["selected_config_hash"] = stable_selection["config_hash"]
    
    async def _update_active_config(self, config: dict) -> None:
        """Update the active FairnessConfig in database."""
        # Deactivate all existing configs
        result = await self.db.execute(
            select(FairnessConfig).where(FairnessConfig.is_active == True)
        )
        active_configs = result.scalars().all()
        
        for existing in active_configs:
            existing.is_active = False
        
        # Create or update config with bandit-selected values
        new_config = FairnessConfig(
            is_active=True,
            gini_threshold=config["gini_threshold"],
            stddev_threshold=config["stddev_threshold"],
            recovery_lightening_factor=config["recovery_lightening_factor"],
            ev_charging_penalty_weight=config["ev_charging_penalty_weight"],
            max_gap_threshold=config.get("max_gap_threshold", 25.0),
            workload_weight_packages=config.get("workload_weight_packages", 1.0),
            workload_weight_weight_kg=config.get("workload_weight_weight_kg", 0.5),
            workload_weight_difficulty=config.get("workload_weight_difficulty", 10.0),
            workload_weight_time=config.get("workload_weight_time", 0.2),
            recovery_mode_enabled=config.get("recovery_mode_enabled", True),
            complexity_debt_hard_threshold=config.get("complexity_debt_hard_threshold", 2.0),
            recovery_penalty_weight=config.get("recovery_penalty_weight", 3.0),
            ev_safety_margin_pct=config.get("ev_safety_margin_pct", 10.0),
        )
        self.db.add(new_config)
        
        await self.db.commit()
        logger.info("Updated active FairnessConfig")
    
    async def _update_driver_models(self) -> None:
        """Update per-driver XGBoost models."""
        logger.info("Updating per-driver effort models...")
        
        # Get all active drivers
        result = await self.db.execute(
            select(Driver.id)
            .order_by(Driver.created_at.asc())
        )
        driver_ids = [row[0] for row in result.fetchall()]
        
        logger.info(f"Found {len(driver_ids)} drivers to update")
        
        successful = 0
        skipped = 0
        failed = 0
        
        for driver_id in driver_ids:
            try:
                result = await self.learning_agent.effort_learner.update_model(driver_id)
                
                if result["status"] == "success":
                    successful += 1
                    logger.debug(f"Updated model for driver {driver_id}: MSE={result['mse']:.3f}")
                elif result["status"] == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                logger.warning(f"Failed to update model for driver {driver_id}: {e}")
                self.metrics["errors"].append(f"driver_model_{driver_id}: {str(e)}")
        
        await self.db.commit()
        
        self.metrics["models_updated"] = successful
        self.metrics["models_skipped"] = skipped
        self.metrics["models_failed"] = failed
        
        logger.info(f"Model updates: {successful} success, {skipped} skipped, {failed} failed")


async def run_daily_learning() -> dict:
    """
    Main entry point for the daily learning cron job.
    
    Returns:
        Dict with execution metrics
    """
    async with async_session_maker() as db:
        pipeline = DailyLearningPipeline(db)
        return await pipeline.run()


def main():
    """CLI entry point."""
    logger.info("=" * 60)
    logger.info("DAILY LEARNING PIPELINE - " + datetime.utcnow().isoformat())
    logger.info("=" * 60)
    
    try:
        # Run the async pipeline
        metrics = asyncio.run(run_daily_learning())
        
        # Print summary
        print("\n" + "=" * 40)
        print("PIPELINE SUMMARY")
        print("=" * 40)
        print(f"Episodes Processed: {metrics.get('episodes_processed', 0)}")
        print(f"Rewards Computed:   {metrics.get('rewards_computed', 0)}")
        print(f"Models Updated:     {metrics.get('models_updated', 0)}")
        print(f"Config Selection:   {metrics.get('config_selection', 'N/A')}")
        print(f"Duration:           {metrics.get('duration_seconds', 0):.2f}s")
        
        if metrics.get("errors"):
            print(f"\nErrors: {len(metrics['errors'])}")
            for error in metrics["errors"][:5]:
                print(f"  - {error}")
        
        print("=" * 40)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
