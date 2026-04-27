"""
Learning Agent service for Phase 8.
Implements Multi-Armed Bandit (Thompson Sampling) for fairness weight tuning
and per-driver XGBoost models for personalized effort prediction.
"""

import hashlib
import itertools
import os
import tempfile
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.learning_episode import LearningEpisode
from app.models.driver_effort_model import DriverEffortModel
from app.models.driver import DriverStatsDaily, DriverFeedback
from app.models.assignment import Assignment
from app.models.allocation_run import AllocationRun
from app.models.fairness_config import FairnessConfig


def hash_config(config: dict) -> str:
    """Generate SHA256 hash of a FairnessConfig dict."""
    # Sort keys for consistent hashing
    config_str = str(sorted(config.items()))
    return hashlib.sha256(config_str.encode()).hexdigest()[:64]


class FairnessBandit:
    """
    Multi-Armed Bandit using Thompson Sampling for fairness weight tuning.
    
    Each arm represents a different FairnessConfig. The bandit learns
    which configurations lead to higher driver satisfaction.
    """
    
    # Define the arm space (discretized config values)
    GINI_OPTIONS = [0.28, 0.33, 0.38]
    STDDEV_OPTIONS = [20.0, 25.0, 30.0]
    RECOVERY_OPTIONS = [0.6, 0.7, 0.8]
    EV_PENALTY_OPTIONS = [0.2, 0.3, 0.4]
    
    def __init__(self, db: AsyncSession):
        """Initialize bandit with database session."""
        self.db = db
        self.arms = self._generate_arm_space()
        self.arm_to_idx = {hash_config(arm): idx for idx, arm in enumerate(self.arms)}
        self.num_arms = len(self.arms)
        
        # Initialize priors (will be loaded from DB)
        self.alphas = np.ones(self.num_arms)
        self.betas = np.ones(self.num_arms)
        self.samples = np.zeros(self.num_arms, dtype=int)
    
    def _generate_arm_space(self) -> List[dict]:
        """Generate all possible arm configurations (discretized FairnessConfigs)."""
        arms = []
        for gini, stddev, recovery, ev_penalty in itertools.product(
            self.GINI_OPTIONS,
            self.STDDEV_OPTIONS,
            self.RECOVERY_OPTIONS,
            self.EV_PENALTY_OPTIONS,
        ):
            arms.append({
                "gini_threshold": gini,
                "stddev_threshold": stddev,
                "recovery_lightening_factor": recovery,
                "ev_charging_penalty_weight": ev_penalty,
                # Fixed defaults for other params
                "max_gap_threshold": 25.0,
                "workload_weight_packages": 1.0,
                "workload_weight_weight_kg": 0.5,
                "workload_weight_difficulty": 10.0,
                "workload_weight_time": 0.2,
                "recovery_mode_enabled": True,
                "complexity_debt_hard_threshold": 2.0,
                "recovery_penalty_weight": 3.0,
                "ev_safety_margin_pct": 10.0,
            })
        return arms
    
    async def load_priors(self) -> None:
        """Load alpha/beta priors from database based on historical episodes."""
        # Get recent episodes (last 30 days)
        cutoff = datetime.utcnow() - timedelta(days=30)
        result = await self.db.execute(
            select(LearningEpisode)
            .where(LearningEpisode.created_at >= cutoff)
            .where(LearningEpisode.episode_reward.isnot(None))
        )
        episodes = result.scalars().all()
        
        # Aggregate rewards per arm
        for episode in episodes:
            config_hash = episode.config_hash
            if config_hash in self.arm_to_idx:
                arm_idx = self.arm_to_idx[config_hash]
                reward = episode.episode_reward
                # Update priors: alpha += reward, beta += (1 - reward)
                self.alphas[arm_idx] += reward
                self.betas[arm_idx] += (1 - reward)
                self.samples[arm_idx] += 1
    
    async def select_arm(self, experimental: bool = False) -> Tuple[dict, int, float, float]:
        """
        Select an arm using Thompson Sampling.
        
        Args:
            experimental: If True, may select a less-explored arm for A/B testing.
            
        Returns:
            Tuple of (config dict, arm_idx, alpha, beta)
        """
        await self.load_priors()
        
        if experimental:
            # For experimental cohort, boost exploration of under-sampled arms
            exploration_bonus = np.log(np.sum(self.samples) + 1) / (self.samples + 1)
            exploration_bonus = exploration_bonus / np.max(exploration_bonus + 0.001)
        else:
            exploration_bonus = np.zeros(self.num_arms)
        
        # Thompson Sampling: sample from Beta(alpha, beta) for each arm
        scores = []
        for arm_idx in range(self.num_arms):
            theta = np.random.beta(self.alphas[arm_idx], self.betas[arm_idx])
            theta += exploration_bonus[arm_idx] * 0.1  # Small exploration boost
            scores.append(theta)
        
        best_arm_idx = int(np.argmax(scores))
        return (
            self.arms[best_arm_idx],
            best_arm_idx,
            float(self.alphas[best_arm_idx]),
            float(self.betas[best_arm_idx]),
        )
    
    async def update(self, config_hash: str, reward: float) -> bool:
        """
        Update posteriors with new episode reward.
        
        Args:
            config_hash: Hash of the config used
            reward: Normalized reward in [0, 1]
            
        Returns:
            True if update successful, False if config not found
        """
        if config_hash not in self.arm_to_idx:
            return False
        
        arm_idx = self.arm_to_idx[config_hash]
        # Clamp reward to [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        # Update priors
        self.alphas[arm_idx] += reward
        self.betas[arm_idx] += (1 - reward)
        self.samples[arm_idx] += 1
        
        return True
    
    def get_arm_statistics(self) -> List[dict]:
        """Get statistics for all arms."""
        stats = []
        for arm_idx, arm_config in enumerate(self.arms):
            config_hash = hash_config(arm_config)
            mean = self.alphas[arm_idx] / (self.alphas[arm_idx] + self.betas[arm_idx])
            stats.append({
                "arm_idx": arm_idx,
                "config_hash": config_hash,
                "config": arm_config,
                "alpha": float(self.alphas[arm_idx]),
                "beta": float(self.betas[arm_idx]),
                "samples": int(self.samples[arm_idx]),
                "mean_reward": float(mean),
            })
        return sorted(stats, key=lambda x: x["mean_reward"], reverse=True)
    
    def get_top_configs(self, n: int = 5) -> List[dict]:
        """Get top N performing configurations."""
        stats = self.get_arm_statistics()
        return stats[:n]


class RewardComputer:
    """
    Computes episode rewards from driver feedback.
    
    Reward formula:
    reward = 0.4 * avg_fairness_rating + 0.3 * (1 - avg_stress_level/10) + 
             0.2 * completion_rate + 0.1 * retention_score
    """
    
    FAIRNESS_WEIGHT = 0.4
    STRESS_WEIGHT = 0.3
    COMPLETION_WEIGHT = 0.2
    RETENTION_WEIGHT = 0.1
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def compute_episode_reward(
        self,
        allocation_run_id: uuid.UUID,
    ) -> Tuple[float, dict]:
        """
        Compute reward for an allocation run based on driver feedback.
        
        Args:
            allocation_run_id: ID of the allocation run
            
        Returns:
            Tuple of (reward, feedback_stats)
        """
        # Get all assignments for this run
        assignments_result = await self.db.execute(
            select(Assignment)
            .where(Assignment.allocation_run_id == allocation_run_id)
        )
        assignments = assignments_result.scalars().all()
        
        if not assignments:
            return 0.5, {"error": "no_assignments"}
        
        assignment_ids = [a.id for a in assignments]
        
        # Get feedback for these assignments
        feedback_result = await self.db.execute(
            select(DriverFeedback)
            .where(DriverFeedback.assignment_id.in_(assignment_ids))
        )
        feedbacks = feedback_result.scalars().all()
        
        # Calculate metrics
        if not feedbacks:
            # No feedback yet, return neutral reward
            return 0.5, {
                "feedback_count": 0,
                "avg_fairness_rating": None,
                "avg_stress_level": None,
                "completion_rate": None,
            }
        
        # Average fairness rating (1-5 scale, normalize to 0-1)
        fairness_ratings = [f.fairness_rating for f in feedbacks if f.fairness_rating]
        avg_fairness = (np.mean(fairness_ratings) - 1) / 4 if fairness_ratings else 0.5
        
        # Average stress level (1-10 scale, normalize and invert)
        stress_levels = [f.stress_level for f in feedbacks if f.stress_level]
        avg_stress = np.mean(stress_levels) / 10 if stress_levels else 0.5
        stress_component = 1 - avg_stress
        
        # Completion rate (based on "would_take_similar_route_again")
        would_take = [f.would_take_similar_route_again for f in feedbacks 
                      if f.would_take_similar_route_again is not None]
        completion_rate = np.mean(would_take) if would_take else 0.5
        
        # Retention score (based on tiredness, inverse relationship)
        tiredness_levels = [f.tiredness_level for f in feedbacks if f.tiredness_level]
        avg_tiredness = np.mean(tiredness_levels) / 5 if tiredness_levels else 0.5
        retention_score = 1 - avg_tiredness
        
        # Compute final reward
        reward = (
            self.FAIRNESS_WEIGHT * avg_fairness +
            self.STRESS_WEIGHT * stress_component +
            self.COMPLETION_WEIGHT * completion_rate +
            self.RETENTION_WEIGHT * retention_score
        )
        
        # Clamp to [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        feedback_stats = {
            "feedback_count": len(feedbacks),
            "avg_fairness_rating": float(np.mean(fairness_ratings)) if fairness_ratings else None,
            "avg_stress_level": float(np.mean(stress_levels)) if stress_levels else None,
            "completion_rate": float(completion_rate),
            "retention_score": float(retention_score),
        }
        
        return reward, feedback_stats


class DriverEffortLearner:
    """
    Per-driver XGBoost models for personalized effort prediction.
    
    Each driver gets their own model trained on their historical
    assignment data. Models are retrained periodically (daily cron).
    """
    
    MIN_TRAINING_SAMPLES = 10
    MAX_TRAINING_SAMPLES = 100
    FEATURE_NAMES = [
        "num_packages",
        "total_weight_kg",
        "num_stops",
        "route_difficulty_score",
        "estimated_time_minutes",
        "experience_days",
        "recent_avg_workload",
        "recent_hard_days",
    ]
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._xgb_available = self._check_xgboost()
    
    def _check_xgboost(self) -> bool:
        """Check if XGBoost is available."""
        try:
            import xgboost
            return True
        except ImportError:
            return False
    
    async def load_model(self, driver_id: uuid.UUID) -> Optional[Any]:
        """Load a driver's effort model from the database (safe JSON format)."""
        result = await self.db.execute(
            select(DriverEffortModel)
            .where(DriverEffortModel.driver_id == driver_id)
            .where(DriverEffortModel.active == True)
        )
        model_record = result.scalar_one_or_none()
        
        if not model_record or not model_record.model_path:
            return None
        
        try:
            import xgboost as xgb
            bst = xgb.Booster()
            bst.load_model(model_record.model_path)
            return bst
        except Exception:
            return None
    
    async def get_model_version(self, driver_id: uuid.UUID) -> Optional[int]:
        """Get current model version for a driver."""
        result = await self.db.execute(
            select(DriverEffortModel.model_version)
            .where(DriverEffortModel.driver_id == driver_id)
            .where(DriverEffortModel.active == True)
        )
        version = result.scalar_one_or_none()
        return version
    
    async def predict_effort(
        self,
        driver_id: uuid.UUID,
        route_features: dict,
    ) -> Tuple[Optional[float], Optional[int]]:
        """
        Predict effort for a driver-route pair.
        
        Args:
            driver_id: Driver's UUID
            route_features: Dict with route features
            
        Returns:
            Tuple of (predicted_effort, model_version) or (None, None)
        """
        if not self._xgb_available:
            return None, None
        
        model = await self.load_model(driver_id)
        if model is None:
            return None, None
        
        version = await self.get_model_version(driver_id)
        
        try:
            import pandas as pd
            
            # Build feature vector
            features = {name: route_features.get(name, 0.0) for name in self.FEATURE_NAMES}
            X = pd.DataFrame([features])
            
            prediction = float(model.predict(X)[0])
            return prediction, version
        except Exception:
            return None, version
    
    async def update_model(self, driver_id: uuid.UUID) -> dict:
        """
        Retrain XGBoost model for a driver using their history.
        
        Returns:
            Dict with training status and metrics
        """
        if not self._xgb_available:
            return {"status": "skipped", "reason": "xgboost_not_available"}
        
        import xgboost as xgb
        import pandas as pd
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Get driver's historical stats with actual effort
        result = await self.db.execute(
            select(DriverStatsDaily)
            .where(DriverStatsDaily.driver_id == driver_id)
            .where(DriverStatsDaily.actual_effort.isnot(None))
            .order_by(DriverStatsDaily.date.desc())
            .limit(self.MAX_TRAINING_SAMPLES)
        )
        stats = result.scalars().all()
        
        if len(stats) < self.MIN_TRAINING_SAMPLES:
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "samples": len(stats),
                "required": self.MIN_TRAINING_SAMPLES,
            }
        
        # Build training data
        X_data = []
        y_data = []
        
        for stat in stats:
            # Get the assignment for this stat
            assignment_result = await self.db.execute(
                select(Assignment)
                .where(Assignment.driver_id == driver_id)
                .where(Assignment.date == stat.date)
                .limit(1)
            )
            assignment = assignment_result.scalar_one_or_none()
            
            if assignment and assignment.route:
                route = assignment.route
                features = {
                    "num_packages": route.num_packages,
                    "total_weight_kg": route.total_weight_kg,
                    "num_stops": route.num_stops,
                    "route_difficulty_score": route.route_difficulty_score,
                    "estimated_time_minutes": route.estimated_time_minutes,
                    "experience_days": (stat.date - stat.driver.created_at.date()).days if stat.driver else 0,
                    "recent_avg_workload": stat.avg_workload_score,
                    "recent_hard_days": 1 if stat.is_hard_day else 0,
                }
                X_data.append(features)
                y_data.append(stat.actual_effort)
        
        if len(X_data) < self.MIN_TRAINING_SAMPLES:
            return {
                "status": "skipped",
                "reason": "insufficient_route_data",
                "samples": len(X_data),
            }
        
        # Convert to DataFrame
        X = pd.DataFrame(X_data)
        y = np.array(y_data)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)
        
        # Compute metrics
        y_pred = model.predict(X)
        mse = float(mean_squared_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))
        
        # Save model using safe JSON format (NO PICKLE)
        from app.config import get_settings
        settings = get_settings()
        model_dir = settings.model_storage_dir
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"driver_{driver_id}_v{timestamp}.json"
        model_path = os.path.join(model_dir, model_filename)
        model.save_model(model_path)
        
        # Compute checksum for integrity
        import hashlib as _hashlib
        sha256 = _hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        model_checksum = sha256.hexdigest()
        
        # Check if model record exists
        existing_result = await self.db.execute(
            select(DriverEffortModel)
            .where(DriverEffortModel.driver_id == driver_id)
        )
        existing = existing_result.scalar_one_or_none()
        
        if existing:
            # Update existing record
            existing.model_version += 1
            existing.model_path = model_path
            existing.model_checksum = model_checksum
            existing.training_samples = len(X_data)
            existing.feature_names = {"names": self.FEATURE_NAMES}
            existing.current_mse = mse
            existing.r2_score = r2
            existing.last_trained_at = datetime.utcnow()
            
            # Update MSE history
            mse_history = existing.mse_history or []
            if isinstance(mse_history, dict):
                mse_history = mse_history.get("values", [])
            mse_history.append(mse)
            mse_history = mse_history[-10:]  # Keep last 10
            existing.mse_history = {"values": mse_history}
            
            version = existing.model_version
        else:
            # Create new record
            new_model = DriverEffortModel(
                driver_id=driver_id,
                model_version=1,
                model_path=model_path,
                model_checksum=model_checksum,
                training_samples=len(X_data),
                feature_names={"names": self.FEATURE_NAMES},
                mse_history={"values": [mse]},
                current_mse=mse,
                r2_score=r2,
                active=True,
                last_trained_at=datetime.utcnow(),
            )
            self.db.add(new_model)
            version = 1
        
        return {
            "status": "success",
            "driver_id": str(driver_id),
            "model_version": version,
            "training_samples": len(X_data),
            "mse": mse,
            "r2_score": r2,
        }
    
    async def get_model_status(self, driver_id: uuid.UUID) -> Optional[dict]:
        """Get status of a driver's effort model."""
        result = await self.db.execute(
            select(DriverEffortModel)
            .where(DriverEffortModel.driver_id == driver_id)
        )
        model = result.scalar_one_or_none()
        
        if not model:
            return None
        
        return {
            "driver_id": str(driver_id),
            "model_version": model.model_version,
            "training_samples": model.training_samples,
            "current_mse": model.current_mse,
            "r2_score": model.r2_score,
            "mse_history": model.mse_history,
            "active": model.active,
            "last_trained_at": model.last_trained_at.isoformat() if model.last_trained_at else None,
        }


class LearningAgent:
    """
    Main Learning Agent that orchestrates bandit and per-driver models.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.bandit = FairnessBandit(db)
        self.reward_computer = RewardComputer(db)
        self.effort_learner = DriverEffortLearner(db)
    
    async def create_episode(
        self,
        allocation_run_id: uuid.UUID,
        fairness_config: dict,
        num_drivers: int,
        num_routes: int,
        is_experimental: bool = False,
    ) -> LearningEpisode:
        """Create a learning episode for an allocation run."""
        config_hash = hash_config(fairness_config)
        arm_idx = self.bandit.arm_to_idx.get(config_hash, -1)
        
        # Get current priors
        await self.bandit.load_priors()
        alpha = float(self.bandit.alphas[arm_idx]) if arm_idx >= 0 else 1.0
        beta = float(self.bandit.betas[arm_idx]) if arm_idx >= 0 else 1.0
        samples = int(self.bandit.samples[arm_idx]) if arm_idx >= 0 else 0
        
        episode = LearningEpisode(
            allocation_run_id=allocation_run_id,
            config_hash=config_hash,
            fairness_config=fairness_config,
            arm_idx=arm_idx,
            num_drivers=num_drivers,
            num_routes=num_routes,
            alpha_prior=alpha,
            beta_prior=beta,
            samples_count=samples,
            is_experimental=is_experimental,
        )
        self.db.add(episode)
        
        return episode
    
    async def process_episode_reward(self, episode_id: uuid.UUID) -> dict:
        """Process and update reward for an episode."""
        # Get episode
        result = await self.db.execute(
            select(LearningEpisode)
            .where(LearningEpisode.id == episode_id)
        )
        episode = result.scalar_one_or_none()
        
        if not episode:
            return {"status": "error", "reason": "episode_not_found"}
        
        if episode.episode_reward is not None:
            return {"status": "skipped", "reason": "already_computed"}
        
        # Compute reward
        reward, stats = await self.reward_computer.compute_episode_reward(
            episode.allocation_run_id
        )
        
        # Update episode
        episode.episode_reward = reward
        episode.reward_computed_at = datetime.utcnow()
        episode.avg_fairness_rating = stats.get("avg_fairness_rating")
        episode.avg_stress_level = stats.get("avg_stress_level")
        episode.completion_rate = stats.get("completion_rate")
        episode.feedback_count = stats.get("feedback_count", 0)
        
        # Update bandit
        await self.bandit.update(episode.config_hash, reward)
        
        return {
            "status": "success",
            "episode_id": str(episode_id),
            "reward": reward,
            "feedback_stats": stats,
        }
    
    async def select_config(self, experimental: bool = False) -> dict:
        """Select a FairnessConfig using the bandit."""
        config, arm_idx, alpha, beta = await self.bandit.select_arm(experimental)
        return {
            "config": config,
            "arm_idx": arm_idx,
            "alpha": alpha,
            "beta": beta,
            "config_hash": hash_config(config),
        }
    
    async def get_learning_status(self) -> dict:
        """Get overall learning status for admin API."""
        await self.bandit.load_priors()
        
        # Get top configs
        top_configs = self.bandit.get_top_configs(5)
        
        # Count active driver models
        result = await self.db.execute(
            select(func.count(DriverEffortModel.driver_id))
            .where(DriverEffortModel.active == True)
        )
        active_models = result.scalar() or 0
        
        # Get average MSE
        result = await self.db.execute(
            select(func.avg(DriverEffortModel.current_mse))
            .where(DriverEffortModel.active == True)
            .where(DriverEffortModel.current_mse.isnot(None))
        )
        avg_mse = result.scalar() or 0.0
        
        # Get recent episode count
        cutoff = datetime.utcnow() - timedelta(days=7)
        result = await self.db.execute(
            select(func.count(LearningEpisode.id))
            .where(LearningEpisode.created_at >= cutoff)
        )
        recent_episodes = result.scalar() or 0
        
        # Get current active config
        result = await self.db.execute(
            select(FairnessConfig)
            .where(FairnessConfig.is_active == True)
            .limit(1)
        )
        current_config_model = result.scalar_one_or_none()
        
        current_config = None
        if current_config_model:
            current_config = {
                "gini_threshold": current_config_model.gini_threshold,
                "stddev_threshold": current_config_model.stddev_threshold,
                "recovery_lightening_factor": current_config_model.recovery_lightening_factor,
                "ev_charging_penalty_weight": current_config_model.ev_charging_penalty_weight,
            }
        
        return {
            "current_config": current_config,
            "top_performing_configs": top_configs,
            "driver_models_active": active_models,
            "avg_prediction_mse": float(avg_mse),
            "recent_episodes_7d": recent_episodes,
            "total_arms": self.bandit.num_arms,
            "bandit_statistics": {
                "total_samples": int(np.sum(self.bandit.samples)),
                "explored_arms": int(np.sum(self.bandit.samples > 0)),
            }
        }
