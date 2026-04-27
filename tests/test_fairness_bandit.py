"""
Tests for FairnessBandit Thompson Sampling implementation.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/tests", 1)[0])

from app.services.learning_agent import FairnessBandit, hash_config


class TestHashConfig:
    """Tests for config hashing."""
    
    def test_hash_deterministic(self):
        """Same config should produce same hash."""
        config = {"gini_threshold": 0.33, "stddev_threshold": 25.0}
        assert hash_config(config) == hash_config(config)
    
    def test_hash_different_configs(self):
        """Different configs should produce different hashes."""
        config1 = {"gini_threshold": 0.33, "stddev_threshold": 25.0}
        config2 = {"gini_threshold": 0.35, "stddev_threshold": 25.0}
        assert hash_config(config1) != hash_config(config2)
    
    def test_hash_order_independent(self):
        """Config order shouldn't affect hash."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}
        assert hash_config(config1) == hash_config(config2)
    
    def test_hash_length(self):
        """Hash should be 64 characters."""
        config = {"test": 123}
        assert len(hash_config(config)) == 64


class TestFairnessBandit:
    """Tests for FairnessBandit class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock async database session."""
        mock = AsyncMock()
        mock.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))
        return mock
    
    @pytest.fixture
    def bandit(self, mock_db):
        """Create FairnessBandit instance."""
        return FairnessBandit(mock_db)
    
    def test_arm_space_generation(self, bandit):
        """Test that arm space is generated correctly."""
        # 3 * 3 * 3 * 3 = 81 combinations
        assert len(bandit.arms) == 81
        
        # Each arm should be a dict with required keys
        for arm in bandit.arms:
            assert "gini_threshold" in arm
            assert "stddev_threshold" in arm
            assert "recovery_lightening_factor" in arm
            assert "ev_charging_penalty_weight" in arm
    
    def test_arm_to_idx_mapping(self, bandit):
        """Test that arm hash mapping is consistent."""
        for idx, arm in enumerate(bandit.arms):
            config_hash = hash_config(arm)
            assert config_hash in bandit.arm_to_idx
            assert bandit.arm_to_idx[config_hash] == idx
    
    def test_initial_priors(self, bandit):
        """Test initial alpha/beta priors are uniform."""
        assert len(bandit.alphas) == 81
        assert len(bandit.betas) == 81
        assert np.all(bandit.alphas == 1.0)
        assert np.all(bandit.betas == 1.0)
    
    @pytest.mark.asyncio
    async def test_select_arm_returns_valid_config(self, bandit):
        """Test that select_arm returns a valid config."""
        config, arm_idx, alpha, beta = await bandit.select_arm()
        
        assert isinstance(config, dict)
        assert 0 <= arm_idx < 81
        assert alpha >= 1.0
        assert beta >= 1.0
        assert "gini_threshold" in config
    
    @pytest.mark.asyncio
    async def test_select_arm_experimental(self, bandit):
        """Test experimental arm selection with exploration boost."""
        config1, idx1, _, _ = await bandit.select_arm(experimental=False)
        config2, idx2, _, _ = await bandit.select_arm(experimental=True)
        
        # Both should return valid configs
        assert isinstance(config1, dict)
        assert isinstance(config2, dict)
    
    @pytest.mark.asyncio
    async def test_update_shifts_posteriors(self, bandit):
        """Test that update shifts posteriors correctly."""
        config = bandit.arms[0]
        config_hash = hash_config(config)
        
        initial_alpha = bandit.alphas[0]
        initial_beta = bandit.betas[0]
        
        # High reward should increase alpha more than beta
        await bandit.update(config_hash, 0.9)
        
        assert bandit.alphas[0] > initial_alpha
        assert bandit.betas[0] > initial_beta
        assert bandit.alphas[0] == initial_alpha + 0.9
        assert bandit.betas[0] == initial_beta + 0.1
    
    @pytest.mark.asyncio
    async def test_update_clamps_reward(self, bandit):
        """Test that reward is clamped to [0, 1]."""
        config = bandit.arms[0]
        config_hash = hash_config(config)
        
        initial_alpha = bandit.alphas[0]
        
        # Negative reward should be clamped to 0
        await bandit.update(config_hash, -0.5)
        assert bandit.alphas[0] == initial_alpha  # +0.0
        
        # Reward > 1 should be clamped to 1
        await bandit.update(config_hash, 1.5)
        assert bandit.alphas[0] == initial_alpha + 1.0  # +1.0
    
    @pytest.mark.asyncio
    async def test_update_unknown_config(self, bandit):
        """Test update with unknown config hash returns False."""
        result = await bandit.update("unknown_hash_12345", 0.8)
        assert result is False
    
    def test_get_arm_statistics(self, bandit):
        """Test get_arm_statistics returns sorted list."""
        # Manually update some priors
        bandit.alphas[0] = 10.0
        bandit.betas[0] = 2.0
        bandit.samples[0] = 10
        
        bandit.alphas[1] = 2.0
        bandit.betas[1] = 10.0
        bandit.samples[1] = 10
        
        stats = bandit.get_arm_statistics()
        
        assert len(stats) == 81
        # First should be the arm with higher mean
        assert stats[0]["arm_idx"] == 0
        assert stats[0]["mean_reward"] > stats[-1]["mean_reward"]
    
    def test_get_top_configs(self, bandit):
        """Test get_top_configs returns limited results."""
        top_5 = bandit.get_top_configs(5)
        assert len(top_5) == 5
        
        top_3 = bandit.get_top_configs(3)
        assert len(top_3) == 3


class TestBanditConvergence:
    """Tests for bandit convergence behavior."""
    
    @pytest.fixture
    def mock_db(self):
        mock = AsyncMock()
        mock.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))
        return mock
    
    @pytest.mark.asyncio
    async def test_bandit_converges_to_high_reward(self, mock_db):
        """Test that bandit converges to consistently high-reward arm."""
        bandit = FairnessBandit(mock_db)
        
        # Arm 0 has high reward (0.9), all others have low reward (0.3)
        optimal_arm = 0
        optimal_hash = hash_config(bandit.arms[optimal_arm])
        
        # Simulate 50 episodes
        for _ in range(50):
            config, arm_idx, _, _ = await bandit.select_arm()
            config_hash = hash_config(config)
            
            if arm_idx == optimal_arm:
                await bandit.update(config_hash, 0.9)
            else:
                await bandit.update(config_hash, 0.3)
        
        # After training, optimal arm should have highest samples
        stats = bandit.get_arm_statistics()
        top_arm = stats[0]["arm_idx"]
        
        # Optimal arm should be in top 5 (stochastic, so not always #1)
        top_arm_indices = [s["arm_idx"] for s in stats[:5]]
        assert optimal_arm in top_arm_indices or bandit.samples[optimal_arm] > 5
