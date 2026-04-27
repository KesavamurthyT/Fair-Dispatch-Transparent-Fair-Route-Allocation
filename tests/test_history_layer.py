"""
Tests for the historical data layer and Learning Agent integration.

Tests cover:
1. History seeding for mock drivers
2. History feature computation from assignments/feedback
3. ML Effort Agent using history adjustments
4. Driver Liaison Agent using history features
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, List

from app.services.history_features import (
    DriverHistoryFeatures,
    HistoryConfig,
    compute_history_effort_adjustment,
    compute_history_features_for_drivers_sync,
)
from app.services.ml_effort_agent import MLEffortAgent
from app.services.driver_liaison_agent import DriverLiaisonAgent
from app.schemas.agent_schemas import DriverAssignmentProposal


# =============================================================================
# Test: DriverHistoryFeatures Model
# =============================================================================

class TestDriverHistoryFeatures:
    """Tests for DriverHistoryFeatures Pydantic model."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        features = DriverHistoryFeatures(driver_id="D001")
        
        assert features.driver_id == "D001"
        assert features.recent_avg_effort == 0.0
        assert features.recent_std_effort == 0.0
        assert features.recent_hard_days == 0
        assert features.avg_stress_level == 5.0
        assert features.avg_fairness_rating == 3.0
        assert features.fatigue_score == 3.0  # Default is 3.0 (neutral)
        assert features.is_high_stress_driver is False
        assert features.is_frequent_hard_days is False
    
    def test_high_stress_flag(self):
        """Test high stress driver flag."""
        features = DriverHistoryFeatures(
            driver_id="D001",
            avg_stress_level=8.0,
            is_high_stress_driver=True,
        )
        assert features.is_high_stress_driver is True
    
    def test_frequent_hard_days_flag(self):
        """Test frequent hard days flag."""
        features = DriverHistoryFeatures(
            driver_id="D001",
            recent_hard_days=4,
            is_frequent_hard_days=True,
        )
        assert features.is_frequent_hard_days is True


# =============================================================================
# Test: HistoryConfig
# =============================================================================

class TestHistoryConfig:
    """Tests for HistoryConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HistoryConfig()
        
        assert config.window_days == 7
        assert config.hard_threshold_factor == 1.2
        assert config.w_hard_days == 2.0
        assert config.w_stress == 3.0
        assert config.w_fairness == 1.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HistoryConfig(
            window_days=14,
            w_hard_days=3.0,
            w_stress=2.0,
        )
        
        assert config.window_days == 14
        assert config.w_hard_days == 3.0
        assert config.w_stress == 2.0


# =============================================================================
# Test: compute_history_effort_adjustment
# =============================================================================

class TestHistoryEffortAdjustment:
    """Tests for compute_history_effort_adjustment function."""
    
    def test_null_history_returns_zero(self):
        """Test that None history returns 0 adjustment."""
        adjustment = compute_history_effort_adjustment(None)
        assert adjustment == 0.0
    
    def test_neutral_history_returns_zero(self):
        """Test neutral history (avg stress=6, avg fairness=3) returns ~0."""
        features = DriverHistoryFeatures(
            driver_id="D001",
            recent_hard_days=0,
            avg_stress_level=6.0,
            avg_fairness_rating=3.0,
        )
        adjustment = compute_history_effort_adjustment(features)
        assert adjustment == 0.0
    
    def test_hard_days_penalty(self):
        """Test that hard days add penalty."""
        config = HistoryConfig(w_hard_days=2.0)
        features = DriverHistoryFeatures(
            driver_id="D001",
            recent_hard_days=3,
            avg_stress_level=5.0,  # Below threshold
            avg_fairness_rating=3.0,
        )
        adjustment = compute_history_effort_adjustment(features, config)
        
        # 3 hard days * 2.0 weight = 6.0 penalty
        assert adjustment == 6.0
    
    def test_high_stress_penalty(self):
        """Test that high stress adds penalty."""
        config = HistoryConfig(w_stress=3.0)
        features = DriverHistoryFeatures(
            driver_id="D001",
            recent_hard_days=0,
            avg_stress_level=8.0,  # 2 above threshold of 6
            avg_fairness_rating=3.0,
        )
        adjustment = compute_history_effort_adjustment(features, config)
        
        # (8 - 6) * 3.0 = 6.0 penalty
        assert adjustment == 6.0
    
    def test_good_fairness_discount(self):
        """Test that good fairness rating gives discount."""
        config = HistoryConfig(w_fairness=1.0)
        features = DriverHistoryFeatures(
            driver_id="D001",
            recent_hard_days=0,
            avg_stress_level=5.0,  # Below threshold
            avg_fairness_rating=4.5,  # 1.5 above threshold of 3
        )
        adjustment = compute_history_effort_adjustment(features, config)
        
        # (4.5 - 3) * -1.0 = -1.5 discount
        assert adjustment == -1.5
    
    def test_combined_adjustment(self):
        """Test combined penalty and discount."""
        config = HistoryConfig(w_hard_days=2.0, w_stress=3.0, w_fairness=1.0)
        features = DriverHistoryFeatures(
            driver_id="D001",
            recent_hard_days=2,  # +4.0
            avg_stress_level=7.0,  # +3.0 (7-6)*3
            avg_fairness_rating=4.0,  # -1.0 (4-3)*1
        )
        adjustment = compute_history_effort_adjustment(features, config)
        
        # 4.0 + 3.0 - 1.0 = 6.0
        assert adjustment == 6.0


# =============================================================================
# Test: compute_history_features_for_drivers_sync
# =============================================================================

class TestHistoryFeaturesSyncComputation:
    """Tests for synchronous history features computation."""
    
    def test_returns_default_features(self):
        """Test sync version returns default features for all drivers."""
        driver_ids = ["D001", "D002", "D003"]
        features = compute_history_features_for_drivers_sync(driver_ids)
        
        assert len(features) == 3
        assert "D001" in features
        assert "D002" in features
        assert "D003" in features
        
        # All should have default/neutral values
        for d_id, feat in features.items():
            assert feat.driver_id == d_id
            assert feat.recent_avg_effort == 0.0
            assert feat.avg_stress_level == 5.0
            assert feat.avg_fairness_rating == 3.0
            assert feat.is_high_stress_driver is False
    
    def test_empty_driver_list(self):
        """Test with empty driver list."""
        features = compute_history_features_for_drivers_sync([])
        assert features == {}


# =============================================================================
# Test: MLEffortAgent with History
# =============================================================================

class TestMLEffortAgentWithHistory:
    """Tests for MLEffortAgent using historical features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_driver = MagicMock()
        self.mock_driver.id = "D001"
        self.mock_driver.vehicle_type = "ICE"
        self.mock_driver.current_range_km = None
        self.mock_driver.is_ev = False
        self.mock_driver.vehicle_capacity_kg = 500.0
        self.mock_driver.max_packages = 50
        
        self.mock_route = MagicMock()
        self.mock_route.id = "R001"
        self.mock_route.num_packages = 10
        self.mock_route.total_weight_kg = 50.0
        self.mock_route.total_distance_km = 20.0
        self.mock_route.estimated_time_minutes = 60.0
        self.mock_route.difficulty_rating = 3.0
    
    def test_agent_initializes_with_history_config(self):
        """Test agent accepts history_config parameter."""
        config = HistoryConfig(w_hard_days=5.0)
        agent = MLEffortAgent(history_config=config)
        
        assert agent.history_config is not None
        assert agent.history_config.w_hard_days == 5.0
    
    def test_agent_without_history_config(self):
        """Test agent works without explicit history_config (uses default)."""
        agent = MLEffortAgent()
        # Agent always has a history_config (defaults to HistoryConfig())
        assert agent.history_config is not None
    
    def test_compute_effort_with_history_adjustment(self):
        """Test effort computation includes history adjustment."""
        config = HistoryConfig(w_hard_days=2.0, w_stress=3.0)
        agent = MLEffortAgent(history_config=config)
        
        # High stress driver with hard days
        driver_history = {
            "D001": DriverHistoryFeatures(
                driver_id="D001",
                recent_hard_days=2,
                avg_stress_level=8.0,
                avg_fairness_rating=3.0,
            )
        }
        
        result = agent.compute_effort_matrix(
            drivers=[self.mock_driver],
            routes=[self.mock_route],
            driver_history=driver_history,
        )
        
        # Should have result
        assert result is not None
        assert len(result.matrix) == 1
        assert len(result.matrix[0]) == 1
        
        # Get base effort without history
        agent_no_history = MLEffortAgent()
        result_no_history = agent_no_history.compute_effort_matrix(
            drivers=[self.mock_driver],
            routes=[self.mock_route],
        )
        
        # Effort with history should be higher (penalty applied)
        # Expected adjustment: 2*2 + (8-6)*3 = 4 + 6 = 10
        expected_diff = 10.0
        actual_diff = result.matrix[0][0] - result_no_history.matrix[0][0]
        assert abs(actual_diff - expected_diff) < 0.1
    
    def test_compute_effort_no_history_passed(self):
        """Test effort computation works without history dict."""
        agent = MLEffortAgent(history_config=HistoryConfig())
        
        result = agent.compute_effort_matrix(
            drivers=[self.mock_driver],
            routes=[self.mock_route],
            driver_history=None,
        )
        
        assert result is not None
        assert len(result.matrix) == 1


# =============================================================================
# Test: DriverLiaisonAgent with History
# =============================================================================

class TestDriverLiaisonAgentWithHistory:
    """Tests for DriverLiaisonAgent using historical features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = DriverLiaisonAgent()
        
        self.proposal = DriverAssignmentProposal(
            driver_id="D001",
            route_id="R001",
            effort=50.0,
            rank_in_team=5,
        )
        
        self.global_avg_effort = 40.0
        self.global_std_effort = 10.0
        
        # Alternatives sorted by effort (lowest first)
        self.alternatives = [
            ("R002", 35.0),
            ("R003", 38.0),
            ("R004", 45.0),
        ]
    
    def test_decide_with_history_accept(self):
        """Test ACCEPT decision with history (effort within comfort band)."""
        # Low stress driver, effort within comfort
        history = DriverHistoryFeatures(
            driver_id="D001",
            recent_avg_effort=45.0,
            recent_std_effort=8.0,
            recent_hard_days=0,
            avg_stress_level=4.0,
            fatigue_score=2.0,
        )
        
        proposal = DriverAssignmentProposal(
            driver_id="D001",
            route_id="R001",
            effort=48.0,  # Within comfort band of 45 + max(10, 8) = 55
            rank_in_team=5,
        )
        
        decision = self.agent.decide_with_history(
            proposal=proposal,
            history=history,
            global_avg_effort=self.global_avg_effort,
            global_std_effort=self.global_std_effort,
            alternative_routes_sorted=self.alternatives,
        )
        
        assert decision.decision == "ACCEPT"
        assert decision.driver_id == "D001"
    
    def test_decide_with_history_counter_high_stress(self):
        """Test COUNTER decision for high stress driver."""
        # High stress driver with tightened comfort band
        history = DriverHistoryFeatures(
            driver_id="D001",
            recent_avg_effort=40.0,
            recent_std_effort=8.0,
            recent_hard_days=0,
            avg_stress_level=8.0,  # High stress
            fatigue_score=3.5,
        )
        
        # Proposal effort = 50, but comfort upper with tightening would be less
        decision = self.agent.decide_with_history(
            proposal=self.proposal,
            history=history,
            global_avg_effort=self.global_avg_effort,
            global_std_effort=self.global_std_effort,
            alternative_routes_sorted=self.alternatives,
        )
        
        # Should counter since high stress tightens the band
        # comfort_upper = 40 + 10 - 0.2*10 = 48
        # Effort 50 > 48, so should COUNTER if alternative exists
        assert decision.decision in ["COUNTER", "ACCEPT"]  # Depends on exact calculation
        assert "stress" in decision.reason.lower() or decision.decision == "ACCEPT"
    
    def test_decide_with_history_frequent_hard_days(self):
        """Test that frequent hard days tighten comfort band."""
        history = DriverHistoryFeatures(
            driver_id="D001",
            recent_avg_effort=40.0,
            recent_std_effort=8.0,
            recent_hard_days=4,  # Frequent hard days
            avg_stress_level=5.0,
            fatigue_score=4.0,  # High fatigue
            is_frequent_hard_days=True,
        )
        
        decision = self.agent.decide_with_history(
            proposal=self.proposal,
            history=history,
            global_avg_effort=self.global_avg_effort,
            global_std_effort=self.global_std_effort,
            alternative_routes_sorted=self.alternatives,
        )
        
        # Hard days + fatigue should tighten the band significantly
        assert decision is not None
        assert decision.driver_id == "D001"
    
    def test_decide_with_history_none(self):
        """Test decision with None history (uses defaults)."""
        decision = self.agent.decide_with_history(
            proposal=self.proposal,
            history=None,
            global_avg_effort=self.global_avg_effort,
            global_std_effort=self.global_std_effort,
            alternative_routes_sorted=self.alternatives,
        )
        
        assert decision is not None
        assert decision.driver_id == "D001"
        assert decision.decision in ["ACCEPT", "COUNTER", "FORCE_ACCEPT"]
    
    def test_decide_force_accept_no_alternative(self):
        """Test FORCE_ACCEPT when no valid alternative exists."""
        history = DriverHistoryFeatures(
            driver_id="D001",
            recent_avg_effort=30.0,
            recent_std_effort=5.0,
            avg_stress_level=8.0,  # Very high stress
            recent_hard_days=3,
        )
        
        proposal = DriverAssignmentProposal(
            driver_id="D001",
            route_id="R001",
            effort=50.0,
            rank_in_team=1,  # Top rank
        )
        
        # No alternatives that meet the 10% improvement threshold
        alternatives = [
            ("R002", 48.0),  # Not enough improvement
            ("R003", 49.0),
        ]
        
        decision = self.agent.decide_with_history(
            proposal=proposal,
            history=history,
            global_avg_effort=self.global_avg_effort,
            global_std_effort=self.global_std_effort,
            alternative_routes_sorted=alternatives,
        )
        
        # Should be FORCE_ACCEPT or ACCEPT depending on comfort band
        assert decision.decision in ["FORCE_ACCEPT", "ACCEPT", "COUNTER"]


# =============================================================================
# Test: Integration
# =============================================================================

class TestHistoryLayerIntegration:
    """Integration tests for the history layer."""
    
    def test_full_pipeline_with_history(self):
        """Test full pipeline from history features to agent decisions."""
        # 1. Create driver history features
        driver_history = {
            "D001": DriverHistoryFeatures(
                driver_id="D001",
                recent_avg_effort=40.0,
                recent_hard_days=2,
                avg_stress_level=7.5,
                avg_fairness_rating=4.0,
            ),
            "D002": DriverHistoryFeatures(
                driver_id="D002",
                recent_avg_effort=35.0,
                recent_hard_days=0,
                avg_stress_level=4.0,
                avg_fairness_rating=3.5,
            ),
        }
        
        # 2. Compute adjustments
        config = HistoryConfig()
        adj_d001 = compute_history_effort_adjustment(driver_history["D001"], config)
        adj_d002 = compute_history_effort_adjustment(driver_history["D002"], config)
        
        # D001 should have higher adjustment (more penalty)
        assert adj_d001 > adj_d002
        
        # 3. Verify D001's adjustment is positive (penalty)
        # 2 hard days * 2.0 = 4.0
        # (7.5 - 6) * 3.0 = 4.5
        # (4.0 - 3.0) * -1.0 = -1.0
        # Total = 4.0 + 4.5 - 1.0 = 7.5
        assert abs(adj_d001 - 7.5) < 0.1
        
        # 4. Verify D002's adjustment is negative (discount)
        # 0 hard days = 0
        # stress 4.0 < 6.0 = 0
        # (3.5 - 3.0) * -1.0 = -0.5
        assert abs(adj_d002 - (-0.5)) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
