"""
Unit tests for Explainability Agent v2 (Phase 4.3).
Tests classification logic and template generation.
"""

import pytest
from app.services.explainability import ExplainabilityAgent
from app.schemas.explainability import DriverExplanationInput, DriverExplanationOutput


class TestCategoryClassification:
    """Tests for _classify_category logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ExplainabilityAgent()
    
    def _base_input(self, **overrides) -> DriverExplanationInput:
        """Create base input with optional overrides."""
        defaults = {
            "driver_id": "driver-1",
            "driver_name": "Test Driver",
            "num_drivers": 5,
            "today_effort": 50.0,
            "today_rank": 3,
            "route_id": "route-1",
            "route_summary": {
                "num_packages": 20,
                "total_weight_kg": 50.0,
                "num_stops": 10,
                "difficulty_score": 2.5,
                "estimated_time_minutes": 180,
            },
            "effort_breakdown": {
                "physical_effort": 15.0,
                "route_complexity": 10.0,
                "time_pressure": 5.0,
            },
            "global_avg_effort": 50.0,
            "global_std_effort": 10.0,
            "global_gini_index": 0.15,
            "global_max_gap": 20.0,
            "history_efforts_last_7_days": [45.0, 50.0, 55.0],
            "history_hard_days_last_7": 1,
            "is_recovery_day": False,
            "had_manual_override": False,
            "liaison_decision": None,
            "swap_applied": False,
        }
        defaults.update(overrides)
        return DriverExplanationInput(**defaults)
    
    def test_near_avg_category(self):
        """Effort close to average should classify as NEAR_AVG."""
        data = self._base_input(today_effort=52.0, global_avg_effort=50.0)
        category = self.agent._classify_category(data)
        assert category == "NEAR_AVG"
    
    def test_heavy_category(self):
        """Above-average effort without negotiation should classify as HEAVY."""
        data = self._base_input(today_effort=70.0, global_avg_effort=50.0)
        category = self.agent._classify_category(data)
        assert category == "HEAVY"
    
    def test_heavy_with_swap_category(self):
        """Above-average with swap applied should classify as HEAVY_WITH_SWAP."""
        data = self._base_input(
            today_effort=70.0,
            global_avg_effort=50.0,
            swap_applied=True,
        )
        category = self.agent._classify_category(data)
        assert category == "HEAVY_WITH_SWAP"
    
    def test_heavy_no_swap_category(self):
        """Above-average with COUNTER but no swap should classify as HEAVY_NO_SWAP."""
        data = self._base_input(
            today_effort=70.0,
            global_avg_effort=50.0,
            liaison_decision="COUNTER",
            swap_applied=False,
        )
        category = self.agent._classify_category(data)
        assert category == "HEAVY_NO_SWAP"
    
    def test_recovery_category(self):
        """Explicit recovery day should classify as RECOVERY."""
        data = self._base_input(
            today_effort=35.0,
            global_avg_effort=50.0,
            is_recovery_day=True,
        )
        category = self.agent._classify_category(data)
        assert category == "RECOVERY"
    
    def test_light_recovery_category(self):
        """Below average with hard day streak should classify as LIGHT_RECOVERY."""
        data = self._base_input(
            today_effort=35.0,
            global_avg_effort=50.0,
            history_hard_days_last_7=3,
        )
        category = self.agent._classify_category(data)
        assert category == "LIGHT_RECOVERY"
    
    def test_light_category(self):
        """Below average without hard streak should classify as LIGHT."""
        data = self._base_input(
            today_effort=35.0,
            global_avg_effort=50.0,
            history_hard_days_last_7=0,
        )
        category = self.agent._classify_category(data)
        assert category == "LIGHT"


class TestDriverTextGeneration:
    """Tests for _build_driver_text templates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ExplainabilityAgent()
    
    def _base_input(self, **overrides) -> DriverExplanationInput:
        """Create base input with optional overrides."""
        defaults = {
            "driver_id": "driver-1",
            "driver_name": "Test Driver",
            "num_drivers": 5,
            "today_effort": 50.0,
            "today_rank": 3,
            "route_id": "route-1",
            "route_summary": {
                "num_packages": 20,
                "total_weight_kg": 50.0,
                "num_stops": 10,
                "difficulty_score": 2.5,
                "estimated_time_minutes": 180,
            },
            "effort_breakdown": {},
            "global_avg_effort": 50.0,
            "global_std_effort": 10.0,
            "global_gini_index": 0.15,
            "global_max_gap": 20.0,
            "history_efforts_last_7_days": [],
            "history_hard_days_last_7": 0,
            "is_recovery_day": False,
            "had_manual_override": False,
            "liaison_decision": None,
            "swap_applied": False,
        }
        defaults.update(overrides)
        return DriverExplanationInput(**defaults)
    
    def test_near_avg_contains_expected_phrases(self):
        """NEAR_AVG text should mention 'moderate' and 'balanced'."""
        data = self._base_input()
        text = self.agent._build_driver_text(data, "NEAR_AVG")
        
        assert "moderate" in text.lower()
        assert "balanced" in text.lower()
        assert "20 packages" in text
        assert "10 stops" in text
    
    def test_heavy_mentions_heavier(self):
        """HEAVY category text should mention 'heavier'."""
        data = self._base_input(today_effort=70.0)
        text = self.agent._build_driver_text(data, "HEAVY")
        
        assert "heavier" in text.lower()
    
    def test_recovery_mentions_lighter(self):
        """RECOVERY text should mention 'intentionally lighter' and 'recover'."""
        data = self._base_input(is_recovery_day=True)
        text = self.agent._build_driver_text(data, "RECOVERY")
        
        assert "lighter" in text.lower()
        assert "recover" in text.lower()
    
    def test_includes_packages_and_stops(self):
        """All templates should include package and stop counts."""
        data = self._base_input()
        
        for category in ["NEAR_AVG", "HEAVY", "LIGHT", "RECOVERY"]:
            text = self.agent._build_driver_text(data, category)
            assert "20 packages" in text
            assert "10 stops" in text


class TestAdminTextGeneration:
    """Tests for _build_admin_text templates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ExplainabilityAgent()
    
    def _base_input(self, **overrides) -> DriverExplanationInput:
        """Create base input with optional overrides."""
        defaults = {
            "driver_id": "driver-1",
            "driver_name": "John Doe",
            "num_drivers": 5,
            "today_effort": 65.0,
            "today_rank": 2,
            "route_id": "route-1",
            "route_summary": {
                "num_packages": 25,
                "total_weight_kg": 75.0,
                "num_stops": 12,
                "difficulty_score": 3.0,
                "estimated_time_minutes": 240,
            },
            "effort_breakdown": {
                "physical_effort": 25.0,
                "route_complexity": 20.0,
                "time_pressure": 10.0,
            },
            "global_avg_effort": 50.0,
            "global_std_effort": 12.0,
            "global_gini_index": 0.12,
            "global_max_gap": 25.0,
            "history_efforts_last_7_days": [55.0, 60.0],
            "history_hard_days_last_7": 2,
            "is_recovery_day": False,
            "had_manual_override": False,
            "liaison_decision": None,
            "swap_applied": False,
        }
        defaults.update(overrides)
        return DriverExplanationInput(**defaults)
    
    def test_admin_text_includes_driver_name(self):
        """Admin text should include driver name."""
        data = self._base_input()
        text = self.agent._build_admin_text(data, "NEAR_AVG")
        assert "John Doe" in text
    
    def test_admin_text_includes_effort_metrics(self):
        """Admin text should include effort score and comparison to average."""
        data = self._base_input()
        text = self.agent._build_admin_text(data, "NEAR_AVG")
        
        assert "65" in text  # effort score
        assert "50" in text  # average
    
    def test_admin_text_includes_gini(self):
        """Admin text should include Gini index."""
        data = self._base_input()
        text = self.agent._build_admin_text(data, "NEAR_AVG")
        
        assert "Gini" in text
        assert "0.12" in text
    
    def test_admin_text_includes_rank(self):
        """Admin text should include rank."""
        data = self._base_input()
        text = self.agent._build_admin_text(data, "NEAR_AVG")
        
        assert "2/5" in text  # rank/total
    
    def test_heavy_with_swap_mentions_swap(self):
        """HEAVY_WITH_SWAP should mention swap in admin text."""
        data = self._base_input(swap_applied=True)
        text = self.agent._build_admin_text(data, "HEAVY_WITH_SWAP")
        
        assert "swap" in text.lower()
    
    def test_recovery_mentions_hard_days(self):
        """RECOVERY text should mention hard days count."""
        data = self._base_input(is_recovery_day=True, history_hard_days_last_7=4)
        text = self.agent._build_admin_text(data, "RECOVERY")
        
        assert "4" in text
        assert "hard day" in text.lower() or "recovery" in text.lower()
    
    def test_manual_override_note_included(self):
        """Manual override should be noted in admin text."""
        data = self._base_input(had_manual_override=True)
        text = self.agent._build_admin_text(data, "NEAR_AVG")
        
        assert "override" in text.lower()


class TestBuildExplanationForDriver:
    """Tests for the main build_explanation_for_driver method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ExplainabilityAgent()
    
    def test_returns_correct_output_type(self):
        """Should return DriverExplanationOutput."""
        data = DriverExplanationInput(
            driver_id="d1",
            driver_name="Driver One",
            num_drivers=3,
            today_effort=50.0,
            today_rank=2,
            route_id="r1",
            route_summary={"num_packages": 15, "total_weight_kg": 40.0, "num_stops": 8, "difficulty_score": 2.0, "estimated_time_minutes": 120},
            global_avg_effort=50.0,
            global_std_effort=10.0,
            global_gini_index=0.1,
            global_max_gap=15.0,
        )
        
        result = self.agent.build_explanation_for_driver(data)
        
        assert isinstance(result, DriverExplanationOutput)
        assert isinstance(result.driver_explanation, str)
        assert isinstance(result.admin_explanation, str)
        assert isinstance(result.category, str)
    
    def test_driver_and_admin_texts_are_different(self):
        """Driver and admin explanations should have different detail levels."""
        data = DriverExplanationInput(
            driver_id="d1",
            driver_name="Driver One",
            num_drivers=3,
            today_effort=50.0,
            today_rank=2,
            route_id="r1",
            route_summary={"num_packages": 15, "total_weight_kg": 40.0, "num_stops": 8, "difficulty_score": 2.0, "estimated_time_minutes": 120},
            global_avg_effort=50.0,
            global_std_effort=10.0,
            global_gini_index=0.1,
            global_max_gap=15.0,
        )
        
        result = self.agent.build_explanation_for_driver(data)
        
        # Admin text should be longer and contain more metrics
        assert len(result.admin_explanation) > len(result.driver_explanation)
        assert "Gini" in result.admin_explanation
        assert "Gini" not in result.driver_explanation


class TestSnapshotGeneration:
    """Tests for DecisionLog snapshot helpers."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ExplainabilityAgent()
    
    def test_input_snapshot(self):
        """Input snapshot should include key metrics."""
        snapshot = self.agent.get_input_snapshot(
            num_drivers=5,
            avg_effort=55.0,
            std_effort=12.0,
            gini_index=0.15,
            category_counts={"NEAR_AVG": 3, "HEAVY": 2},
        )
        
        assert snapshot["num_drivers"] == 5
        assert snapshot["avg_effort"] == 55.0
        assert snapshot["gini_index"] == 0.15
    
    def test_output_snapshot(self):
        """Output snapshot should include totals and category counts."""
        category_counts = {"NEAR_AVG": 3, "HEAVY": 2}
        snapshot = self.agent.get_output_snapshot(
            total_explanations=5,
            category_counts=category_counts,
        )
        
        assert snapshot["total_explanations"] == 5
        assert snapshot["category_counts"]["NEAR_AVG"] == 3
        assert snapshot["category_counts"]["HEAVY"] == 2
