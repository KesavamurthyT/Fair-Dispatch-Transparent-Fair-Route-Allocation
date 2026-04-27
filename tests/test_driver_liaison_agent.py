"""
Unit tests for Driver Liaison Agent.
Tests decision logic for ACCEPT, COUNTER, and FORCE_ACCEPT scenarios.
"""

import pytest
from app.services.driver_liaison_agent import DriverLiaisonAgent
from app.schemas.agent_schemas import (
    DriverAssignmentProposal,
    DriverContext,
    DriverLiaisonDecision,
)


class TestDriverLiaisonDecision:
    """Tests for individual driver decisions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = DriverLiaisonAgent()
    
    def test_accept_when_effort_within_comfort_band(self):
        """ACCEPT when effort is within driver's comfort band."""
        proposal = DriverAssignmentProposal(
            driver_id="driver-1",
            route_id="route-1",
            effort=50.0,
            rank_in_team=3,
        )
        context = DriverContext(
            driver_id="driver-1",
            recent_avg_effort=45.0,
            recent_std_effort=10.0,
            recent_hard_days=0,
            fatigue_score=2.0,
            preferences={},
        )
        
        decision = self.agent.decide_for_driver(
            proposal=proposal,
            context=context,
            global_avg_effort=50.0,
            global_std_effort=8.0,
            alternative_routes_sorted=[("route-2", 40.0), ("route-3", 55.0)],
        )
        
        assert decision.decision == "ACCEPT"
        assert "within comfort band" in decision.reason.lower()
    
    def test_counter_when_effort_above_threshold(self):
        """COUNTER when effort exceeds comfort and better alternative exists."""
        proposal = DriverAssignmentProposal(
            driver_id="driver-1",
            route_id="route-1",
            effort=80.0,  # Much higher than average
            rank_in_team=1,
        )
        context = DriverContext(
            driver_id="driver-1",
            recent_avg_effort=50.0,
            recent_std_effort=8.0,
            recent_hard_days=0,
            fatigue_score=3.0,
            preferences={},
        )
        
        decision = self.agent.decide_for_driver(
            proposal=proposal,
            context=context,
            global_avg_effort=50.0,
            global_std_effort=10.0,
            # Route-2 is 35% lighter (80 -> 52 is > 10% reduction)
            alternative_routes_sorted=[("route-2", 52.0), ("route-3", 70.0)],
        )
        
        assert decision.decision == "COUNTER"
        assert decision.preferred_route_id == "route-2"
        assert "prefer route" in decision.reason.lower()
    
    def test_force_accept_when_no_alternatives(self):
        """FORCE_ACCEPT when effort too high but no fair alternative available."""
        proposal = DriverAssignmentProposal(
            driver_id="driver-1",
            route_id="route-1",
            effort=80.0,
            rank_in_team=1,
        )
        context = DriverContext(
            driver_id="driver-1",
            recent_avg_effort=50.0,
            recent_std_effort=8.0,
            recent_hard_days=0,
            fatigue_score=3.0,
            preferences={},
        )
        
        decision = self.agent.decide_for_driver(
            proposal=proposal,
            context=context,
            global_avg_effort=50.0,
            global_std_effort=10.0,
            # All alternatives are similar or heavier
            alternative_routes_sorted=[("route-2", 78.0), ("route-3", 85.0)],
        )
        
        assert decision.decision == "FORCE_ACCEPT"
        assert "no fair alternative" in decision.reason.lower()
    
    def test_fatigue_tightens_threshold(self):
        """High fatigue should tighten comfort threshold."""
        proposal = DriverAssignmentProposal(
            driver_id="driver-1",
            route_id="route-1",
            effort=62.0,  # Slightly above normal comfort
            rank_in_team=2,
        )
        context = DriverContext(
            driver_id="driver-1",
            recent_avg_effort=50.0,
            recent_std_effort=10.0,
            recent_hard_days=0,
            fatigue_score=4.5,  # High fatigue
            preferences={},
        )
        
        decision = self.agent.decide_for_driver(
            proposal=proposal,
            context=context,
            global_avg_effort=50.0,
            global_std_effort=10.0,
            alternative_routes_sorted=[("route-2", 50.0), ("route-3", 55.0)],
        )
        
        # High fatigue should trigger COUNTER even for moderate effort
        assert decision.decision in ["COUNTER", "FORCE_ACCEPT"]
    
    def test_streak_tightens_threshold(self):
        """Hard day streak should tighten comfort threshold."""
        proposal = DriverAssignmentProposal(
            driver_id="driver-1",
            route_id="route-1",
            effort=62.0,
            rank_in_team=2,
        )
        context = DriverContext(
            driver_id="driver-1",
            recent_avg_effort=50.0,
            recent_std_effort=10.0,
            recent_hard_days=4,  # 4 consecutive hard days
            fatigue_score=3.0,
            preferences={},
        )
        
        decision = self.agent.decide_for_driver(
            proposal=proposal,
            context=context,
            global_avg_effort=50.0,
            global_std_effort=10.0,
            alternative_routes_sorted=[("route-2", 48.0), ("route-3", 55.0)],
        )
        
        # Streak should trigger COUNTER
        assert decision.decision in ["COUNTER", "FORCE_ACCEPT"]


class TestDriverLiaisonBatch:
    """Tests for batch processing of driver decisions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = DriverLiaisonAgent()
    
    def test_run_for_all_drivers(self):
        """Test batch processing returns correct counts."""
        proposals = [
            DriverAssignmentProposal(
                driver_id="driver-1",
                route_id="route-1",
                effort=50.0,  # Normal
                rank_in_team=2,
            ),
            DriverAssignmentProposal(
                driver_id="driver-2",
                route_id="route-2",
                effort=80.0,  # High
                rank_in_team=1,
            ),
        ]
        
        contexts = {
            "driver-1": DriverContext(
                driver_id="driver-1",
                recent_avg_effort=50.0,
                recent_std_effort=10.0,
                recent_hard_days=0,
                fatigue_score=2.0,
                preferences={},
            ),
            "driver-2": DriverContext(
                driver_id="driver-2",
                recent_avg_effort=50.0,
                recent_std_effort=10.0,
                recent_hard_days=0,
                fatigue_score=3.0,
                preferences={},
            ),
        }
        
        # Simple 2x2 effort matrix
        effort_matrix = [
            [50.0, 70.0],  # driver-1's efforts for route-1, route-2
            [55.0, 80.0],  # driver-2's efforts for route-1, route-2
        ]
        
        result = self.agent.run_for_all_drivers(
            proposals=proposals,
            driver_contexts=contexts,
            effort_matrix=effort_matrix,
            driver_ids=["driver-1", "driver-2"],
            route_ids=["route-1", "route-2"],
            global_avg_effort=65.0,
            global_std_effort=15.0,
        )
        
        assert len(result.decisions) == 2
        assert result.num_accept + result.num_counter + result.num_force_accept == 2
    
    def test_snapshot_generation(self):
        """Test input/output snapshot generation."""
        proposals = [
            DriverAssignmentProposal(
                driver_id="d1",
                route_id="r1",
                effort=50.0,
                rank_in_team=1,
            ),
        ]
        
        input_snap = self.agent.get_input_snapshot(proposals, 50.0, 10.0)
        assert "num_drivers" in input_snap
        assert input_snap["num_drivers"] == 1
        
        from app.schemas.agent_schemas import NegotiationResult
        result = NegotiationResult(
            decisions=[],
            num_accept=1,
            num_counter=0,
            num_force_accept=0,
        )
        
        output_snap = self.agent.get_output_snapshot(result)
        assert output_snap["num_accept"] == 1
