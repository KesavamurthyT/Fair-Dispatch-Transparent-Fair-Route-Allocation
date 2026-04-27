"""
Unit tests for Final Resolution Agent.
Tests swap logic and fairness validation.
"""

import pytest
from uuid import UUID
from app.services.final_resolution import FinalResolutionAgent
from app.schemas.agent_schemas import (
    AllocationItem,
    DriverLiaisonDecision,
    FairnessMetrics,
    RoutePlanResult,
)

# Fixed UUIDs for deterministic tests
D1_UUID = UUID("11111111-1111-1111-1111-111111111111")
D2_UUID = UUID("22222222-2222-2222-2222-222222222222")
R1_UUID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
R2_UUID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


class TestFinalResolutionSwaps:
    """Tests for swap acceptance and rejection logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = FinalResolutionAgent(metric_epsilon=0.05)
    
    def test_swap_applied_improves_fairness(self):
        """Swap should be applied when it improves or maintains fairness."""
        # Create a proposal where driver-1 has high effort, driver-2 has low
        proposal = RoutePlanResult(
            allocation=[
                AllocationItem(driver_id=D1_UUID, route_id=R1_UUID, effort=80.0),
                AllocationItem(driver_id=D2_UUID, route_id=R2_UUID, effort=40.0),
            ],
            total_effort=120.0,
            avg_effort=60.0,
            per_driver_effort={str(D1_UUID): 80.0, str(D2_UUID): 40.0},
            proposal_number=1,
        )
        
        decisions = [
            DriverLiaisonDecision(
                driver_id=str(D1_UUID),
                decision="COUNTER",
                preferred_route_id=str(R2_UUID),  # Wants the easier route
                reason="Too heavy",
            ),
            DriverLiaisonDecision(
                driver_id=str(D2_UUID),
                decision="ACCEPT",
                reason="Within comfort",
            ),
        ]
        
        # Effort matrix: d1 can do r2 for 45, d2 can do r1 for 75
        effort_matrix = [
            [80.0, 45.0],  # d1's efforts for r1, r2
            [75.0, 40.0],  # d2's efforts for r1, r2
        ]
        
        current_metrics = FairnessMetrics(
            avg_effort=60.0,
            std_dev=20.0,
            max_gap=40.0,
            gini_index=0.167,
            min_effort=40.0,
            max_effort=80.0,
        )
        
        result = self.agent.resolve_counters(
            approved_proposal=proposal,
            decisions=decisions,
            effort_matrix=effort_matrix,
            driver_ids=[str(D1_UUID), str(D2_UUID)],
            route_ids=[str(R1_UUID), str(R2_UUID)],
            current_metrics=current_metrics,
        )
        
        # Swap should be applied: d1->r2 (45), d2->r1 (75)
        assert len(result.swaps_applied) == 1
        assert result.swaps_applied[0].driver_a == str(D1_UUID)
        assert result.per_driver_effort[str(D1_UUID)] == 45.0
        assert result.per_driver_effort[str(D2_UUID)] == 75.0
    
    def test_swap_rejected_worsens_fairness(self):
        """Swap should be rejected when it significantly worsens fairness."""
        proposal = RoutePlanResult(
            allocation=[
                AllocationItem(driver_id=D1_UUID, route_id=R1_UUID, effort=55.0),
                AllocationItem(driver_id=D2_UUID, route_id=R2_UUID, effort=45.0),
            ],
            total_effort=100.0,
            avg_effort=50.0,
            per_driver_effort={str(D1_UUID): 55.0, str(D2_UUID): 45.0},
            proposal_number=1,
        )
        
        decisions = [
            DriverLiaisonDecision(
                driver_id=str(D1_UUID),
                decision="COUNTER",
                preferred_route_id=str(R2_UUID),
                reason="Want easier",
            ),
        ]
        
        # Effort matrix: swap would make things much worse for d2
        effort_matrix = [
            [55.0, 40.0],  # d1: r1=55, r2=40
            [90.0, 45.0],  # d2: r1=90, r2=45 - huge penalty for d2
        ]
        
        current_metrics = FairnessMetrics(
            avg_effort=50.0,
            std_dev=5.0,
            max_gap=10.0,
            gini_index=0.05,
            min_effort=45.0,
            max_effort=55.0,
        )
        
        result = self.agent.resolve_counters(
            approved_proposal=proposal,
            decisions=decisions,
            effort_matrix=effort_matrix,
            driver_ids=[str(D1_UUID), str(D2_UUID)],
            route_ids=[str(R1_UUID), str(R2_UUID)],
            current_metrics=current_metrics,
        )
        
        # Swap should be rejected: d2 would go from 45 to 90 (100% increase)
        assert len(result.swaps_applied) == 0
        assert str(D1_UUID) in result.unfulfilled_counters
    
    def test_no_swaps_when_no_counters(self):
        """No swaps should occur when there are no COUNTER decisions."""
        proposal = RoutePlanResult(
            allocation=[
                AllocationItem(driver_id=D1_UUID, route_id=R1_UUID, effort=50.0),
                AllocationItem(driver_id=D2_UUID, route_id=R2_UUID, effort=50.0),
            ],
            total_effort=100.0,
            avg_effort=50.0,
            per_driver_effort={str(D1_UUID): 50.0, str(D2_UUID): 50.0},
            proposal_number=1,
        )
        
        decisions = [
            DriverLiaisonDecision(driver_id=str(D1_UUID), decision="ACCEPT", reason="OK"),
            DriverLiaisonDecision(driver_id=str(D2_UUID), decision="ACCEPT", reason="OK"),
        ]
        
        effort_matrix = [[50.0, 60.0], [60.0, 50.0]]
        
        current_metrics = FairnessMetrics(
            avg_effort=50.0,
            std_dev=0.0,
            max_gap=0.0,
            gini_index=0.0,
            min_effort=50.0,
            max_effort=50.0,
        )
        
        result = self.agent.resolve_counters(
            approved_proposal=proposal,
            decisions=decisions,
            effort_matrix=effort_matrix,
            driver_ids=[str(D1_UUID), str(D2_UUID)],
            route_ids=[str(R1_UUID), str(R2_UUID)],
            current_metrics=current_metrics,
        )
        
        assert len(result.swaps_applied) == 0
        assert len(result.unfulfilled_counters) == 0
    
    def test_unfulfilled_counter_recorded(self):
        """Unfulfilled counters should be recorded in result."""
        proposal = RoutePlanResult(
            allocation=[
                AllocationItem(driver_id=D1_UUID, route_id=R1_UUID, effort=70.0),
                AllocationItem(driver_id=D2_UUID, route_id=R2_UUID, effort=50.0),
            ],
            total_effort=120.0,
            avg_effort=60.0,
            per_driver_effort={str(D1_UUID): 70.0, str(D2_UUID): 50.0},
            proposal_number=1,
        )
        
        decisions = [
            DriverLiaisonDecision(
                driver_id=str(D1_UUID),
                decision="COUNTER",
                preferred_route_id="non-existent-route",  # Non-existent route
                reason="Want different route",
            ),
        ]
        
        effort_matrix = [[70.0, 55.0], [65.0, 50.0]]
        
        current_metrics = FairnessMetrics(
            avg_effort=60.0,
            std_dev=10.0,
            max_gap=20.0,
            gini_index=0.083,
            min_effort=50.0,
            max_effort=70.0,
        )
        
        result = self.agent.resolve_counters(
            approved_proposal=proposal,
            decisions=decisions,
            effort_matrix=effort_matrix,
            driver_ids=[str(D1_UUID), str(D2_UUID)],
            route_ids=[str(R1_UUID), str(R2_UUID)],
            current_metrics=current_metrics,
        )
        
        # Counter for non-existent route should be unfulfilled
        assert str(D1_UUID) in result.unfulfilled_counters


class TestFinalResolutionMetrics:
    """Tests for metric computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = FinalResolutionAgent()
    
    def test_compute_metrics(self):
        """Test metric computation."""
        efforts = [40.0, 50.0, 60.0]
        
        metrics = self.agent._compute_metrics(efforts)
        
        assert metrics["avg_effort"] == 50.0
        assert metrics["min_effort"] == 40.0
        assert metrics["max_effort"] == 60.0
        assert metrics["max_gap"] == 20.0
        assert metrics["gini_index"] > 0
    
    def test_compute_gini_perfect_equality(self):
        """Gini should be 0 for perfect equality."""
        values = [50.0, 50.0, 50.0]
        gini = self.agent._compute_gini(values)
        assert gini == 0.0
    
    def test_snapshot_generation(self):
        """Test input/output snapshot generation."""
        metrics = FairnessMetrics(
            avg_effort=50.0,
            std_dev=10.0,
            max_gap=20.0,
            gini_index=0.1,
            min_effort=40.0,
            max_effort=60.0,
        )
        
        input_snap = self.agent.get_input_snapshot(3, metrics, 50.0)
        assert input_snap["num_counters"] == 3
        assert input_snap["original_gini"] == 0.1
        
        from app.schemas.agent_schemas import FinalResolutionResult
        result = FinalResolutionResult(
            allocation=[],
            per_driver_effort={},
            metrics={"gini_index": 0.08},
            swaps_applied=[],
            unfulfilled_counters=[],
        )
        
        output_snap = self.agent.get_output_snapshot(result)
        assert output_snap["num_swaps_applied"] == 0
