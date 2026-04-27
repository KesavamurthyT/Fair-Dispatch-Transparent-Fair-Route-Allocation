"""
Unit tests for FairnessManagerAgent.
Tests Gini calculation, metric computation, and ACCEPT/REOPTIMIZE decision logic.
"""

import pytest

from app.services.fairness_manager_agent import FairnessManagerAgent
from app.schemas.agent_schemas import (
    AllocationItem,
    FairnessThresholds,
    RoutePlanResult,
)
from uuid import uuid4


def create_mock_plan_result(efforts: list[float]) -> RoutePlanResult:
    """Create a mock RoutePlanResult from effort values."""
    allocation = []
    per_driver_effort = {}
    
    for i, effort in enumerate(efforts):
        driver_id = uuid4()
        route_id = uuid4()
        allocation.append(AllocationItem(
            driver_id=driver_id,
            route_id=route_id,
            effort=effort,
        ))
        per_driver_effort[str(driver_id)] = effort
    
    return RoutePlanResult(
        allocation=allocation,
        total_effort=sum(efforts),
        avg_effort=sum(efforts) / len(efforts) if efforts else 0.0,
        per_driver_effort=per_driver_effort,
        proposal_number=1,
    )


class TestFairnessManagerAgent:
    """Test suite for FairnessManagerAgent."""
    
    def test_gini_perfect_equality(self):
        """Test Gini index for perfectly equal distribution."""
        agent = FairnessManagerAgent()
        
        # All drivers have same effort
        plan = create_mock_plan_result([50.0, 50.0, 50.0, 50.0])
        result = agent.check(plan)
        
        assert result.metrics.gini_index == 0.0, "Perfect equality should have Gini = 0"
    
    def test_gini_inequality(self):
        """Test Gini index for unequal distribution."""
        agent = FairnessManagerAgent()
        
        # Unequal efforts
        plan = create_mock_plan_result([10.0, 20.0, 30.0, 100.0])
        result = agent.check(plan)
        
        assert result.metrics.gini_index > 0.0, "Unequal distribution should have Gini > 0"
        assert result.metrics.gini_index <= 1.0, "Gini should not exceed 1"
    
    def test_std_dev_calculation(self):
        """Test standard deviation calculation."""
        agent = FairnessManagerAgent()
        
        # Known values for easy verification
        plan = create_mock_plan_result([60.0, 60.0, 60.0])  # Zero std dev
        result = agent.check(plan)
        
        assert result.metrics.std_dev == 0.0, "Equal values should have std_dev = 0"
        
        # Varied values
        plan2 = create_mock_plan_result([50.0, 60.0, 70.0])
        result2 = agent.check(plan2)
        
        assert result2.metrics.std_dev > 0.0, "Varied values should have std_dev > 0"
    
    def test_max_gap_calculation(self):
        """Test max gap (max - min) calculation."""
        agent = FairnessManagerAgent()
        
        plan = create_mock_plan_result([30.0, 50.0, 80.0])
        result = agent.check(plan)
        
        expected_gap = 80.0 - 30.0
        assert result.metrics.max_gap == expected_gap
        assert result.metrics.min_effort == 30.0
        assert result.metrics.max_effort == 80.0
    
    def test_accept_when_within_thresholds(self):
        """Test ACCEPT status when all thresholds met."""
        agent = FairnessManagerAgent(
            thresholds=FairnessThresholds(
                gini_threshold=0.5,
                stddev_threshold=50.0,
                max_gap_threshold=50.0,
            )
        )
        
        # Equal-ish distribution
        plan = create_mock_plan_result([55.0, 60.0, 65.0])
        result = agent.check(plan)
        
        assert result.status == "ACCEPT"
        assert result.recommendations is None
    
    def test_reoptimize_when_gini_exceeds_threshold(self):
        """Test REOPTIMIZE status when Gini exceeds threshold."""
        agent = FairnessManagerAgent(
            thresholds=FairnessThresholds(
                gini_threshold=0.1,  # Very strict
                stddev_threshold=100.0,
                max_gap_threshold=100.0,
            )
        )
        
        # Unequal distribution
        plan = create_mock_plan_result([10.0, 50.0, 90.0])
        result = agent.check(plan)
        
        assert result.status == "REOPTIMIZE"
        assert result.recommendations is not None
    
    def test_reoptimize_when_stddev_exceeds_threshold(self):
        """Test REOPTIMIZE status when std_dev exceeds threshold."""
        agent = FairnessManagerAgent(
            thresholds=FairnessThresholds(
                gini_threshold=1.0,
                stddev_threshold=5.0,  # Very strict
                max_gap_threshold=100.0,
            )
        )
        
        plan = create_mock_plan_result([30.0, 50.0, 70.0])
        result = agent.check(plan)
        
        assert result.status == "REOPTIMIZE"
    
    def test_reoptimize_when_max_gap_exceeds_threshold(self):
        """Test REOPTIMIZE status when max_gap exceeds threshold."""
        agent = FairnessManagerAgent(
            thresholds=FairnessThresholds(
                gini_threshold=1.0,
                stddev_threshold=100.0,
                max_gap_threshold=10.0,  # Very strict
            )
        )
        
        plan = create_mock_plan_result([40.0, 50.0, 80.0])  # Gap = 40
        result = agent.check(plan)
        
        assert result.status == "REOPTIMIZE"
    
    def test_recommendations_identify_high_effort_drivers(self):
        """Test that recommendations identify high-effort drivers."""
        agent = FairnessManagerAgent(
            thresholds=FairnessThresholds(
                gini_threshold=0.1,
                stddev_threshold=5.0,
                max_gap_threshold=10.0,
            )
        )
        
        plan = create_mock_plan_result([50.0, 50.0, 50.0, 100.0])  # One outlier
        result = agent.check(plan)
        
        assert result.status == "REOPTIMIZE"
        assert result.recommendations is not None
        assert len(result.recommendations.high_effort_driver_ids) > 0
        assert result.recommendations.penalty_factor >= 1.0
    
    def test_outlier_count(self):
        """Test outlier counting (drivers above avg + 2*std_dev)."""
        agent = FairnessManagerAgent()
        
        # Create distribution with clear outlier
        # [10] * 8 + [100]: Mean=20, Std=30, Threshold=20+2*30=80. 100 > 80.
        plan = create_mock_plan_result([10.0] * 8 + [100.0])
        result = agent.check(plan)
        
        assert result.metrics.outlier_count >= 1
    
    def test_empty_plan(self):
        """Test handling of empty plan."""
        agent = FairnessManagerAgent()
        
        plan = create_mock_plan_result([])
        result = agent.check(plan)
        
        assert result.status == "ACCEPT"
        assert result.metrics.gini_index == 0.0
        assert result.metrics.std_dev == 0.0
    
    def test_single_driver(self):
        """Test handling of single driver."""
        agent = FairnessManagerAgent()
        
        plan = create_mock_plan_result([75.0])
        result = agent.check(plan)
        
        assert result.status == "ACCEPT"
        assert result.metrics.gini_index == 0.0
        assert result.metrics.std_dev == 0.0
        assert result.metrics.max_gap == 0.0
    
    def test_thresholds_included_in_result(self):
        """Test that used thresholds are included in result."""
        thresholds = FairnessThresholds(
            gini_threshold=0.25,
            stddev_threshold=20.0,
            max_gap_threshold=30.0,
        )
        agent = FairnessManagerAgent(thresholds=thresholds)
        
        plan = create_mock_plan_result([50.0, 60.0])
        result = agent.check(plan)
        
        assert result.thresholds_used["gini_threshold"] == 0.25
        assert result.thresholds_used["stddev_threshold"] == 20.0
        assert result.thresholds_used["max_gap_threshold"] == 30.0
    
    def test_snapshot_generation(self):
        """Test input/output snapshot generation."""
        agent = FairnessManagerAgent()
        
        plan = create_mock_plan_result([40.0, 50.0, 60.0])
        
        input_snapshot = agent.get_input_snapshot(plan)
        assert "proposal_number" in input_snapshot
        assert "num_drivers" in input_snapshot
        
        result = agent.check(plan)
        output_snapshot = agent.get_output_snapshot(result)
        assert "status" in output_snapshot
        assert "gini_index" in output_snapshot
