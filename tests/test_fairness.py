"""
Unit tests for fairness metrics calculations.
Tests Gini index and fairness score functions.
"""

import pytest
from app.services.fairness import (
    gini_index,
    calculate_fairness_score,
    calculate_global_fairness,
)


class TestGiniIndex:
    """Tests for the Gini index calculation."""
    
    def test_gini_perfect_equality(self):
        """All equal workloads should give Gini = 0."""
        workloads = [50.0, 50.0, 50.0, 50.0]
        result = gini_index(workloads)
        assert result == 0.0
    
    def test_gini_maximum_inequality(self):
        """One person has all work, Gini should be close to 1."""
        workloads = [100.0, 0.0, 0.0, 0.0]
        result = gini_index(workloads)
        # For n=4, maximum Gini is (n-1)/n = 0.75
        assert result > 0.7
    
    def test_gini_moderate_inequality(self):
        """Moderate spread should give moderate Gini."""
        workloads = [20.0, 40.0, 60.0, 80.0]
        result = gini_index(workloads)
        assert 0.1 < result < 0.4
    
    def test_gini_empty_list(self):
        """Empty list should return 0."""
        result = gini_index([])
        assert result == 0.0
    
    def test_gini_single_element(self):
        """Single element should return 0 (no inequality possible)."""
        result = gini_index([100.0])
        assert result == 0.0
    
    def test_gini_two_elements_equal(self):
        """Two equal elements should return 0."""
        result = gini_index([50.0, 50.0])
        assert result == 0.0
    
    def test_gini_two_elements_unequal(self):
        """Two unequal elements should give positive Gini."""
        result = gini_index([30.0, 70.0])
        assert result > 0.0
        assert result < 1.0
    
    def test_gini_zero_sum(self):
        """All zeros should return 0."""
        result = gini_index([0.0, 0.0, 0.0])
        assert result == 0.0
    
    def test_gini_realistic_workloads(self):
        """Realistic workload distribution."""
        # Simulating a fairly balanced day
        workloads = [62.5, 58.3, 71.2, 65.8, 55.4]
        result = gini_index(workloads)
        assert 0.0 < result < 0.15  # Should be well balanced


class TestFairnessScore:
    """Tests for individual fairness score calculation."""
    
    def test_fairness_exactly_average(self):
        """Workload exactly at average should give score of 1.0."""
        result = calculate_fairness_score(50.0, 50.0)
        assert result == 1.0
    
    def test_fairness_above_average(self):
        """Workload above average should give score < 1.0."""
        result = calculate_fairness_score(75.0, 50.0)
        assert result < 1.0
        assert result >= 0.0
    
    def test_fairness_below_average(self):
        """Workload below average should give score < 1.0."""
        result = calculate_fairness_score(25.0, 50.0)
        assert result < 1.0
        assert result >= 0.0
    
    def test_fairness_symmetric(self):
        """Same deviation above and below should give same score."""
        score_above = calculate_fairness_score(60.0, 50.0)
        score_below = calculate_fairness_score(40.0, 50.0)
        assert abs(score_above - score_below) < 0.01
    
    def test_fairness_zero_average(self):
        """Zero average should use 1.0 as denominator."""
        result = calculate_fairness_score(5.0, 0.0)
        assert result >= 0.0
        assert result <= 1.0
    
    def test_fairness_double_average(self):
        """Workload at double average."""
        result = calculate_fairness_score(100.0, 50.0)
        # |100 - 50| / 50 = 1.0, so fairness = 1 - 1 = 0
        assert result == 0.0


class TestGlobalFairness:
    """Tests for global fairness metrics calculation."""
    
    def test_global_fairness_empty(self):
        """Empty workloads should return zero metrics."""
        result = calculate_global_fairness([])
        assert result.avg_workload == 0.0
        assert result.std_dev == 0.0
        assert result.gini_index == 0.0
    
    def test_global_fairness_single(self):
        """Single workload should have zero std dev and gini."""
        result = calculate_global_fairness([50.0])
        assert result.avg_workload == 50.0
        assert result.std_dev == 0.0
        assert result.gini_index == 0.0
    
    def test_global_fairness_equal(self):
        """Equal workloads should have zero std dev and gini."""
        result = calculate_global_fairness([50.0, 50.0, 50.0])
        assert result.avg_workload == 50.0
        assert result.std_dev == 0.0
        assert result.gini_index == 0.0
    
    def test_global_fairness_varied(self):
        """Varied workloads should have positive std dev and gini."""
        result = calculate_global_fairness([30.0, 50.0, 70.0])
        assert result.avg_workload == 50.0
        assert result.std_dev > 0.0
        assert result.gini_index > 0.0
    
    def test_global_fairness_realistic(self):
        """Test with realistic workload values."""
        workloads = [62.5, 58.3, 71.2, 65.8, 55.4]
        result = calculate_global_fairness(workloads)
        
        # Check average
        expected_avg = sum(workloads) / len(workloads)
        assert abs(result.avg_workload - expected_avg) < 0.1
        
        # Check that metrics are reasonable
        assert result.std_dev > 0
        assert result.gini_index > 0
        assert result.gini_index < 0.2  # Should be fairly balanced
