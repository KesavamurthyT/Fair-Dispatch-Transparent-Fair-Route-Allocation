"""
Unit tests for workload score calculations.
Tests route difficulty and workload score functions.
"""

import pytest
from app.services.workload import (
    calculate_route_difficulty,
    estimate_route_time,
    calculate_workload,
    RouteMetrics,
)


class TestRouteDifficulty:
    """Tests for route difficulty calculation."""
    
    def test_difficulty_base(self):
        """Minimal route should have base difficulty."""
        result = calculate_route_difficulty(
            total_weight_kg=0.0,
            num_stops=0,
            avg_fragility=1.0,
        )
        # Base difficulty is 1.0
        assert result >= 1.0
    
    def test_difficulty_increases_with_weight(self):
        """Higher weight should increase difficulty."""
        light = calculate_route_difficulty(10.0, 5, 1.0)
        heavy = calculate_route_difficulty(50.0, 5, 1.0)
        assert heavy > light
    
    def test_difficulty_increases_with_stops(self):
        """More stops should increase difficulty."""
        few_stops = calculate_route_difficulty(20.0, 3, 1.0)
        many_stops = calculate_route_difficulty(20.0, 15, 1.0)
        assert many_stops > few_stops
    
    def test_difficulty_increases_with_fragility(self):
        """Higher fragility should increase difficulty."""
        low_fragility = calculate_route_difficulty(20.0, 5, 1.0)
        high_fragility = calculate_route_difficulty(20.0, 5, 5.0)
        assert high_fragility > low_fragility
    
    def test_difficulty_realistic_route(self):
        """Test realistic route parameters."""
        result = calculate_route_difficulty(
            total_weight_kg=45.0,
            num_stops=12,
            avg_fragility=2.5,
        )
        # Should be a moderate difficulty
        assert 1.0 < result < 10.0


class TestEstimateRouteTime:
    """Tests for route time estimation."""
    
    def test_time_base(self):
        """Empty route should have base time."""
        result = estimate_route_time(0, 0)
        assert result >= 30  # Base time is 30 minutes
    
    def test_time_increases_with_packages(self):
        """More packages should increase time."""
        few = estimate_route_time(5, 3)
        many = estimate_route_time(25, 3)
        assert many > few
    
    def test_time_increases_with_stops(self):
        """More stops should increase time."""
        few_stops = estimate_route_time(10, 3)
        many_stops = estimate_route_time(10, 12)
        assert many_stops > few_stops
    
    def test_time_with_distance(self):
        """Distance should add to time."""
        no_distance = estimate_route_time(10, 5, 0.0)
        with_distance = estimate_route_time(10, 5, 30.0)
        assert with_distance > no_distance
    
    def test_time_realistic_route(self):
        """Test realistic route."""
        result = estimate_route_time(
            num_packages=20,
            num_stops=10,
        )
        # Should take 2-3 hours
        assert 90 < result < 180


class TestWorkloadScore:
    """Tests for workload score calculation."""
    
    def test_workload_dict_input(self):
        """Test with dictionary input."""
        route = {
            "num_packages": 20,
            "total_weight_kg": 40.0,
            "route_difficulty_score": 2.0,
            "estimated_time_minutes": 120,
        }
        result = calculate_workload(route)
        assert result > 0
    
    def test_workload_dataclass_input(self):
        """Test with RouteMetrics dataclass input."""
        route = RouteMetrics(
            num_packages=20,
            total_weight_kg=40.0,
            num_stops=10,
            route_difficulty_score=2.0,
            estimated_time_minutes=120,
        )
        result = calculate_workload(route)
        assert result > 0
    
    def test_workload_formula(self):
        """Verify workload formula calculation."""
        route = {
            "num_packages": 10,
            "total_weight_kg": 20.0,
            "route_difficulty_score": 1.0,
            "estimated_time_minutes": 60,
        }
        # Default weights: a=1.0, b=0.5, c=10.0, d=0.2
        # Expected: 1.0*10 + 0.5*20 + 10.0*1 + 0.2*60 = 10 + 10 + 10 + 12 = 42
        result = calculate_workload(route)
        assert abs(result - 42.0) < 0.1
    
    def test_workload_custom_weights(self):
        """Test with custom weights."""
        route = {
            "num_packages": 10,
            "total_weight_kg": 10.0,
            "route_difficulty_score": 1.0,
            "estimated_time_minutes": 60,
        }
        custom_weights = {"a": 2.0, "b": 1.0, "c": 5.0, "d": 0.5}
        # Expected: 2.0*10 + 1.0*10 + 5.0*1 + 0.5*60 = 20 + 10 + 5 + 30 = 65
        result = calculate_workload(route, custom_weights)
        assert abs(result - 65.0) < 0.1
    
    def test_workload_empty_route(self):
        """Empty route should have minimal workload."""
        route = {
            "num_packages": 0,
            "total_weight_kg": 0.0,
            "route_difficulty_score": 0.0,
            "estimated_time_minutes": 0,
        }
        result = calculate_workload(route)
        assert result == 0.0
    
    def test_workload_heavy_route(self):
        """Heavy route should have high workload."""
        route = {
            "num_packages": 50,
            "total_weight_kg": 100.0,
            "route_difficulty_score": 4.0,
            "estimated_time_minutes": 300,
        }
        result = calculate_workload(route)
        # Should be significant
        assert result > 100
    
    def test_workload_comparison(self):
        """Harder routes should have higher workload."""
        easy_route = {
            "num_packages": 10,
            "total_weight_kg": 15.0,
            "route_difficulty_score": 1.0,
            "estimated_time_minutes": 60,
        }
        hard_route = {
            "num_packages": 30,
            "total_weight_kg": 60.0,
            "route_difficulty_score": 3.5,
            "estimated_time_minutes": 180,
        }
        easy_workload = calculate_workload(easy_route)
        hard_workload = calculate_workload(hard_route)
        assert hard_workload > easy_workload
