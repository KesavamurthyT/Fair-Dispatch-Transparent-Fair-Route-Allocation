"""
Unit tests for MLEffortAgent.
Tests effort matrix computation and breakdown calculation.
"""

import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from app.services.ml_effort_agent import MLEffortAgent
from app.schemas.agent_schemas import EffortWeights


class MockDriver:
    """Mock Driver for testing."""
    def __init__(self, id=None, vehicle_capacity_kg=100.0, is_ev=False, battery_range_km=None, charging_time_minutes=0):
        self.id = id or uuid4()
        self.vehicle_capacity_kg = vehicle_capacity_kg
        self.is_ev = is_ev
        self.vehicle_type = "EV" if is_ev else "ICE"
        self.battery_range_km = battery_range_km
        self.charging_time_minutes = charging_time_minutes


class MockRoute:
    """Mock Route for testing."""
    def __init__(
        self,
        id=None,
        num_packages=10,
        total_weight_kg=20.0,
        num_stops=5,
        route_difficulty_score=2.0,
        estimated_time_minutes=60,
        total_distance_km=25.0,
        charging_time_minutes=0,
    ):
        self.id = id or uuid4()
        self.num_packages = num_packages
        self.total_weight_kg = total_weight_kg
        self.num_stops = num_stops
        self.route_difficulty_score = route_difficulty_score
        self.estimated_time_minutes = estimated_time_minutes
        self.total_distance_km = total_distance_km
        self.charging_time_minutes = charging_time_minutes


class TestMLEffortAgent:
    """Test suite for MLEffortAgent."""
    
    def test_effort_matrix_shape(self):
        """Test that effort matrix has correct dimensions."""
        agent = MLEffortAgent()
        
        drivers = [MockDriver() for _ in range(3)]
        routes = [MockRoute() for _ in range(4)]
        
        result = agent.compute_effort_matrix(drivers, routes)
        
        assert len(result.matrix) == 3, "Should have 3 rows (drivers)"
        assert all(len(row) == 4 for row in result.matrix), "Each row should have 4 columns (routes)"
        assert len(result.driver_ids) == 3
        assert len(result.route_ids) == 4
    
    def test_effort_increases_with_packages(self):
        """Test that effort increases with more packages."""
        agent = MLEffortAgent()
        
        driver = MockDriver()
        route_light = MockRoute(num_packages=5, total_weight_kg=10.0)
        route_heavy = MockRoute(num_packages=20, total_weight_kg=40.0)
        
        result_light = agent.compute_effort_matrix([driver], [route_light])
        result_heavy = agent.compute_effort_matrix([driver], [route_heavy])
        
        effort_light = result_light.matrix[0][0]
        effort_heavy = result_heavy.matrix[0][0]
        
        assert effort_heavy > effort_light, "Heavier route should have more effort"
    
    def test_effort_increases_with_difficulty(self):
        """Test that effort increases with route difficulty."""
        agent = MLEffortAgent()
        
        driver = MockDriver()
        route_easy = MockRoute(route_difficulty_score=1.0)
        route_hard = MockRoute(route_difficulty_score=5.0)
        
        result_easy = agent.compute_effort_matrix([driver], [route_easy])
        result_hard = agent.compute_effort_matrix([driver], [route_hard])
        
        assert result_hard.matrix[0][0] > result_easy.matrix[0][0]
    
    def test_capacity_penalty_applied(self):
        """Test that overloaded routes get penalty."""
        agent = MLEffortAgent()
        
        # Driver with 50kg capacity
        driver = MockDriver(vehicle_capacity_kg=50.0)
        
        # Routes with different weights
        route_under = MockRoute(total_weight_kg=40.0)  # Under capacity
        route_over = MockRoute(total_weight_kg=70.0)   # Over capacity
        
        result_under = agent.compute_effort_matrix([driver], [route_under])
        result_over = agent.compute_effort_matrix([driver], [route_over])
        
        # Over-capacity should have significantly higher effort
        assert result_over.matrix[0][0] > result_under.matrix[0][0]
    
    def test_breakdown_components(self):
        """Test that breakdown contains all components."""
        agent = MLEffortAgent()
        
        driver = MockDriver()
        route = MockRoute()
        
        result = agent.compute_effort_matrix([driver], [route])
        
        key = f"{driver.id}:{route.id}"
        assert key in result.breakdown
        
        breakdown = result.breakdown[key]
        assert hasattr(breakdown, 'physical_effort')
        assert hasattr(breakdown, 'route_complexity')
        assert hasattr(breakdown, 'time_pressure')
        assert hasattr(breakdown, 'capacity_penalty')
        assert hasattr(breakdown, 'total')
        
        # Total should be sum of components
        expected_total = (
            breakdown.physical_effort +
            breakdown.route_complexity +
            breakdown.time_pressure +
            breakdown.capacity_penalty
        )
        assert abs(breakdown.total - expected_total) < 0.01
    
    def test_stats_computed(self):
        """Test that matrix stats are computed correctly."""
        agent = MLEffortAgent()
        
        drivers = [MockDriver() for _ in range(2)]
        routes = [MockRoute(num_packages=i*5 + 5) for i in range(3)]
        
        result = agent.compute_effort_matrix(drivers, routes)
        
        assert "min" in result.stats
        assert "max" in result.stats
        assert "avg" in result.stats
        assert result.stats["min"] <= result.stats["avg"] <= result.stats["max"]
    
    def test_custom_weights(self):
        """Test that custom weights affect effort calculation."""
        # Default weights
        agent_default = MLEffortAgent()
        
        # Custom weights with higher package weight
        custom_weights = EffortWeights(alpha_packages=5.0, beta_weight=0.1)
        agent_custom = MLEffortAgent(weights=custom_weights)
        
        driver = MockDriver()
        route = MockRoute(num_packages=20, total_weight_kg=10.0)
        
        result_default = agent_default.compute_effort_matrix([driver], [route])
        result_custom = agent_custom.compute_effort_matrix([driver], [route])
        
        # Custom should be different due to different weights
        assert result_default.matrix[0][0] != result_custom.matrix[0][0]
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        agent = MLEffortAgent()
        
        result = agent.compute_effort_matrix([], [])
        
        assert result.matrix == []
        assert result.stats["min"] == 0.0
        assert result.stats["max"] == 0.0
        assert result.stats["avg"] == 0.0
    
    def test_snapshot_generation(self):
        """Test input/output snapshot generation for logging."""
        agent = MLEffortAgent()
        
        drivers = [MockDriver() for _ in range(2)]
        routes = [MockRoute() for _ in range(3)]
        
        input_snapshot = agent.get_input_snapshot(drivers, routes)
        assert input_snapshot["num_drivers"] == 2
        assert input_snapshot["num_routes"] == 3
        
        result = agent.compute_effort_matrix(drivers, routes)
        output_snapshot = agent.get_output_snapshot(result)
        assert "matrix_shape" in output_snapshot
        assert "min_effort" in output_snapshot
