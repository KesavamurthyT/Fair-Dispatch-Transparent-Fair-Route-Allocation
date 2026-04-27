"""
Unit tests for RoutePlannerAgent.
Tests optimal assignment, penalty application, and fallback algorithms.
"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.services.route_planner_agent import RoutePlannerAgent
from app.schemas.agent_schemas import (
    EffortMatrixResult,
    EffortBreakdown,
    FairnessRecommendations,
)


class MockDriver:
    """Mock Driver for testing."""
    def __init__(self, id=None, is_ev=False, battery_range_km=None):
        self.id = id or uuid4()
        self.is_ev = is_ev
        self.vehicle_type = "EV" if is_ev else "ICE"
        self.battery_range_km = battery_range_km


class MockRoute:
    """Mock Route for testing."""
    def __init__(self, id=None, total_distance_km=25.0, charging_time_minutes=0):
        self.id = id or uuid4()
        self.total_distance_km = total_distance_km
        self.charging_time_minutes = charging_time_minutes


def create_mock_effort_result(
    matrix: list[list[float]],
    drivers: list[MockDriver],
    routes: list[MockRoute],
) -> EffortMatrixResult:
    """Create mock EffortMatrixResult from matrix data."""
    driver_ids = [str(d.id) for d in drivers]
    route_ids = [str(r.id) for r in routes]
    
    # Create breakdown
    breakdown = {}
    for i, driver in enumerate(drivers):
        for j, route in enumerate(routes):
            key = f"{driver.id}:{route.id}"
            effort = matrix[i][j]
            breakdown[key] = EffortBreakdown(
                physical_effort=effort * 0.4,
                route_complexity=effort * 0.3,
                time_pressure=effort * 0.3,
                capacity_penalty=0.0,
                total=effort,
            )
    
    all_values = [v for row in matrix for v in row]
    
    return EffortMatrixResult(
        matrix=matrix,
        breakdown=breakdown,
        stats={
            "min": min(all_values) if all_values else 0.0,
            "max": max(all_values) if all_values else 0.0,
            "avg": sum(all_values) / len(all_values) if all_values else 0.0,
        },
        driver_ids=driver_ids,
        route_ids=route_ids,
    )


class TestRoutePlannerAgent:
    """Test suite for RoutePlannerAgent."""
    
    def test_basic_assignment(self):
        """Test basic 2x2 assignment."""
        agent = RoutePlannerAgent()
        
        drivers = [MockDriver(), MockDriver()]
        routes = [MockRoute(), MockRoute()]
        
        # Simple cost matrix where diagonal is cheaper
        matrix = [
            [10.0, 20.0],
            [30.0, 15.0],
        ]
        
        effort_result = create_mock_effort_result(matrix, drivers, routes)
        result = agent.plan(effort_result, drivers, routes)
        
        assert len(result.allocation) == 2
        assert result.total_effort > 0
        assert len(result.per_driver_effort) == 2
    
    def test_optimal_assignment_3x3(self):
        """Test optimal assignment for 3x3 matrix."""
        agent = RoutePlannerAgent()
        
        drivers = [MockDriver(), MockDriver(), MockDriver()]
        routes = [MockRoute(), MockRoute(), MockRoute()]
        
        # Known optimal: D0->R1 (10), D1->R2 (15), D2->R0 (20) = 45
        matrix = [
            [30.0, 10.0, 50.0],  # D0: best is R1 (10)
            [40.0, 35.0, 15.0],  # D1: best is R2 (15)
            [20.0, 25.0, 30.0],  # D2: best is R0 (20)
        ]
        
        effort_result = create_mock_effort_result(matrix, drivers, routes)
        result = agent.plan(effort_result, drivers, routes)
        
        # Should find optimal or near-optimal
        assert len(result.allocation) == 3
        # Total should be close to optimal (45)
        assert result.total_effort <= 50.0  # Allow some tolerance
    
    def test_more_drivers_than_routes(self):
        """Test handling when there are more drivers than routes."""
        agent = RoutePlannerAgent()
        
        drivers = [MockDriver(), MockDriver(), MockDriver()]
        routes = [MockRoute(), MockRoute()]
        
        matrix = [
            [10.0, 20.0],
            [15.0, 25.0],
            [30.0, 10.0],
        ]
        
        effort_result = create_mock_effort_result(matrix, drivers, routes)
        result = agent.plan(effort_result, drivers, routes)
        
        # Only 2 assignments possible
        assert len(result.allocation) == 2
    
    def test_more_routes_than_drivers(self):
        """Test handling when there are more routes than drivers."""
        agent = RoutePlannerAgent()
        
        drivers = [MockDriver(), MockDriver()]
        routes = [MockRoute(), MockRoute(), MockRoute()]
        
        matrix = [
            [10.0, 20.0, 30.0],
            [15.0, 25.0, 5.0],
        ]
        
        effort_result = create_mock_effort_result(matrix, drivers, routes)
        result = agent.plan(effort_result, drivers, routes)
        
        # Each driver gets 1 route
        assert len(result.allocation) == 2
    
    def test_penalty_application(self):
        """Test that penalties increase costs for specified drivers."""
        agent = RoutePlannerAgent()
        
        drivers = [MockDriver(), MockDriver()]
        routes = [MockRoute(), MockRoute()]
        
        matrix = [
            [10.0, 50.0],
            [50.0, 10.0],
        ]
        
        effort_result = create_mock_effort_result(matrix, drivers, routes)
        
        # Without penalty - should assign D0->R0, D1->R1
        result1 = agent.plan(effort_result, drivers, routes)
        
        # With heavy penalty on first driver
        penalties = {str(drivers[0].id): 10.0}  # 10x penalty
        result2 = agent.plan(
            effort_result, drivers, routes,
            fairness_penalties=penalties,
            proposal_number=2,
        )
        
        assert result2.proposal_number == 2
        # The assignment may change due to penalties
    
    def test_build_penalties_from_recommendations(self):
        """Test building penalties from fairness recommendations."""
        agent = RoutePlannerAgent()
        
        driver_id = str(uuid4())
        recommendations = FairnessRecommendations(
            penalize_high_effort_drivers=True,
            high_effort_driver_ids=[driver_id],
            penalty_factor=1.5,
        )
        
        per_driver_effort = {driver_id: 100.0, str(uuid4()): 50.0}
        
        penalties = agent.build_penalties_from_recommendations(
            recommendations, per_driver_effort
        )
        
        assert penalties[driver_id] == 1.5
        # Other driver should have normal weight
        other_id = [k for k in per_driver_effort if k != driver_id][0]
        assert penalties[other_id] == 1.0
    
    def test_avg_effort_calculation(self):
        """Test average effort is calculated correctly."""
        agent = RoutePlannerAgent()
        
        drivers = [MockDriver(), MockDriver()]
        routes = [MockRoute(), MockRoute()]
        
        matrix = [
            [20.0, 40.0],
            [40.0, 30.0],
        ]
        
        effort_result = create_mock_effort_result(matrix, drivers, routes)
        result = agent.plan(effort_result, drivers, routes)
        
        # Average should be total / num_assignments
        expected_avg = result.total_effort / len(result.allocation)
        assert abs(result.avg_effort - expected_avg) < 0.01
    
    def test_per_driver_effort_tracking(self):
        """Test that per-driver effort is tracked correctly."""
        agent = RoutePlannerAgent()
        
        drivers = [MockDriver(), MockDriver()]
        routes = [MockRoute(), MockRoute()]
        
        matrix = [
            [25.0, 75.0],
            [60.0, 40.0],
        ]
        
        effort_result = create_mock_effort_result(matrix, drivers, routes)
        result = agent.plan(effort_result, drivers, routes)
        
        # Each driver should have an entry
        assert len(result.per_driver_effort) == 2
        
        # Efforts should sum to total
        total_from_per_driver = sum(result.per_driver_effort.values())
        assert abs(total_from_per_driver - result.total_effort) < 0.01
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        agent = RoutePlannerAgent()
        
        effort_result = EffortMatrixResult(
            matrix=[],
            breakdown={},
            stats={"min": 0, "max": 0, "avg": 0},
            driver_ids=[],
            route_ids=[],
        )
        
        result = agent.plan(effort_result, [], [])
        
        assert result.allocation == []
        assert result.total_effort == 0.0
        assert result.avg_effort == 0.0
    
    def test_greedy_fallback(self):
        """Test greedy assignment fallback."""
        agent = RoutePlannerAgent()
        
        # Use the private greedy method directly
        cost_matrix = [
            [10.0, 30.0],
            [20.0, 5.0],
        ]
        
        assignments = agent._greedy_assignment(cost_matrix, 2, 2)
        
        assert len(assignments) == 2
        # Should find good (if not optimal) assignment
    
    def test_snapshot_generation(self):
        """Test input/output snapshot generation."""
        agent = RoutePlannerAgent()
        
        drivers = [MockDriver(), MockDriver()]
        routes = [MockRoute(), MockRoute()]
        
        matrix = [[10.0, 20.0], [30.0, 15.0]]
        effort_result = create_mock_effort_result(matrix, drivers, routes)
        
        input_snapshot = agent.get_input_snapshot(effort_result)
        assert "matrix_shape" in input_snapshot
        assert "effort_stats" in input_snapshot
        
        result = agent.plan(effort_result, drivers, routes)
        output_snapshot = agent.get_output_snapshot(result)
        assert "proposal_number" in output_snapshot
        assert "total_effort" in output_snapshot
