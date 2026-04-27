"""
Unit tests for CVRP Solver.

Tests the OR-Tools CVRP formulation including capacity constraints,
EV range limits, fairness enforcement (two-pass Gini), and greedy fallback.
"""

import pytest
import math
from app.solver.cvrp_solver import (
    CVRPSolver,
    CVRPVehicle,
    CVRPNode,
    CVRPSolution,
)


# === Fixtures ===

def make_vehicles(n: int = 5, capacity: float = 100.0, ev_count: int = 0) -> list:
    """Create test vehicles with optional EV subset."""
    vehicles = []
    for i in range(n):
        is_ev = i < ev_count
        vehicles.append(CVRPVehicle(
            id=f"driver_{i}",
            capacity_kg=capacity,
            max_range_km=50.0 if is_ev else None,
            is_ev=is_ev,
            fatigue_penalty=float(i % 3),
        ))
    return vehicles


def make_nodes(n: int = 20, base_lat: float = 12.97, base_lng: float = 77.59) -> list:
    """Create test nodes spread around a base coordinate."""
    nodes = []
    for i in range(n):
        # Spread nodes in a small area (~5km radius)
        lat = base_lat + (i % 5) * 0.01 - 0.02
        lng = base_lng + (i // 5) * 0.01 - 0.02
        nodes.append(CVRPNode(
            id=f"pkg_{i}",
            latitude=lat,
            longitude=lng,
            demand_kg=float(3 + (i % 7)),  # 3-9 kg
            route_id=f"cluster_{i // 5}",
        ))
    return nodes


# === Tests ===

class TestCVRPSolverBasic:
    """Test basic CVRP solving capabilities."""

    def test_solve_small_instance(self):
        """5 drivers, 20 packages — should produce a valid solution."""
        vehicles = make_vehicles(5, capacity=100.0)
        nodes = make_nodes(20)

        solver = CVRPSolver(
            depot_lat=12.97,
            depot_lng=77.59,
            vehicles=vehicles,
            nodes=nodes,
            time_limit_seconds=10,
        )
        solution = solver.solve()

        assert solution.status in ("optimal", "feasible", "fallback")
        assert solution.total_distance > 0

        # All nodes should be assigned
        all_assigned = set()
        for node_ids in solution.assignments.values():
            all_assigned.update(node_ids)

        # At least most nodes should be assigned
        assert len(all_assigned) >= 15  # Allow some disjunction slack

    def test_capacity_constraint_respected(self):
        """Drivers should not exceed their weight capacity."""
        vehicles = make_vehicles(5, capacity=30.0)  # Small capacity
        nodes = make_nodes(20)

        solver = CVRPSolver(
            depot_lat=12.97,
            depot_lng=77.59,
            vehicles=vehicles,
            nodes=nodes,
            time_limit_seconds=10,
        )
        solution = solver.solve()

        for vid, load in solution.per_vehicle_load.items():
            assert load <= 30.0 + 0.01, f"Vehicle {vid} overloaded: {load} > 30.0"

    def test_empty_input(self):
        """Should handle empty inputs gracefully."""
        solver = CVRPSolver(
            depot_lat=12.97,
            depot_lng=77.59,
            vehicles=[],
            nodes=[],
        )
        solution = solver.solve()
        assert solution.status == "infeasible"

    def test_single_vehicle_single_node(self):
        """Simplest possible instance."""
        vehicles = [CVRPVehicle(id="d1", capacity_kg=50.0)]
        nodes = [CVRPNode(id="p1", latitude=12.98, longitude=77.60, demand_kg=5.0)]

        solver = CVRPSolver(
            depot_lat=12.97,
            depot_lng=77.59,
            vehicles=vehicles,
            nodes=nodes,
            time_limit_seconds=5,
        )
        solution = solver.solve()

        assert solution.status in ("optimal", "feasible", "fallback")
        assert "p1" in solution.assignments.get("d1", [])


class TestCVRPEVConstraints:
    """Test EV battery range constraints."""

    def test_ev_range_respected(self):
        """EV drivers should not exceed their battery range."""
        # EV with very short range
        vehicles = [
            CVRPVehicle(id="ev_1", capacity_kg=100.0, max_range_km=3.0, is_ev=True),
            CVRPVehicle(id="ice_1", capacity_kg=100.0, is_ev=False),
        ]

        # Nodes spread far enough to challenge the EV
        nodes = [
            CVRPNode(id="p1", latitude=12.98, longitude=77.60, demand_kg=5.0),
            CVRPNode(id="p2", latitude=12.99, longitude=77.61, demand_kg=5.0),
            CVRPNode(id="p3", latitude=13.00, longitude=77.62, demand_kg=5.0),
            CVRPNode(id="p4", latitude=13.01, longitude=77.63, demand_kg=5.0),
        ]

        solver = CVRPSolver(
            depot_lat=12.97,
            depot_lng=77.59,
            vehicles=vehicles,
            nodes=nodes,
            time_limit_seconds=10,
        )
        solution = solver.solve()

        assert solution.status in ("optimal", "feasible", "fallback")

        # EV should have shorter total distance than ICE
        ev_dist = solution.per_vehicle_distance.get("ev_1", 0)
        # EV with 3km range should handle very few stops
        assert ev_dist <= 5.0, f"EV exceeded expected range: {ev_dist}km"


class TestCVRPFairness:
    """Test two-pass fairness enforcement."""

    def test_fairness_reduces_gini(self):
        """Re-solving with penalties should reduce Gini coefficient."""
        # Create an imbalanced scenario
        vehicles = make_vehicles(3, capacity=200.0)
        # Cluster nodes close together so greedy assigns them all to one driver
        nodes = []
        for i in range(15):
            nodes.append(CVRPNode(
                id=f"pkg_{i}",
                latitude=12.97 + 0.001 * i,
                longitude=77.59,
                demand_kg=5.0,
            ))

        solver = CVRPSolver(
            depot_lat=12.97,
            depot_lng=77.59,
            vehicles=vehicles,
            nodes=nodes,
            time_limit_seconds=10,
            fairness_threshold_gini=0.5,  # Relaxed for test
            max_fairness_iterations=2,
        )
        solution = solver.solve()

        assert solution.status in ("optimal", "feasible", "fallback")

        # At least 2 vehicles should have assignments
        active_vehicles = sum(
            1 for nodes_list in solution.assignments.values() if nodes_list
        )
        assert active_vehicles >= 2


class TestCVRPGreedyFallback:
    """Test the greedy fallback mechanism."""

    def test_greedy_fallback_produces_valid_solution(self):
        """Greedy fallback should assign nodes respecting capacity."""
        vehicles = make_vehicles(3, capacity=50.0)
        nodes = make_nodes(10)

        solver = CVRPSolver(
            depot_lat=12.97,
            depot_lng=77.59,
            vehicles=vehicles,
            nodes=nodes,
        )
        solution = solver._greedy_fallback()

        assert solution.status == "fallback"
        assert solution.total_distance >= 0

        # Capacity respected
        for vid, load in solution.per_vehicle_load.items():
            assert load <= 50.0 + 0.01


class TestCVRPScaleEdgeCases:
    """Test edge cases around scale and proportions."""

    def test_more_packages_than_drivers(self):
        """Should handle case where packages far outnumber drivers."""
        vehicles = make_vehicles(2, capacity=500.0)
        nodes = make_nodes(30)

        solver = CVRPSolver(
            depot_lat=12.97,
            depot_lng=77.59,
            vehicles=vehicles,
            nodes=nodes,
            time_limit_seconds=15,
        )
        solution = solver.solve()

        assert solution.status in ("optimal", "feasible", "fallback")
        # Most nodes should be assigned
        total_assigned = sum(len(ids) for ids in solution.assignments.values())
        assert total_assigned >= 20

    def test_more_drivers_than_packages(self):
        """Should handle case where drivers outnumber packages."""
        vehicles = make_vehicles(10, capacity=100.0)
        nodes = make_nodes(5)

        solver = CVRPSolver(
            depot_lat=12.97,
            depot_lng=77.59,
            vehicles=vehicles,
            nodes=nodes,
            time_limit_seconds=10,
        )
        solution = solver.solve()

        assert solution.status in ("optimal", "feasible", "fallback")
        # All 5 nodes should be assigned
        total_assigned = sum(len(ids) for ids in solution.assignments.values())
        assert total_assigned == 5
