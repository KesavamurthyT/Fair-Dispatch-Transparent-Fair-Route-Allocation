"""
Unit tests for allocation logic.
Tests route clustering and driver-route assignment.
"""

import pytest
from app.services.clustering import cluster_packages, order_stops_by_nearest_neighbor, haversine_distance
from app.services.allocation import allocate_routes, build_cost_matrix, greedy_allocate


class TestClustering:
    """Tests for package clustering."""
    
    def test_cluster_single_package(self):
        """Single package should create single cluster."""
        packages = [
            {"latitude": 12.97, "longitude": 77.59, "weight_kg": 2.0, "address": "Addr 1"}
        ]
        result = cluster_packages(packages, num_drivers=3)
        assert len(result) == 1
        assert result[0].num_packages == 1
    
    def test_cluster_multiple_packages(self):
        """Multiple packages should create appropriate clusters."""
        packages = [
            {"latitude": 12.97, "longitude": 77.59, "weight_kg": 2.0, "address": "Addr 1"},
            {"latitude": 12.98, "longitude": 77.60, "weight_kg": 3.0, "address": "Addr 2"},
            {"latitude": 12.99, "longitude": 77.61, "weight_kg": 1.5, "address": "Addr 3"},
            {"latitude": 13.00, "longitude": 77.62, "weight_kg": 2.5, "address": "Addr 4"},
        ]
        result = cluster_packages(packages, num_drivers=2, target_per_route=2)
        
        # Should create 2 clusters with 2 packages each
        assert len(result) == 2
        total_packages = sum(c.num_packages for c in result)
        assert total_packages == 4
    
    def test_cluster_weight_calculation(self):
        """Cluster should correctly sum package weights."""
        packages = [
            {"latitude": 12.97, "longitude": 77.59, "weight_kg": 2.0, "address": "Addr 1"},
            {"latitude": 12.97, "longitude": 77.59, "weight_kg": 3.0, "address": "Addr 2"},
        ]
        result = cluster_packages(packages, num_drivers=1, target_per_route=10)
        
        assert len(result) == 1
        assert result[0].total_weight_kg == 5.0
    
    def test_cluster_unique_stops(self):
        """Cluster should count unique addresses as stops."""
        packages = [
            {"latitude": 12.97, "longitude": 77.59, "weight_kg": 2.0, "address": "Same Address"},
            {"latitude": 12.97, "longitude": 77.59, "weight_kg": 3.0, "address": "Same Address"},
            {"latitude": 12.98, "longitude": 77.60, "weight_kg": 1.0, "address": "Different Address"},
        ]
        result = cluster_packages(packages, num_drivers=1, target_per_route=10)
        
        assert len(result) == 1
        assert result[0].num_packages == 3
        assert result[0].num_stops == 2  # Only 2 unique addresses
    
    def test_cluster_empty_packages(self):
        """Empty package list should return empty clusters."""
        result = cluster_packages([], num_drivers=3)
        assert len(result) == 0
    
    def test_cluster_more_drivers_than_packages(self):
        """Should create at most num_packages clusters."""
        packages = [
            {"latitude": 12.97, "longitude": 77.59, "weight_kg": 2.0, "address": "Addr 1"},
            {"latitude": 12.98, "longitude": 77.60, "weight_kg": 3.0, "address": "Addr 2"},
        ]
        result = cluster_packages(packages, num_drivers=5, target_per_route=1)
        
        assert len(result) <= 2


class TestStopOrdering:
    """Tests for stop ordering using nearest neighbor."""
    
    def test_order_single_package(self):
        """Single package should remain unchanged."""
        packages = [
            {"latitude": 12.97, "longitude": 77.59, "address": "Addr 1"}
        ]
        result = order_stops_by_nearest_neighbor(packages, 12.90, 77.50)
        assert len(result) == 1
        assert result[0]["address"] == "Addr 1"
    
    def test_order_nearest_first(self):
        """Should visit nearest package first."""
        packages = [
            {"latitude": 13.00, "longitude": 77.60, "address": "Far"},
            {"latitude": 12.91, "longitude": 77.51, "address": "Near"},
        ]
        result = order_stops_by_nearest_neighbor(packages, 12.90, 77.50)
        
        # Near should be first
        assert result[0]["address"] == "Near"
        assert result[1]["address"] == "Far"
    
    def test_order_empty_list(self):
        """Empty list should return empty."""
        result = order_stops_by_nearest_neighbor([], 12.90, 77.50)
        assert len(result) == 0


class TestHaversineDistance:
    """Tests for haversine distance calculation."""
    
    def test_distance_same_point(self):
        """Same point should have zero distance."""
        result = haversine_distance(12.97, 77.59, 12.97, 77.59)
        assert result == 0.0
    
    def test_distance_known_locations(self):
        """Test with known approximate distance."""
        # Bangalore to Chennai is approximately 350 km
        result = haversine_distance(12.9716, 77.5946, 13.0827, 80.2707)
        assert 280 < result < 320  # Approximate range
    
    def test_distance_symmetric(self):
        """Distance should be symmetric."""
        dist1 = haversine_distance(12.97, 77.59, 13.00, 77.60)
        dist2 = haversine_distance(13.00, 77.60, 12.97, 77.59)
        assert abs(dist1 - dist2) < 0.001


class TestAllocation:
    """Tests for driver-route allocation."""
    
    def test_allocate_equal_drivers_routes(self):
        """Equal drivers and routes should match 1:1."""
        drivers = [
            {"external_id": "d1", "vehicle_capacity_kg": 100},
            {"external_id": "d2", "vehicle_capacity_kg": 100},
            {"external_id": "d3", "vehicle_capacity_kg": 100},
        ]
        routes = [
            {"workload_score": 50.0, "total_weight_kg": 30.0},
            {"workload_score": 60.0, "total_weight_kg": 40.0},
            {"workload_score": 55.0, "total_weight_kg": 35.0},
        ]
        
        result = allocate_routes(drivers, routes)
        
        assert len(result) == 3
        driver_indices = {r.driver_index for r in result}
        route_indices = {r.route_index for r in result}
        assert driver_indices == {0, 1, 2}
        assert route_indices == {0, 1, 2}
    
    def test_allocate_more_drivers(self):
        """More drivers than routes should leave some unassigned."""
        drivers = [
            {"external_id": "d1", "vehicle_capacity_kg": 100},
            {"external_id": "d2", "vehicle_capacity_kg": 100},
            {"external_id": "d3", "vehicle_capacity_kg": 100},
        ]
        routes = [
            {"workload_score": 50.0, "total_weight_kg": 30.0},
            {"workload_score": 60.0, "total_weight_kg": 40.0},
        ]
        
        result = allocate_routes(drivers, routes)
        
        assert len(result) == 2  # Only 2 routes assigned
    
    def test_allocate_more_routes(self):
        """More routes than drivers should leave some routes unassigned."""
        drivers = [
            {"external_id": "d1", "vehicle_capacity_kg": 100},
        ]
        routes = [
            {"workload_score": 50.0, "total_weight_kg": 30.0},
            {"workload_score": 60.0, "total_weight_kg": 40.0},
            {"workload_score": 55.0, "total_weight_kg": 35.0},
        ]
        
        result = allocate_routes(drivers, routes)
        
        assert len(result) == 1  # Only 1 driver available
    
    def test_allocate_empty_inputs(self):
        """Empty inputs should return empty results."""
        assert allocate_routes([], []) == []
        assert allocate_routes([{"external_id": "d1", "vehicle_capacity_kg": 100}], []) == []
        assert allocate_routes([], [{"workload_score": 50.0, "total_weight_kg": 30.0}]) == []


class TestCostMatrix:
    """Tests for cost matrix building."""
    
    def test_cost_matrix_shape(self):
        """Cost matrix should have correct shape."""
        drivers = [
            {"external_id": "d1", "vehicle_capacity_kg": 100},
            {"external_id": "d2", "vehicle_capacity_kg": 100},
        ]
        routes = [
            {"workload_score": 50.0, "total_weight_kg": 30.0},
            {"workload_score": 60.0, "total_weight_kg": 40.0},
            {"workload_score": 55.0, "total_weight_kg": 35.0},
        ]
        
        result = build_cost_matrix(drivers, routes)
        
        assert result.shape == (2, 3)
    
    def test_cost_matrix_values(self):
        """Cost should be based on workload score."""
        drivers = [{"external_id": "d1", "vehicle_capacity_kg": 100}]
        routes = [{"workload_score": 42.0, "total_weight_kg": 30.0}]
        
        result = build_cost_matrix(drivers, routes)
        
        assert result[0, 0] == 42.0
    
    def test_cost_matrix_capacity_penalty(self):
        """Over-capacity should add penalty."""
        drivers = [{"external_id": "d1", "vehicle_capacity_kg": 20}]  # Low capacity
        routes = [{"workload_score": 50.0, "total_weight_kg": 50.0}]  # Heavy route
        
        result = build_cost_matrix(drivers, routes)
        
        # Should be penalized: 50 + (50-20)*10 = 50 + 300 = 350
        assert result[0, 0] > 50.0


class TestGreedyAllocate:
    """Tests for greedy allocation fallback."""
    
    def test_greedy_basic(self):
        """Greedy should match in order."""
        drivers = [
            {"external_id": "d1"},
            {"external_id": "d2"},
        ]
        routes = [
            {"workload_score": 50.0},
            {"workload_score": 60.0},
        ]
        
        result = greedy_allocate(drivers, routes)
        
        assert len(result) == 2
        assert result[0].driver_index == 0
        assert result[0].route_index == 0
        assert result[1].driver_index == 1
        assert result[1].route_index == 1
