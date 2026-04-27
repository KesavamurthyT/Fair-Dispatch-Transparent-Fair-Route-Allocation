"""
Unit tests for Phase 7: EV-aware routing and Recovery Mode.
Tests EV feasibility checks, charging overhead, recovery service, and agent integration.
"""

import pytest
from datetime import date, timedelta
from uuid import uuid4

from app.services.ev_utils import (
    is_route_feasible_for_ev,
    calculate_ev_charging_overhead,
    get_ev_effort_adjustment,
)
from app.services.recovery_service import calculate_recovery_penalty


# ==================== EV UTILITIES TESTS ====================

class TestIsRouteFeasibleForEV:
    """Tests for is_route_feasible_for_ev function."""
    
    def test_feasible_route_within_range(self):
        """Route within battery range is feasible."""
        result = is_route_feasible_for_ev(
            battery_range_km=100.0,
            route_distance_km=50.0,
            safety_margin_pct=10.0,
        )
        assert result is True
    
    def test_infeasible_route_exceeds_range(self):
        """Route exceeding battery range with margin is infeasible."""
        result = is_route_feasible_for_ev(
            battery_range_km=100.0,
            route_distance_km=95.0,  # > 90 (100 - 10% margin)
            safety_margin_pct=10.0,
        )
        assert result is False
    
    def test_exactly_at_effective_range(self):
        """Route exactly at effective range is feasible."""
        result = is_route_feasible_for_ev(
            battery_range_km=100.0,
            route_distance_km=90.0,  # Exactly at margin
            safety_margin_pct=10.0,
        )
        assert result is True
    
    def test_zero_battery_range_returns_false(self):
        """Zero battery range returns infeasible."""
        result = is_route_feasible_for_ev(
            battery_range_km=0.0,
            route_distance_km=50.0,
        )
        assert result is False
    
    def test_none_distance_returns_true(self):
        """No distance info assumes feasible."""
        result = is_route_feasible_for_ev(
            battery_range_km=100.0,
            route_distance_km=None,
        )
        assert result is True
    
    def test_custom_safety_margin(self):
        """Custom safety margin affects feasibility."""
        # With 20% margin, effective range = 80
        result = is_route_feasible_for_ev(
            battery_range_km=100.0,
            route_distance_km=85.0,
            safety_margin_pct=20.0,
        )
        assert result is False


class TestCalculateEVChargingOverhead:
    """Tests for calculate_ev_charging_overhead function."""
    
    def test_no_overhead_below_70_usage(self):
        """No overhead when usage is below 70%."""
        overhead = calculate_ev_charging_overhead(
            route_distance_km=60.0,
            battery_range_km=100.0,
            charging_time_minutes=30,
        )
        assert overhead == 0.0
    
    def test_overhead_at_80_percent_usage(self):
        """Overhead applied at 80% battery usage."""
        overhead = calculate_ev_charging_overhead(
            route_distance_km=80.0,
            battery_range_km=100.0,
            charging_time_minutes=30,
            penalty_weight=0.3,
        )
        # (0.8 - 0.7) * 30 * 0.3 = 0.9
        assert overhead == pytest.approx(0.9, abs=0.01)
    
    def test_overhead_at_100_percent_usage(self):
        """Overhead at 100% usage."""
        overhead = calculate_ev_charging_overhead(
            route_distance_km=100.0,
            battery_range_km=100.0,
            charging_time_minutes=30,
            penalty_weight=0.3,
        )
        # (1.0 - 0.7) * 30 * 0.3 = 2.7
        assert overhead == pytest.approx(2.7, abs=0.01)
    
    def test_zero_battery_range_no_overhead(self):
        """Zero battery range returns no overhead."""
        overhead = calculate_ev_charging_overhead(
            route_distance_km=50.0,
            battery_range_km=0.0,
            charging_time_minutes=30,
        )
        assert overhead == 0.0


class TestGetEVEffortAdjustment:
    """Tests for get_ev_effort_adjustment function."""
    
    def test_non_ev_driver_returns_feasible_no_overhead(self):
        """Non-EV driver is always feasible with no overhead."""
        feasible, overhead = get_ev_effort_adjustment(
            driver_is_ev=False,
            battery_range_km=None,
            charging_time_minutes=None,
            route_distance_km=100.0,
        )
        assert feasible is True
        assert overhead == 0.0
    
    def test_ev_driver_feasible_route(self):
        """EV driver with feasible route returns proper overhead."""
        feasible, overhead = get_ev_effort_adjustment(
            driver_is_ev=True,
            battery_range_km=100.0,
            charging_time_minutes=30,
            route_distance_km=50.0,
        )
        assert feasible is True
        assert overhead == 0.0  # Below 70% usage
    
    def test_ev_driver_infeasible_route(self):
        """EV driver with infeasible route returns infinity overhead."""
        feasible, overhead = get_ev_effort_adjustment(
            driver_is_ev=True,
            battery_range_km=100.0,
            charging_time_minutes=30,
            route_distance_km=95.0,  # Exceeds 90% margin
        )
        assert feasible is False
        assert overhead == float('inf')
    
    def test_ev_driver_no_battery_range_info(self):
        """EV driver without battery info assumes feasible."""
        feasible, overhead = get_ev_effort_adjustment(
            driver_is_ev=True,
            battery_range_km=None,
            charging_time_minutes=30,
            route_distance_km=100.0,
        )
        assert feasible is True
        assert overhead == 0.0


# ==================== RECOVERY SERVICE TESTS ====================

class TestCalculateRecoveryPenalty:
    """Tests for calculate_recovery_penalty function."""
    
    def test_no_penalty_when_no_recovery_target(self):
        """No penalty when not in recovery mode."""
        penalty = calculate_recovery_penalty(
            effort=100.0,
            recovery_target=None,
            penalty_weight=3.0,
        )
        assert penalty == 0.0
    
    def test_no_penalty_effort_below_target(self):
        """No penalty when effort is below recovery target."""
        penalty = calculate_recovery_penalty(
            effort=50.0,
            recovery_target=70.0,
            penalty_weight=3.0,
        )
        assert penalty == 0.0
    
    def test_penalty_effort_exceeds_target(self):
        """Penalty applied when effort exceeds recovery target."""
        penalty = calculate_recovery_penalty(
            effort=80.0,
            recovery_target=70.0,
            penalty_weight=3.0,
        )
        # (80 - 70) * 3.0 = 30
        assert penalty == 30.0
    
    def test_penalty_exactly_at_target(self):
        """No penalty when effort equals recovery target."""
        penalty = calculate_recovery_penalty(
            effort=70.0,
            recovery_target=70.0,
            penalty_weight=3.0,
        )
        assert penalty == 0.0


# ==================== INTEGRATION TESTS ====================

class TestEVIntegrationWithMLEffortAgent:
    """Integration tests for EV features in MLEffortAgent."""
    
    def test_ev_driver_feasibility_in_matrix(self):
        """Verify EV feasibility is checked during matrix computation."""
        from app.services.ml_effort_agent import MLEffortAgent
        from unittest.mock import MagicMock
        from app.models.driver import VehicleType
        
        # Create mock EV driver
        ev_driver = MagicMock()
        ev_driver.id = uuid4()
        ev_driver.is_ev = True
        ev_driver.battery_range_km = 100.0
        ev_driver.charging_time_minutes = 30
        ev_driver.vehicle_capacity_kg = 100.0
        
        # Create mock ICE driver
        ice_driver = MagicMock()
        ice_driver.id = uuid4()
        ice_driver.is_ev = False
        ice_driver.battery_range_km = None
        ice_driver.charging_time_minutes = None
        ice_driver.vehicle_capacity_kg = 100.0
        
        # Create mock route - infeasible for EV
        far_route = MagicMock()
        far_route.id = uuid4()
        far_route.num_packages = 10
        far_route.total_weight_kg = 50.0
        far_route.num_stops = 5
        far_route.route_difficulty_score = 1.5
        far_route.estimated_time_minutes = 120
        far_route.total_distance_km = 95.0  # Infeasible for EV
        
        ml_agent = MLEffortAgent()
        result = ml_agent.compute_effort_matrix(
            drivers=[ev_driver, ice_driver],
            routes=[far_route],
            ev_config={"safety_margin_pct": 10.0, "charging_penalty_weight": 0.3},
        )
        
        # Check that EV driver-route is marked infeasible
        ev_key = f"{ev_driver.id}:{far_route.id}"
        ice_key = f"{ice_driver.id}:{far_route.id}"
        
        assert ev_key in result.infeasible_pairs
        assert ice_key not in result.infeasible_pairs
        
        # Check matrix has high cost for infeasible pair
        assert result.matrix[0][0] == 99999.0  # EV driver
        assert result.matrix[1][0] < 99999.0   # ICE driver


class TestRecoveryPenaltyInRoutePlanner:
    """Integration tests for recovery penalty in RoutePlannerAgent."""
    
    def test_recovery_penalty_affects_assignment(self):
        """Verify recovery penalty steers assignment towards lighter routes."""
        from app.services.route_planner_agent import RoutePlannerAgent
        from app.schemas.agent_schemas import EffortMatrixResult, EffortBreakdown
        from unittest.mock import MagicMock
        
        # Create drivers
        recovery_driver = MagicMock()
        recovery_driver.id = uuid4()
        
        normal_driver = MagicMock()
        normal_driver.id = uuid4()
        
        # Create routes - one easy, one hard
        easy_route = MagicMock()
        easy_route.id = uuid4()
        
        hard_route = MagicMock()
        hard_route.id = uuid4()
        
        # Create effort matrix
        driver_ids = [str(recovery_driver.id), str(normal_driver.id)]
        route_ids = [str(easy_route.id), str(hard_route.id)]
        
        # Matrix: recovery_driver can do easy(50) or hard(80)
        #         normal_driver can do easy(50) or hard(80)
        matrix = [[50.0, 80.0], [50.0, 80.0]]
        
        effort_result = EffortMatrixResult(
            matrix=matrix,
            breakdown={
                f"{driver_ids[0]}:{route_ids[0]}": EffortBreakdown(
                    physical_effort=25, route_complexity=15, time_pressure=10, capacity_penalty=0, total=50
                ),
                f"{driver_ids[0]}:{route_ids[1]}": EffortBreakdown(
                    physical_effort=40, route_complexity=25, time_pressure=15, capacity_penalty=0, total=80
                ),
                f"{driver_ids[1]}:{route_ids[0]}": EffortBreakdown(
                    physical_effort=25, route_complexity=15, time_pressure=10, capacity_penalty=0, total=50
                ),
                f"{driver_ids[1]}:{route_ids[1]}": EffortBreakdown(
                    physical_effort=40, route_complexity=25, time_pressure=15, capacity_penalty=0, total=80
                ),
            },
            stats={"min": 50, "max": 80, "avg": 65, "num_cells": 4},
            driver_ids=driver_ids,
            route_ids=route_ids,
            infeasible_pairs=[],
        )
        
        planner = RoutePlannerAgent()
        
        # Without recovery, both drivers get arbitrary assignments
        result_no_recovery = planner.plan(
            effort_result=effort_result,
            drivers=[recovery_driver, normal_driver],
            routes=[easy_route, hard_route],
        )
        
        # With recovery, recovery driver should get easy route
        recovery_targets = {
            str(recovery_driver.id): 60.0,  # Target max 60
            str(normal_driver.id): None,    # No recovery
        }
        
        result_with_recovery = planner.plan(
            effort_result=effort_result,
            drivers=[recovery_driver, normal_driver],
            routes=[easy_route, hard_route],
            recovery_targets=recovery_targets,
            recovery_penalty_weight=10.0,  # High penalty
        )
        
        # Check recovery driver got the easy route
        for alloc in result_with_recovery.allocation:
            if alloc.driver_id == recovery_driver.id:
                assert alloc.effort == 50.0  # Got easy route
