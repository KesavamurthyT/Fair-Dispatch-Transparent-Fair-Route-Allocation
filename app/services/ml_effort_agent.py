"""
ML Effort Agent - Builds effort matrix for driver-route pairs.
Phase 4.1 implementation with deterministic scoring.
Now enhanced with historical features for personalized effort calculation.
"""

import uuid
from typing import Dict, List, Optional

from app.models.driver import Driver
from app.models.route import Route
from app.schemas.agent_schemas import (
    EffortBreakdown,
    EffortMatrixResult,
    EffortWeights,
)
from app.services.history_features import (
    DriverHistoryFeatures,
    HistoryConfig,
    compute_history_effort_adjustment,
)


class MLEffortAgent:
    """
    ML Effort Agent computes an effort matrix for all driver-route combinations.
    
    In Phase 4.1, uses a deterministic formula. Future phases can swap in
    an ML model for prediction.
    
    Now includes historical context:
    - recent_avg_effort and recent_hard_days affect perceived effort
    - avg_stress_level and avg_fairness_rating from feedback influence scoring
    """
    
    def __init__(
        self,
        weights: Optional[EffortWeights] = None,
        history_config: Optional[HistoryConfig] = None,
    ):
        """
        Initialize agent with configurable weights.
        
        Args:
            weights: Custom effort weights. Uses defaults if not provided.
            history_config: Configuration for historical adjustments.
        """
        self.weights = weights or EffortWeights()
        self.history_config = history_config or HistoryConfig()
    
    def compute_effort_matrix(
        self,
        drivers: List[Driver],
        routes: List[Route],
        driver_stats: Optional[Dict[str, dict]] = None,
        ev_config: Optional[dict] = None,
        driver_history: Optional[Dict[str, DriverHistoryFeatures]] = None,
    ) -> EffortMatrixResult:
        """
        Compute effort matrix for all driver-route pairs.
        
        Args:
            drivers: List of Driver objects
            routes: List of Route objects
            driver_stats: Optional dict of driver_id -> {recent_avg_workload, fatigue_level, etc}
            ev_config: Optional EV config {safety_margin_pct, charging_penalty_weight}
            driver_history: Optional dict of driver_id -> DriverHistoryFeatures for historical adjustments
            
        Returns:
            EffortMatrixResult with matrix, breakdowns, statistics, and infeasible pairs
        """
        from app.services.ev_utils import get_ev_effort_adjustment
        
        driver_stats = driver_stats or {}
        driver_history = driver_history or {}
        ev_config = ev_config or {}
        safety_margin = ev_config.get("safety_margin_pct", 10.0)
        charging_weight = ev_config.get("charging_penalty_weight", 0.3)
        
        # Build ordered ID lists
        driver_ids = [str(d.id) for d in drivers]
        route_ids = [str(r.id) for r in routes]
        
        # Initialize matrix and breakdown storage
        matrix: List[List[float]] = []
        breakdown: Dict[str, EffortBreakdown] = {}
        all_efforts: List[float] = []
        infeasible_pairs: List[str] = []
        
        for driver in drivers:
            row: List[float] = []
            driver_id_str = str(driver.id)
            
            # Get driver stats if available
            stats = driver_stats.get(driver_id_str, {})
            
            # Get driver history for personalized adjustments
            history = driver_history.get(driver_id_str)
            history_adjustment = compute_history_effort_adjustment(history, self.history_config)
            
            for route in routes:
                route_id_str = str(route.id)
                key = f"{driver_id_str}:{route_id_str}"
                
                # Compute base effort breakdown
                effort_breakdown = self._compute_effort_breakdown(
                    driver=driver,
                    route=route,
                    driver_stats=stats,
                )
                
                # Check EV feasibility and add charging overhead
                ev_feasible, ev_overhead = get_ev_effort_adjustment(
                    driver_is_ev=driver.is_ev,
                    battery_range_km=driver.battery_range_km,
                    charging_time_minutes=driver.charging_time_minutes,
                    route_distance_km=route.total_distance_km,
                    safety_margin_pct=safety_margin,
                    penalty_weight=charging_weight,
                )
                
                if not ev_feasible:
                    # Mark as infeasible - use very high cost
                    infeasible_pairs.append(key)
                    final_effort = 99999.0
                else:
                    # Add EV charging overhead and history adjustment to effort
                    final_effort = effort_breakdown.total + ev_overhead + history_adjustment
                
                # Update breakdown with adjusted total
                breakdown[key] = EffortBreakdown(
                    physical_effort=effort_breakdown.physical_effort,
                    route_complexity=effort_breakdown.route_complexity,
                    time_pressure=effort_breakdown.time_pressure,
                    capacity_penalty=effort_breakdown.capacity_penalty + ev_overhead + history_adjustment,
                    total=round(max(0, final_effort), 2),
                )
                
                row.append(round(max(0, final_effort), 2))
                if ev_feasible:
                    all_efforts.append(final_effort)
            
            matrix.append(row)
        
        # Compute statistics (excluding infeasible)
        stats_dict = {
            "min": min(all_efforts) if all_efforts else 0.0,
            "max": max(all_efforts) if all_efforts else 0.0,
            "avg": sum(all_efforts) / len(all_efforts) if all_efforts else 0.0,
            "num_cells": len(all_efforts),
            "num_infeasible": len(infeasible_pairs),
        }
        
        return EffortMatrixResult(
            matrix=matrix,
            breakdown=breakdown,
            stats=stats_dict,
            driver_ids=driver_ids,
            route_ids=route_ids,
            infeasible_pairs=infeasible_pairs,
        )
    
    def _compute_effort_breakdown(
        self,
        driver: Driver,
        route: Route,
        driver_stats: dict,
    ) -> EffortBreakdown:
        """
        Compute effort breakdown for a single driver-route pair.
        
        Formula:
        effort = α·packages + β·weight + γ·difficulty + δ·time + ε·mismatch_penalty
        
        Breakdown components:
        - physical_effort: packages + weight + physical component of difficulty
        - route_complexity: stops contribution + difficulty score
        - time_pressure: estimated time factor
        - capacity_penalty: overload penalty
        """
        w = self.weights
        
        # Route features
        num_packages = route.num_packages or 0
        total_weight_kg = route.total_weight_kg or 0.0
        num_stops = route.num_stops or 0
        difficulty = route.route_difficulty_score or 1.0
        estimated_time = route.estimated_time_minutes or 60
        
        # Driver features
        vehicle_capacity = driver.vehicle_capacity_kg or 100.0
        
        # Physical effort component
        # Includes packages, weight, and physical aspects of difficulty (stairs, heavy items)
        physical_effort = (
            w.alpha_packages * num_packages +
            w.beta_weight * total_weight_kg +
            (w.gamma_difficulty * difficulty * 0.4)  # 40% of difficulty is physical
        )
        
        # Route complexity component
        # Includes navigation difficulty, number of stops, parking challenges
        route_complexity = (
            (w.gamma_difficulty * difficulty * 0.6) +  # 60% of difficulty is complexity
            (num_stops * 0.5)  # Each stop adds complexity
        )
        
        # Time pressure component
        time_pressure = w.delta_time * estimated_time
        
        # Capacity mismatch penalty
        capacity_penalty = 0.0
        if vehicle_capacity > 0:
            load_ratio = total_weight_kg / vehicle_capacity
            if load_ratio > 1.0:
                # Overloaded - significant penalty
                capacity_penalty = w.epsilon_mismatch * (load_ratio - 1.0) * 10
            elif load_ratio > 0.9:
                # Near capacity - small penalty
                capacity_penalty = w.epsilon_mismatch * (load_ratio - 0.9) * 2
        
        # Factor in driver fatigue if available
        fatigue_level = driver_stats.get("fatigue_level", 0)
        if fatigue_level > 0:
            # Increase effort perception with fatigue
            fatigue_multiplier = 1.0 + (fatigue_level * 0.1)
            physical_effort *= fatigue_multiplier
        
        # Total effort
        total_effort = (
            physical_effort +
            route_complexity +
            time_pressure +
            capacity_penalty
        )
        
        return EffortBreakdown(
            physical_effort=round(physical_effort, 2),
            route_complexity=round(route_complexity, 2),
            time_pressure=round(time_pressure, 2),
            capacity_penalty=round(capacity_penalty, 2),
            total=round(total_effort, 2),
        )
    
    def get_input_snapshot(
        self,
        drivers: List[Driver],
        routes: List[Route],
    ) -> dict:
        """Generate input snapshot for DecisionLog."""
        return {
            "num_drivers": len(drivers),
            "num_routes": len(routes),
            "weights": self.weights.model_dump(),
        }
    
    def get_output_snapshot(self, result: EffortMatrixResult) -> dict:
        """Generate output snapshot for DecisionLog."""
        return {
            "matrix_shape": [len(result.matrix), len(result.matrix[0]) if result.matrix else 0],
            "min_effort": result.stats.get("min", 0),
            "max_effort": result.stats.get("max", 0),
            "avg_effort": result.stats.get("avg", 0),
        }
