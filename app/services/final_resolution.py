"""
Final Resolution Agent - Local swap logic for honoring COUNTER decisions.
Phase 4.2 implementation for post-negotiation optimization.
"""

import statistics
from typing import Dict, List, Optional

from app.schemas.agent_schemas import (
    DriverLiaisonDecision,
    FinalResolutionResult,
    FairnessMetrics,
    RoutePlanResult,
    SwapRecord,
)


class FinalResolutionAgent:
    """
    Final Resolution Agent attempts to honor COUNTER decisions using local swaps.
    
    Strategy:
    1. Build driver↔route mappings from approved proposal
    2. For each COUNTER decision:
       - Try 1-to-1 swap with current route holder
       - Evaluate fairness impact
       - Accept swap only if fairness stays within tolerance
    3. Return updated allocation with swap records
    """
    
    # Default tolerance for accepting swaps
    METRIC_EPSILON: float = 0.02  # Allow slight worsening (2%)
    
    def __init__(self, metric_epsilon: float = 0.02):
        """
        Initialize agent with configurable tolerance.
        
        Args:
            metric_epsilon: Maximum allowed metric degradation (as ratio)
        """
        self.METRIC_EPSILON = metric_epsilon
    
    def resolve_counters(
        self,
        approved_proposal: RoutePlanResult,
        decisions: List[DriverLiaisonDecision],
        effort_matrix: List[List[float]],
        driver_ids: List[str],
        route_ids: List[str],
        current_metrics: FairnessMetrics,
    ) -> FinalResolutionResult:
        """
        Attempt to resolve COUNTER decisions via local swaps.
        
        Args:
            approved_proposal: The accepted proposal from fairness manager
            decisions: All driver liaison decisions
            effort_matrix: 2D effort matrix [driver_idx][route_idx]
            driver_ids: Driver IDs in matrix row order
            route_ids: Route IDs in matrix column order
            current_metrics: Current fairness metrics of approved proposal
        
        Returns:
            FinalResolutionResult with potentially updated allocation
        """
        # Build index maps
        driver_idx_map = {did: idx for idx, did in enumerate(driver_ids)}
        route_idx_map = {rid: idx for idx, rid in enumerate(route_ids)}
        
        # Build current mappings from proposal
        driver_to_route: Dict[str, str] = {}
        route_to_driver: Dict[str, str] = {}
        per_driver_effort: Dict[str, float] = {}
        
        for item in approved_proposal.allocation:
            did = str(item.driver_id)
            rid = str(item.route_id)
            driver_to_route[did] = rid
            route_to_driver[rid] = did
            per_driver_effort[did] = item.effort
        
        # Track current metrics
        current_gini = current_metrics.gini_index
        current_std = current_metrics.std_dev
        current_max_gap = current_metrics.max_gap
        
        swaps_applied: List[SwapRecord] = []
        unfulfilled_counters: List[str] = []
        
        # Filter COUNTER decisions
        counter_decisions = [d for d in decisions if d.decision == "COUNTER"]
        
        # Process each counter
        for counter in counter_decisions:
            driver_a = counter.driver_id
            preferred_route = counter.preferred_route_id
            
            # Skip if no preferred route specified
            if not preferred_route:
                unfulfilled_counters.append(driver_a)
                continue
            
            # Skip if driver A not in mapping
            if driver_a not in driver_to_route:
                unfulfilled_counters.append(driver_a)
                continue
            
            route_a = driver_to_route[driver_a]
            
            # Skip if preferred route not assigned to anyone
            if preferred_route not in route_to_driver:
                unfulfilled_counters.append(driver_a)
                continue
            
            driver_b = route_to_driver[preferred_route]
            route_b = preferred_route
            
            # Skip if trying to swap with self
            if driver_a == driver_b:
                unfulfilled_counters.append(driver_a)
                continue
            
            # Get indices for effort lookup
            idx_a = driver_idx_map.get(driver_a)
            idx_b = driver_idx_map.get(driver_b)
            idx_route_a = route_idx_map.get(route_a)
            idx_route_b = route_idx_map.get(route_b)
            
            if any(x is None for x in [idx_a, idx_b, idx_route_a, idx_route_b]):
                unfulfilled_counters.append(driver_a)
                continue
            
            # Current efforts
            effort_a_before = per_driver_effort[driver_a]
            effort_b_before = per_driver_effort[driver_b]
            
            # New efforts after swap
            effort_a_after = effort_matrix[idx_a][idx_route_b]  # A gets route B
            effort_b_after = effort_matrix[idx_b][idx_route_a]  # B gets route A
            
            # Evaluate swap: compute new metrics
            test_efforts = per_driver_effort.copy()
            test_efforts[driver_a] = effort_a_after
            test_efforts[driver_b] = effort_b_after
            
            new_metrics = self._compute_metrics(list(test_efforts.values()))
            
            # Check if swap is acceptable
            if self._is_swap_acceptable(
                current_gini, current_std, current_max_gap,
                new_metrics, effort_b_before, effort_b_after
            ):
                # Apply swap
                driver_to_route[driver_a] = route_b
                driver_to_route[driver_b] = route_a
                route_to_driver[route_a] = driver_b
                route_to_driver[route_b] = driver_a
                per_driver_effort[driver_a] = effort_a_after
                per_driver_effort[driver_b] = effort_b_after
                
                # Update current metrics
                current_gini = new_metrics["gini_index"]
                current_std = new_metrics["std_dev"]
                current_max_gap = new_metrics["max_gap"]
                
                # Record swap
                swaps_applied.append(SwapRecord(
                    driver_a=driver_a,
                    driver_b=driver_b,
                    route_a=route_a,
                    route_b=route_b,
                    effort_a_before=effort_a_before,
                    effort_a_after=effort_a_after,
                    effort_b_before=effort_b_before,
                    effort_b_after=effort_b_after,
                ))
            else:
                unfulfilled_counters.append(driver_a)
        
        # Build final allocation list
        allocation = [
            {"driver_id": did, "route_id": rid, "effort": per_driver_effort[did]}
            for did, rid in driver_to_route.items()
        ]
        
        final_metrics = self._compute_metrics(list(per_driver_effort.values()))
        
        return FinalResolutionResult(
            allocation=allocation,
            per_driver_effort=per_driver_effort,
            metrics=final_metrics,
            swaps_applied=swaps_applied,
            unfulfilled_counters=unfulfilled_counters,
        )
    
    def _compute_metrics(self, efforts: List[float]) -> Dict[str, float]:
        """Compute fairness metrics from effort values."""
        if not efforts:
            return {
                "avg_effort": 0.0,
                "std_dev": 0.0,
                "max_gap": 0.0,
                "gini_index": 0.0,
            }
        
        n = len(efforts)
        avg = statistics.mean(efforts)
        min_e = min(efforts)
        max_e = max(efforts)
        std = statistics.stdev(efforts) if n > 1 else 0.0
        gini = self._compute_gini(efforts)
        
        return {
            "avg_effort": round(avg, 2),
            "std_dev": round(std, 2),
            "max_gap": round(max_e - min_e, 2),
            "gini_index": round(gini, 4),
            "min_effort": round(min_e, 2),
            "max_effort": round(max_e, 2),
        }
    
    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient."""
        if not values or len(values) == 1:
            return 0.0
        
        n = len(values)
        mean = statistics.mean(values)
        if mean == 0:
            return 0.0
        
        total_diff = sum(
            abs(values[i] - values[j])
            for i in range(n)
            for j in range(n)
        )
        
        return min(total_diff / (2 * n * n * mean), 1.0)
    
    def _is_swap_acceptable(
        self,
        old_gini: float,
        old_std: float,
        old_max_gap: float,
        new_metrics: Dict[str, float],
        effort_b_before: float,
        effort_b_after: float,
    ) -> bool:
        """
        Check if a swap is acceptable based on fairness impact.
        
        Accept if:
        - New metrics not significantly worse (within epsilon)
        - OR any metric strictly improves
        - AND driver B's new effort is not drastically worse
        """
        new_gini = new_metrics["gini_index"]
        new_std = new_metrics["std_dev"]
        new_max_gap = new_metrics["max_gap"]
        
        # Check if any metric improves
        improves = (
            new_gini < old_gini - 0.001 or
            new_std < old_std - 0.1 or
            new_max_gap < old_max_gap - 0.1
        )
        
        # Check if metrics stay within tolerance
        gini_ok = new_gini <= old_gini * (1 + self.METRIC_EPSILON)
        std_ok = new_std <= old_std * (1 + self.METRIC_EPSILON) + 0.5
        gap_ok = new_max_gap <= old_max_gap * (1 + self.METRIC_EPSILON) + 0.5
        
        within_tolerance = gini_ok and std_ok and gap_ok
        
        # Check driver B impact - don't drastically increase their effort
        # Allow up to 30% increase for driver B
        b_increase_ok = effort_b_after <= effort_b_before * 1.30 + 5.0
        
        return (within_tolerance or improves) and b_increase_ok
    
    def get_input_snapshot(
        self,
        num_counters: int,
        current_metrics: FairnessMetrics,
        global_avg: float,
    ) -> dict:
        """Generate input snapshot for DecisionLog."""
        return {
            "num_counters": num_counters,
            "original_gini": current_metrics.gini_index,
            "original_std_dev": current_metrics.std_dev,
            "original_max_gap": current_metrics.max_gap,
            "global_avg_effort": round(global_avg, 2),
            "metric_epsilon": self.METRIC_EPSILON,
        }
    
    def get_output_snapshot(self, result: FinalResolutionResult) -> dict:
        """Generate output snapshot for DecisionLog."""
        return {
            "num_swaps_applied": len(result.swaps_applied),
            "num_unfulfilled": len(result.unfulfilled_counters),
            "final_gini": result.metrics.get("gini_index", 0),
            "final_std_dev": result.metrics.get("std_dev", 0),
            "final_max_gap": result.metrics.get("max_gap", 0),
            "swaps": [
                {"driver_a": s.driver_a, "driver_b": s.driver_b}
                for s in result.swaps_applied[:5]  # First 5 swaps
            ],
        }
