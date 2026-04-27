"""
Route allocation service.
Uses Hungarian algorithm to optimally assign routes to drivers.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class AllocationResult:
    """Result of route allocation for a driver."""
    driver_index: int
    route_index: int
    cost: float


def build_cost_matrix(
    drivers: List[Dict[str, Any]],
    routes: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Build a cost matrix for the assignment problem.
    
    In Phase 1, the cost is simply the route workload score.
    Future phases could incorporate driver preferences, history, etc.
    
    Args:
        drivers: List of driver dicts
        routes: List of route dicts with workload_score
    
    Returns:
        Cost matrix of shape (num_drivers, num_routes)
    """
    num_drivers = len(drivers)
    num_routes = len(routes)
    
    # Initialize cost matrix
    cost_matrix = np.zeros((num_drivers, num_routes))
    
    for i, driver in enumerate(drivers):
        for j, route in enumerate(routes):
            # Base cost is the route workload score
            cost = route.get("workload_score", 0.0)
            
            # Future: could add driver-specific costs here
            # e.g., penalize if route weight exceeds vehicle capacity
            vehicle_capacity = driver.get("vehicle_capacity_kg", 100.0)
            route_weight = route.get("total_weight_kg", 0.0)
            
            if route_weight > vehicle_capacity:
                # Penalize over-capacity assignments
                cost += (route_weight - vehicle_capacity) * 10
            
            cost_matrix[i, j] = cost
    
    return cost_matrix


def allocate_routes(
    drivers: List[Dict[str, Any]],
    routes: List[Dict[str, Any]],
) -> List[AllocationResult]:
    """
    Allocate routes to drivers using the Hungarian algorithm.
    
    The Hungarian algorithm finds the optimal assignment that minimizes
    total cost (in this case, tries to balance workload).
    
    Args:
        drivers: List of driver dicts
        routes: List of route dicts with workload_score
    
    Returns:
        List of AllocationResult objects
    """
    num_drivers = len(drivers)
    num_routes = len(routes)
    
    if num_drivers == 0 or num_routes == 0:
        return []
    
    # Handle rectangular matrices (more drivers than routes or vice versa)
    if num_drivers != num_routes:
        # Pad to make square matrix
        n = max(num_drivers, num_routes)
        cost_matrix = np.full((n, n), fill_value=1e9)  # High cost for dummy assignments
        
        actual_costs = build_cost_matrix(drivers, routes)
        cost_matrix[:num_drivers, :num_routes] = actual_costs
    else:
        cost_matrix = build_cost_matrix(drivers, routes)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Extract valid assignments (ignore dummy assignments)
    results = []
    for driver_idx, route_idx in zip(row_ind, col_ind):
        if driver_idx < num_drivers and route_idx < num_routes:
            results.append(AllocationResult(
                driver_index=int(driver_idx),
                route_index=int(route_idx),
                cost=cost_matrix[driver_idx, route_idx],
            ))
    
    return results


def greedy_allocate(
    drivers: List[Dict[str, Any]],
    routes: List[Dict[str, Any]],
) -> List[AllocationResult]:
    """
    Simple greedy allocation as a fallback.
    Assigns each route to the next available driver in order.
    
    Args:
        drivers: List of driver dicts
        routes: List of route dicts
    
    Returns:
        List of AllocationResult objects
    """
    results = []
    num_assignments = min(len(drivers), len(routes))
    
    for i in range(num_assignments):
        results.append(AllocationResult(
            driver_index=i,
            route_index=i,
            cost=routes[i].get("workload_score", 0.0),
        ))
    
    return results
