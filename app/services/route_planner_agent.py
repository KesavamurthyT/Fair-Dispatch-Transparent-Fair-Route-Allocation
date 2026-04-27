"""
Route Planner Agent - Optimal driver-route assignment using OR-Tools.
Phase 4.1 implementation with support for fairness penalty re-optimization.
"""

import uuid
from typing import Dict, List, Optional

from app.models.driver import Driver
from app.models.route import Route
from app.schemas.agent_schemas import (
    AllocationItem,
    EffortMatrixResult,
    RoutePlanResult,
    FairnessRecommendations,
)


class RoutePlannerAgent:
    """
    Route Planner Agent uses linear assignment optimization to match
    drivers to routes, minimizing total effort (cost).
    
    Supports two-pass optimization:
    - Proposal 1: Pure effort-based assignment
    - Proposal 2: Effort + fairness penalty for rebalancing
    """
    
    def __init__(self):
        """Initialize the route planner agent."""
        self._or_tools_available = self._check_or_tools()
    
    def _check_or_tools(self) -> bool:
        """Check if OR-Tools is available."""
        try:
            from ortools.linear_solver import pywraplp
            return True
        except ImportError:
            return False
    
    def plan(
        self,
        effort_result: EffortMatrixResult,
        drivers: List[Driver],
        routes: List[Route],
        fairness_penalties: Optional[Dict[str, float]] = None,
        recovery_targets: Optional[Dict[str, float]] = None,
        recovery_penalty_weight: float = 3.0,
        proposal_number: int = 1,
    ) -> RoutePlanResult:
        """
        Generate optimal driver-route assignment.
        
        Args:
            effort_result: Effort matrix from MLEffortAgent
            drivers: List of drivers in matrix row order
            routes: List of routes in matrix column order
            fairness_penalties: Optional dict of driver_id -> penalty multiplier
            recovery_targets: Optional dict of driver_id -> max effort target
            recovery_penalty_weight: Weight for recovery penalty (default 3.0)
            proposal_number: 1 for initial, 2 for re-optimized
            
        Returns:
            RoutePlanResult with assignments and metrics
        """
        matrix = effort_result.matrix
        driver_ids = effort_result.driver_ids
        route_ids = effort_result.route_ids
        infeasible_pairs = set(effort_result.infeasible_pairs)
        
        if not matrix or not drivers or not routes:
            return RoutePlanResult(
                allocation=[],
                total_effort=0.0,
                avg_effort=0.0,
                per_driver_effort={},
                proposal_number=proposal_number,
            )
        
        # Apply fairness and recovery penalties
        cost_matrix = self._apply_penalties(
            matrix=matrix,
            driver_ids=driver_ids,
            route_ids=route_ids,
            fairness_penalties=fairness_penalties,
            recovery_targets=recovery_targets,
            recovery_penalty_weight=recovery_penalty_weight,
            infeasible_pairs=infeasible_pairs,
        )
        
        # Solve assignment problem
        if self._or_tools_available:
            assignments = self._solve_with_ortools(cost_matrix, len(drivers), len(routes))
        else:
            # Fallback to Hungarian algorithm implementation
            assignments = self._solve_hungarian(cost_matrix, len(drivers), len(routes))
        
        # Build result
        allocation: List[AllocationItem] = []
        per_driver_effort: Dict[str, float] = {}
        total_effort = 0.0
        
        for driver_idx, route_idx in assignments:
            if route_idx < len(routes):
                driver = drivers[driver_idx]
                route = routes[route_idx]
                effort = matrix[driver_idx][route_idx]  # Use original effort, not penalized
                
                # Skip if infeasible (shouldn't happen, but safety check)
                key = f"{driver_ids[driver_idx]}:{route_ids[route_idx]}"
                if key in infeasible_pairs:
                    continue
                
                allocation.append(AllocationItem(
                    driver_id=driver.id,
                    route_id=route.id,
                    effort=round(effort, 2),
                ))
                
                per_driver_effort[str(driver.id)] = round(effort, 2)
                total_effort += effort
        
        avg_effort = total_effort / len(allocation) if allocation else 0.0
        
        return RoutePlanResult(
            allocation=allocation,
            total_effort=round(total_effort, 2),
            avg_effort=round(avg_effort, 2),
            per_driver_effort=per_driver_effort,
            proposal_number=proposal_number,
        )
    
    def _apply_penalties(
        self,
        matrix: List[List[float]],
        driver_ids: List[str],
        route_ids: List[str],
        fairness_penalties: Optional[Dict[str, float]] = None,
        recovery_targets: Optional[Dict[str, float]] = None,
        recovery_penalty_weight: float = 3.0,
        infeasible_pairs: Optional[set] = None,
    ) -> List[List[float]]:
        """Apply fairness and recovery penalties to cost matrix."""
        from app.services.recovery_service import calculate_recovery_penalty
        
        fairness_penalties = fairness_penalties or {}
        recovery_targets = recovery_targets or {}
        infeasible_pairs = infeasible_pairs or set()
        
        cost_matrix = []
        for i, row in enumerate(matrix):
            driver_id = driver_ids[i]
            fairness_mult = fairness_penalties.get(driver_id, 1.0)
            recovery_target = recovery_targets.get(driver_id)
            
            new_row = []
            for j, effort in enumerate(row):
                route_id = route_ids[j]
                key = f"{driver_id}:{route_id}"
                
                # Check if infeasible
                if key in infeasible_pairs:
                    new_row.append(99999.0)
                    continue
                
                # Apply fairness penalty
                penalized = effort * fairness_mult
                
                # Apply recovery penalty
                recovery_penalty = calculate_recovery_penalty(
                    effort, recovery_target, recovery_penalty_weight
                )
                
                new_row.append(penalized + recovery_penalty)
            
            cost_matrix.append(new_row)
        
        return cost_matrix
    
    def _solve_with_ortools(
        self,
        cost_matrix: List[List[float]],
        num_drivers: int,
        num_routes: int,
    ) -> List[tuple]:
        """
        Solve assignment using OR-Tools linear solver.
        
        Returns list of (driver_idx, route_idx) assignments.
        """
        from ortools.linear_solver import pywraplp
        
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            # Fallback to GLOP
            solver = pywraplp.Solver.CreateSolver('GLOP')
        
        # Decision variables: x[i][j] = 1 if driver i assigned to route j
        x = {}
        for i in range(num_drivers):
            for j in range(num_routes):
                x[i, j] = solver.BoolVar(f'x_{i}_{j}')
        
        # Constraints: each driver assigned to at most 1 route
        for i in range(num_drivers):
            solver.Add(sum(x[i, j] for j in range(num_routes)) <= 1)
        
        # Constraints: each route assigned to exactly 1 driver
        for j in range(num_routes):
            solver.Add(sum(x[i, j] for i in range(num_drivers)) == 1)
        
        # Objective: minimize total cost
        objective_terms = []
        for i in range(num_drivers):
            for j in range(num_routes):
                objective_terms.append(cost_matrix[i][j] * x[i, j])
        
        solver.Minimize(sum(objective_terms))
        
        # Solve
        status = solver.Solve()
        
        assignments = []
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for i in range(num_drivers):
                for j in range(num_routes):
                    if x[i, j].solution_value() > 0.5:
                        assignments.append((i, j))
        else:
            # If OR-Tools fails, fallback to greedy
            assignments = self._greedy_assignment(cost_matrix, num_drivers, num_routes)
        
        return assignments
    
    def _solve_hungarian(
        self,
        cost_matrix: List[List[float]],
        num_drivers: int,
        num_routes: int,
    ) -> List[tuple]:
        """
        Solve assignment using Hungarian algorithm.
        Falls back to scipy if available, otherwise uses greedy.
        """
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
            
            # Pad matrix if needed (more drivers than routes or vice versa)
            max_dim = max(num_drivers, num_routes)
            padded = np.full((max_dim, max_dim), 1e9)
            
            for i in range(num_drivers):
                for j in range(num_routes):
                    padded[i][j] = cost_matrix[i][j]
            
            row_ind, col_ind = linear_sum_assignment(padded)
            
            # Filter to valid assignments
            assignments = []
            for i, j in zip(row_ind, col_ind):
                if i < num_drivers and j < num_routes and padded[i][j] < 1e8:
                    assignments.append((i, j))
            
            return assignments
            
        except ImportError:
            return self._greedy_assignment(cost_matrix, num_drivers, num_routes)
    
    def _greedy_assignment(
        self,
        cost_matrix: List[List[float]],
        num_drivers: int,
        num_routes: int,
    ) -> List[tuple]:
        """
        Greedy fallback assignment.
        Assigns routes to drivers in order of lowest cost.
        """
        # Build list of all (cost, driver_idx, route_idx)
        candidates = []
        for i in range(num_drivers):
            for j in range(num_routes):
                candidates.append((cost_matrix[i][j], i, j))
        
        # Sort by cost
        candidates.sort(key=lambda x: x[0])
        
        assigned_drivers = set()
        assigned_routes = set()
        assignments = []
        
        for cost, driver_idx, route_idx in candidates:
            if driver_idx not in assigned_drivers and route_idx not in assigned_routes:
                assignments.append((driver_idx, route_idx))
                assigned_drivers.add(driver_idx)
                assigned_routes.add(route_idx)
                
                if len(assignments) == min(num_drivers, num_routes):
                    break
        
        return assignments
    
    def build_penalties_from_recommendations(
        self,
        recommendations: FairnessRecommendations,
        per_driver_effort: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Build penalty dict from fairness recommendations.
        
        High-effort drivers get their route costs multiplied by penalty_factor,
        making the solver prefer to give them easier routes.
        """
        penalties = {}
        
        for driver_id in per_driver_effort:
            if driver_id in recommendations.high_effort_driver_ids:
                penalties[driver_id] = recommendations.penalty_factor
            else:
                penalties[driver_id] = 1.0
        
        return penalties
    
    def get_input_snapshot(
        self,
        effort_result: EffortMatrixResult,
        penalties: Optional[Dict[str, float]] = None,
    ) -> dict:
        """Generate input snapshot for DecisionLog."""
        return {
            "matrix_shape": [
                len(effort_result.matrix),
                len(effort_result.matrix[0]) if effort_result.matrix else 0
            ],
            "effort_stats": effort_result.stats,
            "has_penalties": penalties is not None and len(penalties) > 0,
            "num_penalized_drivers": len([p for p in (penalties or {}).values() if p > 1.0]),
        }
    
    def get_output_snapshot(self, result: RoutePlanResult) -> dict:
        """Generate output snapshot for DecisionLog."""
        efforts = list(result.per_driver_effort.values())
        return {
            "proposal_number": result.proposal_number,
            "num_assignments": len(result.allocation),
            "total_effort": result.total_effort,
            "avg_effort": result.avg_effort,
            "min_driver_effort": min(efforts) if efforts else 0,
            "max_driver_effort": max(efforts) if efforts else 0,
        }
