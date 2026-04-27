"""
Capacitated Vehicle Routing Problem (CVRP) solver using OR-Tools.

Replaces the previous Linear Assignment Problem (LAP) approach with a
proper CVRP formulation that supports:
  - Capacity constraints (driver weight limits)
  - Multiple stops per driver
  - EV range constraints (battery distance limits)
  - Fairness-aware cost penalties (two-pass approach)
  - Graceful fallback to greedy heuristic on solver timeout

Usage:
    from app.solver.cvrp_solver import CVRPSolver

    solver = CVRPSolver(
        distance_matrix=distance_matrix,
        demands=demands,
        vehicle_capacities=capacities,
    )
    solution = solver.solve()
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Scaling factor: OR-Tools requires integer costs
COST_SCALE = 100

# Deterministic solver seed — reproducible across identical inputs.
# Override via env var CVRP_RANDOM_SEED for A/B testing.
CVRP_RANDOM_SEED = int(os.getenv("CVRP_RANDOM_SEED", "42"))


@dataclass
class CVRPVehicle:
    """Represents a driver/vehicle in the CVRP model."""
    id: str
    capacity_kg: float
    max_range_km: Optional[float] = None  # None = unlimited (ICE)
    is_ev: bool = False
    fatigue_penalty: float = 0.0  # higher = more fatigued, add to cost


@dataclass
class CVRPNode:
    """Represents a stop (package delivery location) in the CVRP model."""
    id: str
    latitude: float
    longitude: float
    demand_kg: float  # package weight
    route_id: Optional[str] = None  # cluster/route ID this node belongs to


@dataclass
class CVRPSolution:
    """Result of CVRP solving."""
    assignments: Dict[str, List[str]]  # vehicle_id -> [node_ids in order]
    total_distance: float
    per_vehicle_distance: Dict[str, float]
    per_vehicle_load: Dict[str, float]
    per_vehicle_effort: Dict[str, float]
    status: str  # "optimal", "feasible", "fallback", "infeasible"
    solver_time_ms: int = 0
    # Fairness observability metadata (Fix 5)
    fairness_iterations: int = 0
    initial_gini: float = 0.0
    final_gini: float = 0.0


class CVRPSolver:
    """
    Capacitated Vehicle Routing Problem solver using OR-Tools.

    Formulation:
      - Vehicles = drivers (each starting and ending at depot)
      - Nodes = package delivery stops
      - Demand = package weight (kg)
      - Vehicle capacity = driver weight capacity (kg)
      - Distance dimension = travel distance (for EV range constraints)
      - Cost = distance * alpha + fatigue_penalty * beta + fairness_penalty * gamma

    Two-Pass Fairness:
      1. Solve for minimum cost (distance + fatigue)
      2. Compute Gini index of per-vehicle effort
      3. If Gini > threshold, add penalties to overburdened vehicles and re-solve
    """

    def __init__(
        self,
        depot_lat: float,
        depot_lng: float,
        vehicles: List[CVRPVehicle],
        nodes: List[CVRPNode],
        time_limit_seconds: int = 30,
        fairness_threshold_gini: float = 0.33,
        max_fairness_iterations: int = 3,
    ):
        self.depot_lat = depot_lat
        self.depot_lng = depot_lng
        self.vehicles = vehicles
        self.nodes = nodes
        self.time_limit_seconds = time_limit_seconds
        self.fairness_threshold_gini = fairness_threshold_gini
        self.max_fairness_iterations = max_fairness_iterations

        # Build distance matrix (depot + all nodes)
        self._build_distance_matrix()

    def _haversine(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate Haversine distance in km."""
        R = 6371.0
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _build_distance_matrix(self):
        """Build NxN distance matrix: node 0 = depot, nodes 1..N = delivery stops."""
        n = len(self.nodes) + 1  # +1 for depot
        self.distance_matrix = np.zeros((n, n), dtype=np.float64)

        # All coordinates: [depot, node0, node1, ...]
        coords = [(self.depot_lat, self.depot_lng)]
        for node in self.nodes:
            coords.append((node.latitude, node.longitude))

        for i in range(n):
            for j in range(i + 1, n):
                d = self._haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                self.distance_matrix[i][j] = d
                self.distance_matrix[j][i] = d

    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient of a list of non-negative values."""
        if not values or all(v == 0 for v in values):
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
        return (2 * cumulative) / (n * total) - (n + 1) / n

    def solve(self) -> CVRPSolution:
        """
        Solve the CVRP with two-pass fairness enforcement.

        Pass 1: Minimize distance + fatigue cost.
        Pass 2+: If Gini > threshold, penalize overburdened vehicles and re-solve.
        """
        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        except ImportError:
            logger.warning("OR-Tools constraint solver not available, using fallback")
            return self._greedy_fallback()

        num_nodes = len(self.nodes) + 1  # +1 for depot
        num_vehicles = len(self.vehicles)

        if num_nodes <= 1 or num_vehicles == 0:
            return CVRPSolution(
                assignments={},
                total_distance=0.0,
                per_vehicle_distance={},
                per_vehicle_load={},
                per_vehicle_effort={},
                status="infeasible",
            )

        # Vehicle penalty multipliers (starts at 1.0, increased for overburdened)
        vehicle_cost_multipliers = [1.0] * num_vehicles

        best_solution = None
        initial_gini: float = 0.0
        current_gini: float = 0.0
        iterations_done: int = 0

        for iteration in range(self.max_fairness_iterations + 1):
            solution = self._solve_iteration(
                num_nodes, num_vehicles, vehicle_cost_multipliers
            )

            if solution is None or solution.status == "infeasible":
                if best_solution:
                    return best_solution
                return self._greedy_fallback()

            best_solution = solution
            iterations_done = iteration

            # Check fairness
            efforts = list(solution.per_vehicle_effort.values())
            if not efforts:
                break

            current_gini = self._compute_gini(efforts)

            # Capture initial Gini before any penalties
            if iteration == 0:
                initial_gini = current_gini

            # Structured fairness logging (Fix 5)
            logger.info(
                "Fairness iteration %d | Gini=%.4f | threshold=%.4f",
                iteration,
                current_gini,
                self.fairness_threshold_gini,
            )

            if current_gini <= self.fairness_threshold_gini:
                solution.status = "optimal" if iteration == 0 else "feasible"
                break

            if iteration < self.max_fairness_iterations:
                # Add penalties for overburdened vehicles
                avg_effort = np.mean(efforts) if efforts else 0
                for idx, vehicle in enumerate(self.vehicles):
                    vid = vehicle.id
                    v_effort = solution.per_vehicle_effort.get(vid, 0)
                    if v_effort > avg_effort * 1.2:
                        # Increase cost multiplier for this vehicle
                        penalty_factor = 1.0 + (v_effort - avg_effort) / max(avg_effort, 1.0)
                        vehicle_cost_multipliers[idx] *= penalty_factor
                        logger.debug(
                            "Penalizing vehicle %s: effort=%.1f, multiplier=%.2f",
                            vid,
                            v_effort,
                            vehicle_cost_multipliers[idx],
                        )

        # Attach fairness observability metadata to the best solution
        if best_solution is not None:
            best_solution.fairness_iterations = iterations_done
            best_solution.initial_gini = round(initial_gini, 4)
            best_solution.final_gini = round(current_gini, 4)

        return best_solution

    def _solve_iteration(
        self,
        num_nodes: int,
        num_vehicles: int,
        cost_multipliers: List[float],
    ) -> Optional[CVRPSolution]:
        """Run a single CVRP solve iteration with given cost multipliers."""
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        import time

        start_time = time.time()

        # Depot index = 0 for all vehicles
        manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        # --- Distance / cost callback ---
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node][to_node] * COST_SCALE)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Set cost per vehicle (with fairness multipliers)
        for v_idx in range(num_vehicles):
            multiplier = cost_multipliers[v_idx]
            fatigue = self.vehicles[v_idx].fatigue_penalty

            def cost_callback(from_index, to_index, m=multiplier, f=fatigue):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                base_cost = self.distance_matrix[from_node][to_node] * COST_SCALE
                # Add fatigue penalty for this vehicle
                fatigue_cost = f * COST_SCALE * 0.1
                return int((base_cost + fatigue_cost) * m)

            cb_idx = routing.RegisterTransitCallback(cost_callback)
            routing.SetArcCostEvaluatorOfVehicle(cb_idx, v_idx)

        # --- Capacity constraint ---
        demands = [0]  # depot has 0 demand
        for node in self.nodes:
            demands.append(int(node.demand_kg * COST_SCALE))

        def demand_callback(from_index):
            node = manager.IndexToNode(from_index)
            return demands[node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        vehicle_capacities = [
            int(v.capacity_kg * COST_SCALE) for v in self.vehicles
        ]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # no slack
            vehicle_capacities,
            True,  # start cumul to zero
            "Capacity",
        )

        # --- EV Range constraint (distance dimension) ---
        has_ev = any(v.is_ev and v.max_range_km for v in self.vehicles)
        if has_ev:
            routing.AddDimension(
                transit_callback_index,
                0,  # no slack
                int(max(v.max_range_km or 99999 for v in self.vehicles) * COST_SCALE),
                True,
                "Distance",
            )
            distance_dimension = routing.GetDimensionOrDie("Distance")

            for v_idx, vehicle in enumerate(self.vehicles):
                if vehicle.is_ev and vehicle.max_range_km:
                    max_dist = int(vehicle.max_range_km * COST_SCALE)
                    end_idx = routing.End(v_idx)
                    distance_dimension.CumulVar(end_idx).SetMax(max_dist)

        # --- Allow unserved nodes (with high penalty) ---
        # This prevents infeasibility when capacity is tight
        penalty = int(10000 * COST_SCALE)
        for node_idx in range(1, num_nodes):
            routing.AddDisjunction([manager.NodeToIndex(node_idx)], penalty)

        # --- Search parameters (deterministic via fixed random seed) ---
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = self.time_limit_seconds

        # Fix 1: Apply deterministic seed — guarded for OR-Tools version compatibility.
        # Some versions expose random_seed; others use a different API.
        # PATH_CHEAPEST_ARC (first pass) is always deterministic regardless.
        try:
            search_params.random_seed = CVRP_RANDOM_SEED
        except AttributeError:
            # OR-Tools version installed does not support random_seed field.
            # Determinism is still provided by PATH_CHEAPEST_ARC first solution.
            logger.debug(
                "OR-Tools random_seed not supported in this version — "
                "determinism relies on PATH_CHEAPEST_ARC first solution strategy."
            )
        try:
            search_params.log_search = False  # suppress noisy solver output
        except AttributeError:
            pass  # older versions: no log_search field

        # --- Solve ---
        solution = routing.SolveWithParameters(search_params)

        elapsed_ms = int((time.time() - start_time) * 1000)

        if not solution:
            logger.warning("CVRP solver returned no solution")
            return None

        # --- Extract solution ---
        return self._extract_solution(manager, routing, solution, elapsed_ms)

    def _extract_solution(
        self,
        manager,
        routing,
        solution,
        elapsed_ms: int,
    ) -> CVRPSolution:
        """Extract assignments from OR-Tools solution."""
        assignments: Dict[str, List[str]] = {}
        per_vehicle_distance: Dict[str, float] = {}
        per_vehicle_load: Dict[str, float] = {}
        per_vehicle_effort: Dict[str, float] = {}
        total_distance = 0.0

        for v_idx, vehicle in enumerate(self.vehicles):
            route_nodes = []
            route_distance = 0.0
            route_load = 0.0
            index = routing.Start(v_idx)

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node > 0:  # skip depot
                    route_nodes.append(self.nodes[node - 1].id)
                    route_load += self.nodes[node - 1].demand_kg

                next_index = solution.Value(routing.NextVar(index))
                next_node = manager.IndexToNode(next_index)
                route_distance += self.distance_matrix[node][next_node]
                index = next_index

            vid = vehicle.id
            assignments[vid] = route_nodes
            per_vehicle_distance[vid] = round(route_distance, 2)
            per_vehicle_load[vid] = round(route_load, 2)

            # Effort = weighted combo of distance, load, and fatigue
            effort = (
                route_distance * 1.0
                + route_load * 0.5
                + vehicle.fatigue_penalty * 5.0
                + len(route_nodes) * 2.0
            )
            per_vehicle_effort[vid] = round(effort, 2)
            total_distance += route_distance

        return CVRPSolution(
            assignments=assignments,
            total_distance=round(total_distance, 2),
            per_vehicle_distance=per_vehicle_distance,
            per_vehicle_load=per_vehicle_load,
            per_vehicle_effort=per_vehicle_effort,
            status="feasible",
            solver_time_ms=elapsed_ms,
        )

    def _greedy_fallback(self) -> CVRPSolution:
        """
        Greedy nearest-neighbor fallback for when OR-Tools fails or times out.
        Assigns each unserved node to the nearest vehicle with remaining capacity.
        """
        logger.warning("Using greedy fallback for CVRP")

        assignments: Dict[str, List[str]] = {v.id: [] for v in self.vehicles}
        remaining_capacity = {v.id: v.capacity_kg for v in self.vehicles}
        per_vehicle_distance: Dict[str, float] = {v.id: 0.0 for v in self.vehicles}
        per_vehicle_load: Dict[str, float] = {v.id: 0.0 for v in self.vehicles}
        per_vehicle_effort: Dict[str, float] = {v.id: 0.0 for v in self.vehicles}

        # Current position of each vehicle (starts at depot)
        vehicle_positions = {v.id: (self.depot_lat, self.depot_lng) for v in self.vehicles}

        # Sort nodes by demand descending (assign heavy packages first)
        sorted_nodes = sorted(self.nodes, key=lambda n: n.demand_kg, reverse=True)

        unserved = []

        for node in sorted_nodes:
            best_vehicle = None
            best_distance = float("inf")

            for vehicle in self.vehicles:
                vid = vehicle.id
                if remaining_capacity[vid] < node.demand_kg:
                    continue

                # Check EV range
                if vehicle.is_ev and vehicle.max_range_km:
                    curr_dist = per_vehicle_distance[vid]
                    pos = vehicle_positions[vid]
                    add_dist = self._haversine(pos[0], pos[1], node.latitude, node.longitude)
                    return_dist = self._haversine(
                        node.latitude, node.longitude, self.depot_lat, self.depot_lng
                    )
                    if curr_dist + add_dist + return_dist > vehicle.max_range_km:
                        continue

                pos = vehicle_positions[vid]
                dist = self._haversine(pos[0], pos[1], node.latitude, node.longitude)
                if dist < best_distance:
                    best_distance = dist
                    best_vehicle = vehicle

            if best_vehicle:
                vid = best_vehicle.id
                assignments[vid].append(node.id)
                remaining_capacity[vid] -= node.demand_kg
                per_vehicle_distance[vid] += best_distance
                per_vehicle_load[vid] += node.demand_kg
                vehicle_positions[vid] = (node.latitude, node.longitude)
            else:
                unserved.append(node.id)

        if unserved:
            logger.warning(f"Greedy fallback: {len(unserved)} unserved nodes")

        # Calculate efforts
        total_distance = 0.0
        for vehicle in self.vehicles:
            vid = vehicle.id
            dist = per_vehicle_distance[vid]
            load = per_vehicle_load[vid]
            n_stops = len(assignments[vid])
            effort = dist * 1.0 + load * 0.5 + vehicle.fatigue_penalty * 5.0 + n_stops * 2.0
            per_vehicle_effort[vid] = round(effort, 2)
            per_vehicle_distance[vid] = round(dist, 2)
            per_vehicle_load[vid] = round(load, 2)
            total_distance += dist

        return CVRPSolution(
            assignments=assignments,
            total_distance=round(total_distance, 2),
            per_vehicle_distance=per_vehicle_distance,
            per_vehicle_load=per_vehicle_load,
            per_vehicle_effort=per_vehicle_effort,
            status="fallback",
        )
