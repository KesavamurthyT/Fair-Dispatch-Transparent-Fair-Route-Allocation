import abc
import numpy as np
import networkx as nx
try:
    import osmnx as ox
except ImportError:
    ox = None
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from models import Driver, Package

@dataclass
class Solution:
    """Stores the result of the optimization."""
    assignments: Dict[str, List[str]] = field(default_factory=dict) # Driver ID -> List of Package IDs
    routes: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict) # Driver ID -> List of (lat, lon) coordinates for the path
    total_distance: float = 0.0
    fairness_score: float = 0.0 # Variance in route distance

class DistanceCache:
    """Simple cache for network distances to avoid re-computation."""
    _cache = {}

    @classmethod
    def get_dist(cls, graph, source, target):
        key = (source, target)
        if key in cls._cache:
            return cls._cache[key]
        try:
            val = nx.shortest_path_length(graph, source, target, weight='length')
            cls._cache[key] = val
            return val
        except nx.NetworkXNoPath:
            return float('inf')

    @classmethod
    def clear(cls):
        cls._cache = {}

class RouteOptimizer(abc.ABC):
    """Abstract Base Class for Route Optimization Strategies."""
    
    @abc.abstractmethod
    def solve(self, drivers: List[Driver], packages: List[Package], graph: Any) -> Solution:
        """Calculate optimal routes."""
        pass

    def _get_path_length(self, graph, path_nodes):
        """Calculate total length of a path in meters."""
        length = 0
        for i in range(len(path_nodes) - 1):
            if path_nodes[i] == path_nodes[i+1]: continue
            length += DistanceCache.get_dist(graph, path_nodes[i], path_nodes[i+1])
        return length

    def _get_route_coordinates(self, graph, path_nodes):
        """Convert list of node IDs to (lat, lon) coordinates."""
        coords = []
        for i in range(len(path_nodes)):
            # If we need the actual path between nodes, we must compute it
            # For visualization, we DO want the full path
            if i < len(path_nodes) - 1 and path_nodes[i] != path_nodes[i+1]:
                try:
                    segment = nx.shortest_path(graph, path_nodes[i], path_nodes[i+1], weight='length')
                    # Don't add the last point to avoid duplicates with next segment start
                    for node in segment[:-1]:
                        coords.append((graph.nodes[node]['y'], graph.nodes[node]['x']))
                except nx.NetworkXNoPath:
                    pass
            else:
                 # Last point
                 node = path_nodes[i]
                 coords.append((graph.nodes[node]['y'], graph.nodes[node]['x']))
                 
        # Ensure last point is added
        if path_nodes:
            last = path_nodes[-1]
            coords.append((graph.nodes[last]['y'], graph.nodes[last]['x']))
            
        return coords

class SimpleNearestNeighbor(RouteOptimizer):
    """Greedy strategy that assigns nearest available package to driver."""

    def solve(self, drivers: List[Driver], packages: List[Package], graph: Any) -> Solution:
        solution = Solution()
        DistanceCache.clear() # Clear cache for new run
        
        unassigned_packages = packages.copy()
        driver_paths = {d.id: [d.node_id] for d in drivers} 
        driver_distances = {d.id: 0.0 for d in drivers}
        
        while unassigned_packages:
            for driver in drivers:
                if not unassigned_packages:
                    break
                
                current_node = driver_paths[driver.id][-1]
                
                best_pkg = None
                best_dist = float('inf')
                
                for pkg in unassigned_packages:
                    dist = DistanceCache.get_dist(graph, current_node, pkg.node_id)
                    if dist < best_dist:
                        best_dist = dist
                        best_pkg = pkg
                
                if best_pkg:
                    driver_paths[driver.id].append(best_pkg.node_id)
                    driver_distances[driver.id] += best_dist
                    solution.assignments.setdefault(driver.id, []).append(best_pkg.id)
                    unassigned_packages.remove(best_pkg)
        
        for driver in drivers:
            solution.routes[driver.id] = self._get_route_coordinates(graph, driver_paths[driver.id])
            
        solution.total_distance = sum(driver_distances.values())
        if driver_distances:
            solution.fairness_score = np.var(list(driver_distances.values()))
            
        return solution

class ClusterAndRoute(RouteOptimizer):
    """K-Means Clustering + TSP for balanced zones."""
    
    def solve(self, drivers: List[Driver], packages: List[Package], graph: Any) -> Solution:
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("sklearn not found, using Mock/Fallback or failing gracefully.")
            return Solution() # Fallback
            
        solution = Solution()
        DistanceCache.clear()
        
        if not packages or not drivers:
            return solution
             
        # 1. Cluster packages
        pkg_coords = np.array([[p.lat, p.lon] for p in packages])
        n_clusters = min(len(packages), len(drivers))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(pkg_coords)
        
        cluster_map = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            cluster_map[label].append(packages[idx])
            
        # 2. Match drivers to clusters (Greedy by Centroid)
        available_clusters = list(cluster_map.keys())
        driver_assignments = {}
        
        for driver in drivers:
            if not available_clusters:
                break
            
            best_c_idx = -1
            best_dist = float('inf')
            
            for c_idx in available_clusters:
                pts = cluster_map[c_idx]
                if not pts: continue
                avg_lat = np.mean([p.lat for p in pts])
                avg_lon = np.mean([p.lon for p in pts])
                
                # Euclidean dist 
                d = (driver.lat - avg_lat)**2 + (driver.lon - avg_lon)**2
                if d < best_dist:
                    best_dist = d
                    best_c_idx = c_idx
            
            if best_c_idx != -1:
                driver_assignments[driver.id] = cluster_map[best_c_idx]
                available_clusters.remove(best_c_idx)
                
        # 3. Route within clusters (Nearest Neighbor TSP)
        total_dist = 0
        dist_values = []
        
        for driver_id, pkgs in driver_assignments.items():
            driver = next(d for d in drivers if d.id == driver_id)
            current_node = driver.node_id
            route_nodes = [current_node]
            unvisited = pkgs.copy()
            driver_dist = 0
            route_pkg_ids = []
            
            while unvisited:
                best_p = None
                best_d_val = float('inf')
                
                for p in unvisited:
                    d_val = DistanceCache.get_dist(graph, current_node, p.node_id)
                    if d_val < best_d_val:
                        best_d_val = d_val
                        best_p = p
                
                if best_p:
                    unvisited.remove(best_p)
                    current_node = best_p.node_id
                    route_nodes.append(current_node)
                    driver_dist += best_d_val
                    route_pkg_ids.append(best_p.id)
            
            solution.assignments[driver_id] = route_pkg_ids
            total_dist += driver_dist
            dist_values.append(driver_dist)
            
            solution.routes[driver_id] = self._get_route_coordinates(graph, route_nodes)

        solution.total_distance = total_dist
        if dist_values:
            solution.fairness_score = np.var(dist_values)
            
        return solution

class EfficiencyVRP(RouteOptimizer):
    """
    Efficiency Focused Strategy (Multi-Depot VRP Heuristic).
    Goal: Minimize Total Cost irrespective of Fairness.
    Logic: Assign each package to the NEAREST driver, then optimize that driver's route.
    This differs from ClusterAndRoute because ClusterAndRoute forces 'n' clusters = 'n' drivers (roughly equal work).
    EfficiencyVRP might leave some drivers IDLE if they are far away, saving fuel.
    """
    
    def solve(self, drivers: List[Driver], packages: List[Package], graph: Any) -> Solution:
        solution = Solution()
        DistanceCache.clear()
        
        if not packages or not drivers:
            return solution
            
        driver_packages = {d.id: [] for d in drivers}
        
        # 1. Assignment Phase: Assign each package to the closest driver (by road dist)
        # This minimizes the "stem distance" and generally creates tight local routes.
        # This disregards load balancing (fairness).
        
        for pkg in packages:
            best_driver = None
            best_dist = float('inf')
            
            for driver in drivers:
                # To speed up, check Euclidean first? 
                # Prompts ask for optimal, so cache handles the network dist
                dist = DistanceCache.get_dist(graph, driver.node_id, pkg.node_id)
                if dist < best_dist:
                    best_dist = dist
                    best_driver = driver
            
            if best_driver:
                driver_packages[best_driver.id].append(pkg)
                
        # 2. Routing Phase: Optimize each driver's assigned pile
        total_dist = 0
        dist_values = []
        
        for driver in drivers:
            my_pkgs = driver_packages[driver.id]
            if not my_pkgs:
                solution.routes[driver.id] = []
                dist_values.append(0)
                continue
                
            # TSP solving for this driver
            current_node = driver.node_id
            route_nodes = [current_node]
            unvisited = my_pkgs.copy()
            driver_dist = 0
            route_pkg_ids = []
            
            while unvisited:
                best_p = None
                best_d_val = float('inf')
                
                for p in unvisited:
                    d_val = DistanceCache.get_dist(graph, current_node, p.node_id)
                    if d_val < best_d_val:
                        best_d_val = d_val
                        best_p = p
                
                if best_p:
                    unvisited.remove(best_p)
                    current_node = best_p.node_id
                    route_nodes.append(current_node)
                    driver_dist += best_d_val
                    route_pkg_ids.append(best_p.id)
            
            solution.assignments[driver.id] = route_pkg_ids
            total_dist += driver_dist
            dist_values.append(driver_dist)
            solution.routes[driver.id] = self._get_route_coordinates(graph, route_nodes)
            
        solution.total_distance = total_dist
        if dist_values:
            solution.fairness_score = np.var(dist_values)
            
        return solution

