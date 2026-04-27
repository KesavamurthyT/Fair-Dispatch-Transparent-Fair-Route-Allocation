"""
Route clustering service.
Uses K-Means to group packages by geographic proximity.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sklearn.cluster import KMeans

from app.config import get_settings


@dataclass
class ClusterResult:
    """Result of clustering operation."""
    cluster_id: int
    packages: List[Dict[str, Any]]
    total_weight_kg: float
    num_packages: int
    num_stops: int
    centroid: tuple  # (lat, lng)


def cluster_packages(
    packages: List[Dict[str, Any]],
    num_drivers: int,
    target_per_route: int = None,
) -> List[ClusterResult]:
    """
    Cluster packages into routes using K-Means based on geographic location.
    
    Args:
        packages: List of package dicts with latitude, longitude, weight_kg, address
        num_drivers: Number of available drivers
        target_per_route: Target packages per route (default from settings)
    
    Returns:
        List of ClusterResult objects, each representing a route
    """
    settings = get_settings()
    target_per_route = target_per_route or settings.target_packages_per_route
    
    if not packages:
        return []
    
    # For fair dispatch: create one route per driver (up to num packages)
    # This ensures each driver gets assigned work
    num_routes = min(num_drivers, len(packages))
    
    # Handle edge case: fewer packages than routes (already covered above)
    
    # Extract coordinates for clustering
    coords = np.array([
        [p["latitude"], p["longitude"]] for p in packages
    ])
    
    # Handle single cluster case
    if num_routes == 1:
        cluster_labels = np.zeros(len(packages), dtype=int)
        centroid = (coords[:, 0].mean(), coords[:, 1].mean())
        centroids = [centroid]
    else:
        # Perform K-Means clustering
        kmeans = KMeans(
            n_clusters=num_routes,
            random_state=42,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(coords)
        centroids = [(c[0], c[1]) for c in kmeans.cluster_centers_]
    
    # Group packages by cluster
    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(packages[i])
    
    # Build cluster results
    results = []
    for cluster_id, cluster_packages_list in clusters.items():
        # Calculate cluster metrics
        total_weight = sum(p["weight_kg"] for p in cluster_packages_list)
        num_pkgs = len(cluster_packages_list)
        
        # Count unique addresses as stops
        unique_addresses = set(p["address"] for p in cluster_packages_list)
        num_stops = len(unique_addresses)
        
        results.append(ClusterResult(
            cluster_id=int(cluster_id),
            packages=cluster_packages_list,
            total_weight_kg=total_weight,
            num_packages=num_pkgs,
            num_stops=num_stops,
            centroid=centroids[cluster_id] if cluster_id < len(centroids) else (0, 0),
        ))
    
    # Sort by cluster_id for consistent ordering
    results.sort(key=lambda x: x.cluster_id)
    
    return results


def order_stops_by_nearest_neighbor(
    packages: List[Dict[str, Any]],
    start_lat: float,
    start_lng: float,
) -> List[Dict[str, Any]]:
    """
    Order packages using nearest neighbor heuristic for TSP approximation.
    
    Args:
        packages: List of package dicts with latitude, longitude
        start_lat: Starting latitude (warehouse)
        start_lng: Starting longitude (warehouse)
    
    Returns:
        Ordered list of packages
    """
    if len(packages) <= 1:
        return packages
    
    remaining = packages.copy()
    ordered = []
    current_lat, current_lng = start_lat, start_lng
    
    while remaining:
        # Find nearest package
        min_dist = float("inf")
        nearest_idx = 0
        
        for i, pkg in enumerate(remaining):
            dist = haversine_distance(
                current_lat, current_lng,
                pkg["latitude"], pkg["longitude"],
            )
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # Move to nearest package
        nearest = remaining.pop(nearest_idx)
        ordered.append(nearest)
        current_lat = nearest["latitude"]
        current_lng = nearest["longitude"]
    
    return ordered


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the great circle distance between two points in kilometers.
    """
    from math import radians, cos, sin, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lng = radians(lng2 - lng1)
    
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lng / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c
