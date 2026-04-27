"""
Workload score calculation service.
Computes route difficulty, estimated time, and overall workload score.
"""

from typing import Union
from dataclasses import dataclass

from app.config import get_settings


@dataclass
class RouteMetrics:
    """Route metrics for workload calculation."""
    num_packages: int
    total_weight_kg: float
    num_stops: int
    route_difficulty_score: float
    estimated_time_minutes: int


def calculate_route_difficulty(
    total_weight_kg: float,
    num_stops: int,
    avg_fragility: float = 1.0,
) -> float:
    """
    Calculate route difficulty score based on weight, stops, and fragility.
    
    Args:
        total_weight_kg: Total weight of packages in kg
        num_stops: Number of delivery stops
        avg_fragility: Average fragility level (1-5)
    
    Returns:
        Route difficulty score (higher = more difficult)
    """
    settings = get_settings()
    
    # Base difficulty
    difficulty = settings.difficulty_base
    
    # Weight contribution
    difficulty += total_weight_kg * settings.difficulty_weight_per_kg
    
    # Stops contribution
    difficulty += num_stops * settings.difficulty_weight_per_stop
    
    # Fragility multiplier (fragility 1 = 1.0x, fragility 5 = 1.4x)
    fragility_multiplier = 1.0 + (avg_fragility - 1) * 0.1
    difficulty *= fragility_multiplier
    
    return round(difficulty, 2)


def estimate_route_time(
    num_packages: int,
    num_stops: int,
    total_distance_km: float = 0.0,
) -> int:
    """
    Estimate route completion time in minutes.
    
    Args:
        num_packages: Number of packages to deliver
        num_stops: Number of delivery stops
        total_distance_km: Total route distance (optional)
    
    Returns:
        Estimated time in minutes
    """
    settings = get_settings()
    
    # Base time for the route
    time = settings.base_route_time
    
    # Time per package (handling, scanning, handover)
    time += num_packages * settings.time_per_package
    
    # Time per stop (parking, navigation, walking)
    time += num_stops * settings.time_per_stop
    
    # Driving time (assume 30 km/h average speed in urban areas)
    if total_distance_km > 0:
        time += (total_distance_km / 30) * 60
    
    return int(round(time))


def calculate_workload(
    route_metrics: Union[RouteMetrics, dict],
    weights: dict = None,
) -> float:
    """
    Calculate the overall workload score for a route.
    
    Formula: workload = a * num_packages + b * total_weight_kg 
                       + c * route_difficulty_score + d * estimated_time_minutes
    
    Args:
        route_metrics: RouteMetrics object or dict with route info
        weights: Optional custom weights dict {a, b, c, d}
    
    Returns:
        Workload score (higher = more work)
    """
    settings = get_settings()
    
    # Default weights from settings
    if weights is None:
        weights = {
            "a": settings.workload_weight_a,
            "b": settings.workload_weight_b,
            "c": settings.workload_weight_c,
            "d": settings.workload_weight_d,
        }
    
    # Extract metrics
    if isinstance(route_metrics, dict):
        num_packages = route_metrics["num_packages"]
        total_weight_kg = route_metrics["total_weight_kg"]
        route_difficulty_score = route_metrics["route_difficulty_score"]
        estimated_time_minutes = route_metrics["estimated_time_minutes"]
    else:
        num_packages = route_metrics.num_packages
        total_weight_kg = route_metrics.total_weight_kg
        route_difficulty_score = route_metrics.route_difficulty_score
        estimated_time_minutes = route_metrics.estimated_time_minutes
    
    # Calculate workload score
    workload = (
        weights["a"] * num_packages +
        weights["b"] * total_weight_kg +
        weights["c"] * route_difficulty_score +
        weights["d"] * estimated_time_minutes
    )
    
    return round(workload, 2)
