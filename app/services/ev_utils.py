"""
EV Utilities for Phase 7.
Provides EV feasibility checks and charging overhead calculations.
"""

from typing import Optional


def is_route_feasible_for_ev(
    battery_range_km: float,
    route_distance_km: float,
    safety_margin_pct: float = 10.0,
) -> bool:
    """
    Check if a route is feasible for an EV driver.
    
    Args:
        battery_range_km: Driver's EV battery range in km
        route_distance_km: Total route distance in km
        safety_margin_pct: Safety margin percentage (default 10%)
    
    Returns:
        True if the route is within EV range with safety margin
    """
    if battery_range_km <= 0:
        return False
    if route_distance_km is None or route_distance_km <= 0:
        return True  # No distance info, assume feasible
    
    effective_range = battery_range_km * (1.0 - safety_margin_pct / 100.0)
    return route_distance_km <= effective_range


def calculate_ev_charging_overhead(
    route_distance_km: float,
    battery_range_km: float,
    charging_time_minutes: int,
    penalty_weight: float = 0.3,
) -> float:
    """
    Calculate effort overhead for EV drivers due to range/charging constraints.
    
    Penalizes routes that use a high percentage of battery,
    reflecting the mental/logistical burden of range management.
    
    Args:
        route_distance_km: Total route distance
        battery_range_km: EV battery range
        charging_time_minutes: Typical charging time
        penalty_weight: Weight for penalty (default 0.3)
    
    Returns:
        Additional effort score for EV charging overhead
    """
    if battery_range_km <= 0 or route_distance_km <= 0:
        return 0.0
    
    # Calculate usage ratio (0.0 to 1.0+)
    usage_ratio = route_distance_km / battery_range_km
    
    # Only penalize when usage exceeds 70% of range
    if usage_ratio <= 0.7:
        return 0.0
    
    # Overhead increases as usage approaches or exceeds 100%
    # More burden = more mental/planning effort
    charging_overhead = (usage_ratio - 0.7) * charging_time_minutes * penalty_weight
    
    return max(0.0, charging_overhead)


def get_ev_effort_adjustment(
    driver_is_ev: bool,
    battery_range_km: Optional[float],
    charging_time_minutes: Optional[int],
    route_distance_km: Optional[float],
    safety_margin_pct: float = 10.0,
    penalty_weight: float = 0.3,
) -> tuple[bool, float]:
    """
    Get EV feasibility and effort adjustment for a driver-route pair.
    
    Args:
        driver_is_ev: Whether driver uses EV
        battery_range_km: EV battery range (None for ICE)
        charging_time_minutes: EV charging time (None for ICE)
        route_distance_km: Route distance
        safety_margin_pct: Safety margin for range check
        penalty_weight: Weight for charging overhead
    
    Returns:
        Tuple of (is_feasible, effort_adjustment)
    """
    if not driver_is_ev:
        return (True, 0.0)
    
    if battery_range_km is None or battery_range_km <= 0:
        # EV without range info - assume feasible, no adjustment
        return (True, 0.0)
    
    if route_distance_km is None:
        # No distance info - assume feasible, no adjustment
        return (True, 0.0)
    
    # Check feasibility
    feasible = is_route_feasible_for_ev(
        battery_range_km, route_distance_km, safety_margin_pct
    )
    
    if not feasible:
        return (False, float('inf'))
    
    # Calculate charging overhead
    overhead = calculate_ev_charging_overhead(
        route_distance_km,
        battery_range_km,
        charging_time_minutes or 30,
        penalty_weight,
    )
    
    return (True, overhead)
