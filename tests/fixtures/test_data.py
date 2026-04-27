"""
Test data generators for creating realistic test scenarios.
"""

import uuid
import random
from datetime import date, datetime, timedelta
import numpy as np
from faker import Faker

fake = Faker()

def generate_drivers(count: int = 50, ev_ratio: float = 0.2) -> list:
    """
    Generate realistic driver data.
    
    Args:
        count: Number of drivers to generate
        ev_ratio: Percentage of drivers that are EV (0.0 to 1.0)
        
    Returns:
        List of dictionaries containing driver data
    """
    drivers = []
    
    for i in range(count):
        is_ev = i < (count * ev_ratio)
        driver_id = f"drv_{i:03d}"
        
        drivers.append({
            "id": driver_id,
            "name": fake.name(),
            "vehicle_capacity_kg": 100.0 if is_ev else 150.0,
            "preferred_language": random.choice(["en", "ta", "hi", "te", "kn"]),
            # Internal fields (not in API input but useful for model creation)
            "gameified_experience_points": random.randint(100, 5000),
            "is_ev": is_ev,
            "battery_range_km": 120.0 if is_ev else None,
            "charging_time_minutes": 45 if is_ev else None,
            "experience_years": np.random.uniform(0.5, 5.0),
            "complexity_debt": np.random.uniform(0, 3.0)
        })
        
    # Shuffle so EVs are distributed
    random.shuffle(drivers)
    return drivers

def generate_routes(count: int = 50, difficulty_levels: list = [1.0, 1.5, 2.0, 2.5, 3.0]) -> list:
    """
    Generate realistic route data.
    
    Args:
        count: Number of routes to generate
        difficulty_levels: List of difficulty multipliers
        
    Returns:
        List of dictionaries containing route data
    """
    routes = []
    
    for i in range(count):
        route_id = f"rt_{i:03d}"
        diff = difficulty_levels[i % len(difficulty_levels)]
        
        routes.append({
            "route_id": route_id,
            "cluster_id": i,
            "stops": int(8 + diff * 4),
            "total_weight_kg": 40 + diff * 20,
            "total_distance_km": 35 + diff * 15,
            "parking_difficulty": diff,
            "traffic_complexity": diff,
            "estimated_time_minutes": int((35 + diff * 15) * 4)  # Rough estimate
        })
        
    return routes

def generate_packages(count: int = 50, center_lat: float = 12.9716, center_lng: float = 77.5946) -> list:
    """
    Generate realistic package data.
    
    Args:
        count: Number of packages
        center_lat: Warehouse latitude
        center_lng: Warehouse longitude
        
    Returns:
        List of dictionaries valid for PackageInput schema
    """
    packages = []
    
    for i in range(count):
        # Generate random location within ~10km of center
        lat_offset = random.uniform(-0.1, 0.1)
        lng_offset = random.uniform(-0.1, 0.1)
        
        packages.append({
            "id": f"pkg_{i:04d}",
            "weight_kg": round(random.uniform(0.5, 15.0), 1),
            "fragility_level": random.randint(1, 5),
            "address": fake.address(),
            "latitude": center_lat + lat_offset,
            "longitude": center_lng + lng_offset,
            "priority": random.choice(["NORMAL", "NORMAL", "NORMAL", "HIGH", "EXPRESS"])
        })
        
    return packages

def generate_allocation_request(drivers: list, routes: list, date_str: str = None) -> dict:
    """
    Generate a complete allocation request payload.
    Note: This transforms the internal route/driver structures into the API schema format.
    
    Args:
        drivers: List of driver dicts from generate_drivers
        routes: List of route dicts from generate_routes
        date_str: Date string for allocation (YYYY-MM-DD)
        
    Returns:
        Dictionary matching AllocationRequest schema
    """
    if not date_str:
        date_str = date.today().isoformat()
        
    # Create packages from routes (reverse engineer helpful for consistent tests)
    # Or just generate fresh packages if routes are just for reference
    # For full E2E, we usually provide packages and let the system cluster them.
    # However, if we want to force specific route characteristics, we might need
    # to carefully craft packages.
    # For now, let's just generate a bunch of packages.
    
    total_packages_needed = sum(r["stops"] * 2 for r in routes) # Approx 2 packages per stop
    packages = generate_packages(total_packages_needed)
    
    # Filter drivers for API input (remove internal fields)
    api_drivers = []
    for d in drivers:
        api_drivers.append({
            "id": d["id"],
            "name": d["name"],
            "vehicle_capacity_kg": d["vehicle_capacity_kg"],
            "preferred_language": d["preferred_language"]
        })
        
    return {
        "allocation_date": date_str,
        "warehouse": {"lat": 12.9716, "lng": 77.5946},
        "packages": packages,
        "drivers": api_drivers
    }
