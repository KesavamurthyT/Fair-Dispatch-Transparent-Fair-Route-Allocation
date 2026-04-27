"""
History Seeding Service - Seeds mock historical data for drivers.
Creates 50 mock drivers with 7 days of past assignments and feedback.
"""

import random
from datetime import date, timedelta
from uuid import uuid4
from typing import List

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.driver import Driver, DriverFeedback, HardestAspect, PreferredLanguage, VehicleType
from app.models.assignment import Assignment
from app.models.route import Route
from app.models.allocation_run import AllocationRun, AllocationRunStatus


NUM_MOCK_DRIVERS = 50
NUM_DAYS_HISTORY = 7  # 1 week of past data per driver

# Mock driver names (Indian names)
MOCK_NAMES = [
    "Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sunita Reddy", "Vikram Singh",
    "Anita Nair", "Suresh Rao", "Meena Iyer", "Karthik Menon", "Lakshmi Devi",
    "Arun Krishnan", "Pooja Verma", "Ravi Shankar", "Geeta Mishra", "Sanjay Gupta",
    "Nisha Pillai", "Mohan Das", "Rekha Joshi", "Deepak Agarwal", "Kavitha Rajan",
    "Ganesh Murugan", "Divya Shetty", "Prakash Hegde", "Swathi Naidu", "Venkat Rao",
    "Asha Kulkarni", "Rahul Desai", "Sneha Bhat", "Manoj Nambiar", "Shobha Kamath",
    "Ajay Menon", "Bharathi Raja", "Krishna Kumar", "Padma Lakshmi", "Sunil Varma",
    "Radha Krishnan", "Gopal Reddy", "Uma Maheswari", "Naveen Chand", "Jaya Prabha",
    "Ramesh Babu", "Latha Murthy", "Vijay Kumar", "Saritha Nair", "Balaji Srinivas",
    "Kamala Devi", "Harish Rao", "Vasantha Kumari", "Dinesh Patil", "Revathi Iyer"
]

# Hardest aspects for random selection
HARDEST_ASPECTS = list(HardestAspect)


async def seed_mock_history(db: AsyncSession) -> None:
    """
    Seed 50 mock drivers with past assignments + feedback.
    Safe to run once at startup if DB is empty.
    """
    # Check if mock drivers already exist
    result = await db.execute(
        select(func.count(Driver.id)).where(Driver.external_id.like("mock_driver_%"))
    )
    existing_mock_count = result.scalar() or 0
    
    if existing_mock_count >= NUM_MOCK_DRIVERS:
        # Already seeded
        return
    
    today = date.today()
    drivers: List[Driver] = []
    
    # 1) Create 50 mock drivers
    for i in range(NUM_MOCK_DRIVERS):
        # Randomize vehicle type and capacity
        vehicle_type = random.choice([VehicleType.ICE, VehicleType.EV, VehicleType.BICYCLE])
        
        if vehicle_type == VehicleType.BICYCLE:
            capacity = random.uniform(15.0, 30.0)
            battery_range = None
            charging_time = None
        elif vehicle_type == VehicleType.EV:
            capacity = random.uniform(100.0, 200.0)
            battery_range = random.uniform(80.0, 200.0)
            charging_time = random.randint(30, 120)
        else:  # ICE
            capacity = random.uniform(100.0, 250.0)
            battery_range = None
            charging_time = None
        
        driver = Driver(
            id=uuid4(),
            external_id=f"mock_driver_{i+1:03d}",
            name=MOCK_NAMES[i] if i < len(MOCK_NAMES) else f"Mock Driver {i+1}",
            vehicle_capacity_kg=round(capacity, 1),
            vehicle_type=vehicle_type,
            preferred_language=random.choice(list(PreferredLanguage)),
            battery_range_km=battery_range,
            charging_time_minutes=charging_time,
        )
        db.add(driver)
        drivers.append(driver)
    
    await db.flush()  # Get IDs
    
    # 2) For last 7 days, create routes, assignments, feedback
    for day_offset in range(1, NUM_DAYS_HISTORY + 1):
        run_date = today - timedelta(days=day_offset)
        
        # Create allocation run for this day
        alloc_run = AllocationRun(
            id=uuid4(),
            date=run_date,
            num_drivers=NUM_MOCK_DRIVERS,
            num_routes=NUM_MOCK_DRIVERS,
            num_packages=NUM_MOCK_DRIVERS * 5,
            global_gini_index=round(0.20 + random.random() * 0.15, 4),  # 0.20-0.35
            global_std_dev=round(8.0 + random.random() * 8.0, 2),  # 8-16
            global_max_gap=round(15.0 + random.random() * 15.0, 2),  # 15-30
            status=AllocationRunStatus.SUCCESS,
        )
        db.add(alloc_run)
        await db.flush()
        
        for driver in drivers:
            # Create a route for this driver
            num_packages = random.randint(4, 12)
            num_stops = random.randint(3, num_packages)
            total_weight = round(random.uniform(20.0, 80.0), 1)
            difficulty = round(1.0 + random.random() * 2.5, 2)  # 1.0-3.5
            est_time = random.randint(90, 240)  # 1.5-4 hours
            distance = round(random.uniform(15.0, 60.0), 1)
            
            route = Route(
                id=uuid4(),
                date=run_date,
                cluster_id=random.randint(0, 5),
                total_weight_kg=total_weight,
                num_packages=num_packages,
                num_stops=num_stops,
                route_difficulty_score=difficulty,
                estimated_time_minutes=est_time,
                total_distance_km=distance,
                allocation_run_id=alloc_run.id,
            )
            db.add(route)
            await db.flush()
            
            # Compute workload score (same formula as ML Effort Agent)
            workload = (
                1.0 * num_packages +
                0.5 * total_weight +
                10.0 * difficulty +
                0.2 * est_time
            )
            workload = round(workload, 2)
            
            # Fairness score based on how close to average
            avg_workload = 45.0  # Approximate average
            deviation = abs(workload - avg_workload) / avg_workload
            fairness = max(0.5, min(1.0, 1.0 - deviation * 0.5))
            
            assignment = Assignment(
                id=uuid4(),
                date=run_date,
                driver_id=driver.id,
                route_id=route.id,
                workload_score=workload,
                fairness_score=round(fairness, 2),
                allocation_run_id=alloc_run.id,
                explanation=f"Historical assignment for {run_date}",
                driver_explanation=f"You had a route with {num_packages} packages.",
                admin_explanation=f"Standard allocation for {driver.name}.",
            )
            db.add(assignment)
            await db.flush()
            
            # Create feedback (simulate realistic patterns)
            # Higher workload → higher stress, lower fairness rating
            base_stress = 3 + int(workload / 20)  # 3-7 typically
            stress_level = max(1, min(10, base_stress + random.randint(-1, 2)))
            
            base_fairness = 5 - int(workload / 30)  # Higher workload → lower rating
            fairness_rating = max(1, min(5, base_fairness + random.randint(-1, 1)))
            
            tiredness = max(1, min(5, 2 + int(workload / 25) + random.randint(-1, 1)))
            
            feedback = DriverFeedback(
                id=uuid4(),
                driver_id=driver.id,
                assignment_id=assignment.id,
                fairness_rating=fairness_rating,
                stress_level=stress_level,
                tiredness_level=tiredness,
                hardest_aspect=random.choice(HARDEST_ASPECTS),
                comments=f"Day {day_offset} feedback for mock data.",
                route_difficulty_self_report=max(1, min(5, int(difficulty) + random.randint(-1, 1))),
                would_take_similar_route_again=random.random() > 0.3,
            )
            db.add(feedback)
    
    await db.commit()
    print(f"✅ Seeded {NUM_MOCK_DRIVERS} mock drivers with {NUM_DAYS_HISTORY} days of history each.")


async def get_mock_driver_count(db: AsyncSession) -> int:
    """Get the count of mock drivers in the database."""
    result = await db.execute(
        select(func.count(Driver.id)).where(Driver.external_id.like("mock_driver_%"))
    )
    return result.scalar() or 0
