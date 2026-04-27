"""
Live Monitor Service.
Polls the database for the latest allocation run and adapts it for the dashboard.
"""

import sys
import os
import asyncio
from sqlalchemy import select
from sqlalchemy.orm import selectinload

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database import async_session_maker
from app.models import AllocationRun, Route, Assignment, Driver, Package
from solver import Solution

async def get_latest_allocation_run(session):
    """Fetch the most recent allocation run."""
    result = await session.execute(
        select(AllocationRun)
        .order_by(AllocationRun.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()

async def fetch_full_run_data(run_id):
    """
    Fetch all data for a specific run and build a Solution object.
    """
    async with async_session_maker() as session:
        # Load run with related data might be complex due to async lazy loading
        # So we fetch explicitly
        
        # 1. Fetch Routes with Packages
        routes_query = (
            select(Route)
            .where(Route.assignments.any(Assignment.allocation_run_id == run_id))
            .options(
                selectinload(Route.route_packages).selectinload("package"),
                selectinload(Route.assignments).selectinload("driver")
            )
        )
        
        # Note: AllocationRun links to Assignments, Assignments link to Routes.
        # But Route doesn't have a direct link to AllocationRun in the model shown?
        # Wait, AllocationRun has ID. Assignment has allocation_run_id.
        # So we query Assignments first? Or Routes via Assignments?
        # Let's query Routes that have assignments belonging to this run.
        
        result = await session.execute(routes_query)
        routes_db = result.scalars().all()
        
        # 2. Build Solution Object
        solution = Solution()
        solution.total_distance = sum(r.total_distance_km or 0 for r in routes_db) * 1000 # Convert km to meters
        
        # We need to reconstruct routes: DriverID -> List[(lat, lon)]
        # And assignments: DriverID -> List[PackageID]
        
        for route in routes_db:
            # Each route should have one assignment usually, but model allows list?
            # Let's assume 1 active assignment per route for this run
            # actually Assignment has 'allocation_run_id'.
            # A route is created FOR a run.
            
            # Find the driver for this route in this run
            # The route might index assignments.
            
            # Let's look at the assignments on this route that match our run_id
            relevant_assignments = [
                a for a in route.assignments 
                if a.allocation_run_id == run_id
            ]
            
            if not relevant_assignments:
                continue
                
            driver = relevant_assignments[0].driver
            
            # Build Package list in order
            # RoutePackage has stop_order
            sorted_rps = sorted(route.route_packages, key=lambda rp: rp.stop_order)
            
            path_coords = []
            pkg_ids = []
            
            # Warehouse start (assume first point or implicit?)
            # The route packages are the stops.
            # Visualizer needs coordinates.
            
            for rp in sorted_rps:
                pkg = rp.package
                path_coords.append((pkg.latitude, pkg.longitude))
                pkg_ids.append(pkg.external_id)
                
            solution.routes[driver.id] = path_coords # This is simple point-to-point. 
            # Real pathing requires OSRM/Graph, but for "Monitor", straight lines 
            # or verifying against the graph in dashboard is ok. 
            # If we want the real path, we might need to re-calc it using the dashboard's graph 
            # based on these stops.
            
            solution.assignments[driver.id] = pkg_ids
            
        return solution

def get_latest_solution_sync():
    """Wrapper to run async code synchronously for Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def task():
        async with async_session_maker() as session:
            latest_run = await get_latest_allocation_run(session)
            if not latest_run:
                return None, None
            
            sol = await fetch_full_run_data(latest_run.id)
            return latest_run, sol
            
    return loop.run_until_complete(task())
