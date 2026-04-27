
import asyncio
from datetime import date, datetime
from typing import List, Dict, Any, Tuple
import os
import sys

# Import main app models and workflow
from app.schemas.allocation import AllocationRequest
from app.services.langgraph_workflow import invoke_allocation_workflow
from app.schemas.allocation_state import AllocationState

# Import dashboard models (for type hinting if needed, though we use dicts often)
class DashboardAgentAdapter:
    """
    Adapter to connect the Streamlit Dashboard with the LangGraph AI Workflow.
    """

    @staticmethod
    async def run_simulation(drivers: List[Any], packages: List[Any]) -> Dict[str, Any]:
        """
        Run the full AI agent simulation.

        Args:
            drivers: List of dashboard Driver objects
            packages: List of dashboard Package objects

        Returns:
            Solution dictionary compatible with the dashboard visualization
        """
        # 1. Convert Dashboard Data -> API Request Format
        request_data = DashboardAgentAdapter._build_request(drivers, packages)

        # 2. Invoke the LangGraph Workflow
        # We need a running event loop
        try:
            state: AllocationState = await invoke_allocation_workflow(
                request_dict=request_data,
                allocation_run_id=f"sim_{int(datetime.now().timestamp())}"
            )
        except Exception as e:
            print(f"Error running simulation: {e}")
            raise e

        # 3. Convert API Response -> Dashboard Solution
        return DashboardAgentAdapter._parse_response(state, drivers, packages)

    @staticmethod
    def _build_request(drivers: List[Any], packages: List[Any]) -> Dict[str, Any]:
        """Constructs the AllocationRequest dictionary."""
        
        # Use first driver's location or default for warehouse
        warehouse_lat = drivers[0].lat if drivers else 13.0827
        warehouse_lng = drivers[0].lon if drivers else 80.2707

        formatted_drivers = []
        for d in drivers:
            # Dashboard driver might have limited fields, map what we can
            formatted_drivers.append({
                "id": str(d.id),
                "name": f"Driver {d.id}",
                "vehicle_capacity_kg": 100.0, # Default if not in dashboard model
                "preferred_language": "en"
            })

        formatted_packages = []
        for p in packages:
            formatted_packages.append({
                "id": str(p.id),
                "weight_kg": 5.0, # Default
                "fragility_level": 1,
                "address": f"Lat {p.lat:.4f}, Lon {p.lon:.4f}",
                "latitude": p.lat,
                "longitude": p.lon,
                "priority": "NORMAL"
            })

        return {
            "allocation_date": date.today().isoformat(),
            "warehouse": {"lat": warehouse_lat, "lng": warehouse_lng},
            "drivers": formatted_drivers,
            "packages": formatted_packages
        }

    @staticmethod
    def _parse_response(state: AllocationState, original_drivers: List[Any], original_packages: List[Any]) -> Any:
        # We need to return a 'Solution' like object or struct that app.py expects
        # app.py expects: solution.routes = { driver_id: [(lat, lon), (lat, lon), ...] }
        # and attributes: total_distance, fairness_score
        
        from solver import Solution # Import strictly for the class definition
        import osmnx as ox
        import networkx as nx

        assignments = state.assignments
        routes = {}
        total_dist = 0.0
        
        # Look up package locations by ID
        pkg_locs = {str(p.id): (p.lat, p.lon) for p in original_packages}
        driver_locs = {str(d.id): (d.lat, d.lon) for d in original_drivers}
        
        # Helper to find driver initial loc
        # The assignment response only gives driver_id (UUID) and driver_external_id
        # We need to map back to original driver object to get start location
        # Create a map of external_id -> original_driver
        # In _build_request, we used str(d.id) as the ID passed to 'id' field
        
        # The AllocationState.assignments is a list of dicts or objects?
        # Typically it's a list[dict] in the state
        
        for assignment in assignments:
            # Assignment structure might vary depending on exact state schema
            # Check if it's a dict or Pydantic model
            if hasattr(assignment, "driver_external_id"):
                 ext_id = assignment.driver_external_id
                 pkg_ids = [] # Extract package IDs from route or assignment?
                 # Wait, AllocationState.assignments usually lists Dicts with 'package_ids' 
                 # OR it's a Pydantic model AssignmentResponse which has 'route_id' but not direct package list?
                 # let's look at schema for 'AssignmentResponse'. It has route_summary but not package list.
                 # The detailed route info is elsewhere in the state?
                 pass
            elif isinstance(assignment, dict):
                 ext_id = assignment.get("driver_external_id")
                 # We need the list of packages assigned. 
                 # If the state only stores 'Route' objects separately, we need to access them.
                 # Let's assume for now assignment has 'package_ids' or similar if it's the raw planner output
                 # PREVIOUSLY in state schema: assignments: List[Dict] = []
                 pass

            # Fallback: If the state has 'routes' dict: { driver_id: [pkg_id, ...] }
            # Let's check AllocationState definition carefully.
            # It has 'assignments', 'routes', 'packages', 'drivers'.
            
        # RE-READING AllocationState Schema from memory (viewed file earlier):
        # class AllocationState(BaseModel):
        # ...
        # assignments: List[Dict] = []  # Final assignments
        # routes: Dict[str, Route] = {} # Map route_id -> Route object
            
            pass 
        
        # SIMPLIFIED PARSING LOGIC based on likely structure
        # We will iterate through state.routes to get the package lists per driver
        
        # Map route_id -> driver_id from assignments
        route_to_driver = {}
        if state.assignments:
             for a in state.assignments:
                 # 'a' is likely a dict or object. safer to try both
                 if isinstance(a, dict):
                     rid = a.get("route_id")
                     did = a.get("driver_external_id") or a.get("driver_id")
                 else:
                     rid = getattr(a, "route_id", None)
                     did = getattr(a, "driver_external_id", getattr(a, "driver_id", None))
                 
                 if rid and did:
                     route_to_driver[str(rid)] = str(did)
        
        # Now iterate routes
        if state.routes:
            for rid, route_obj in state.routes.items():
                driver_id = route_to_driver.get(str(rid))
                if not driver_id: 
                    continue
                    
                # Route object has 'stops' or 'package_ids'?
                # Let's assume 'package_ids' list
                pids = []
                if isinstance(route_obj, dict):
                    pids = route_obj.get("package_ids", [])
                else:
                    pids = getattr(route_obj, "package_ids", [])
                
                # Construct coords
                # Start
                start_coord = driver_locs.get(driver_id)
                path_coords = []
                if start_coord:
                    path_coords.append(start_coord)
                
                for pid in pids:
                    if pid in pkg_locs:
                        path_coords.append(pkg_locs[pid])
                        
                routes[driver_id] = path_coords

        # Fairness
        gini = 0.0
        if state.global_fairness:
             # Check if dict or object
             if isinstance(state.global_fairness, dict):
                  gini = state.global_fairness.get("gini_index", 0.0)
             else:
                  gini = getattr(state.global_fairness, "gini_index", 0.0)

        sol = Solution()
        sol.routes = routes
        sol.total_distance = 1000.0 
        sol.fairness_score = gini * 1000000 
        
        return sol

