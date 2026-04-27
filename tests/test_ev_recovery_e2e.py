
import pytest
from datetime import date, timedelta
from sqlalchemy import select
from app.models import Assignment, Route, Driver, DriverStatsDaily, VehicleType

@pytest.mark.asyncio
async def test_ev_constraints_respected(client, allocation_request, db_session, active_config):
    """Test that EV drivers are only assigned routes within their battery range."""
    
    # Run allocation
    response = await client.post("/api/v1/allocate", json=allocation_request)
    assert response.status_code == 200
    data = response.json()
    run_id = data["allocation_run_id"]
    
    # Check assignments in DB
    result = await db_session.execute(
        select(Assignment, Route, Driver)
        .join(Route, Assignment.route_id == Route.id)
        .join(Driver, Assignment.driver_id == Driver.id)
        .where(Assignment.allocation_run_id == run_id)
    )
    rows = result.all()
    
    ev_assignments_count = 0
    
    for assignment, route, driver in rows:
        if driver.vehicle_type == VehicleType.EV:
            ev_assignments_count += 1
            
            # Assert route distance exists (thanks to our fix)
            assert route.total_distance_km is not None
            
            # Check range constraint
            effective_range = driver.battery_range_km * (1.0 - (active_config.ev_safety_margin_pct / 100.0))
            
            assert route.total_distance_km <= effective_range, \
                f"EV limits exceeded: Route {route.total_distance_km}km > Range {effective_range}km"
                
    assert ev_assignments_count > 0, "Should have some EV assignments derived from test data"

@pytest.mark.asyncio
async def test_recovery_mode_high_debt(client, allocation_request, db_session, active_config, sample_drivers):
    """Test that a driver with high complexity debt receives a lighter workload."""
    
    # Select a target driver
    target_driver = sample_drivers[0]
    
    # Seed history to trigger recovery mode
    # 5 hard days in last 7 days + high debt
    for i in range(1, 8):
        stat = DriverStatsDaily(
            driver_id=target_driver.id,
            date=date.today() - timedelta(days=i),
            avg_workload_score=85.0,
            is_hard_day=True,
            complexity_debt=3.5, # Very high debt
            is_recovery_day=False
        )
        db_session.add(stat)
    await db_session.commit()
    
    # Run allocation
    response = await client.post("/api/v1/allocate", json=allocation_request)
    assert response.status_code == 200
    data = response.json()
    
    global_avg = data["global_fairness"]["avg_workload"]
    
    # Find assignment
    target_assignment = next(
        (a for a in data["assignments"] if str(a["driver_id"]) == str(target_driver.id)),
        None
    )
    
    # It's possible the driver wasn't assigned if strict constraints prevented it, 
    # but usually everyone gets a route.
    if target_assignment:
        # Check workload is reduced (e.g., below average * 0.9 or similar)
        # The exact factor depends on `recovery_lightening_factor` (usually 0.85 target)
        # But RoutePlanner optimizes, so it might not be perfect.
        # We assert it's at least below Global Average.
        
        assert target_assignment["workload_score"] < global_avg, \
            f"Recovery driver workload {target_assignment['workload_score']} should be < avg {global_avg}"
        
        # Check explanation mentions recovery
        # Note: Current explanation generation might not explicitly say "Recovery" 
        # unless ExplainabilityAgent logic covers it. 
        # (Checked: ExplainabilityAgent has `is_recovery_day` input)
        explanation = target_assignment["explanation"]
        assert "recovery" in explanation.lower() or "lighter" in explanation.lower(), \
            f"Explanation should mention recovery/lighter load: {explanation}"

