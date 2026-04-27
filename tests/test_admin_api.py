
import pytest
from app.models import AllocationRun
from sqlalchemy import select

@pytest.mark.asyncio
async def test_agent_timeline_api(client, allocation_request, db_session, active_config):
    """Phase 5: Verify agent workflow visualization."""
    
    # 1. Run allocation to generate data
    alloc_resp = await client.post("/api/v1/allocate", json=allocation_request)
    assert alloc_resp.status_code == 200
    run_id = alloc_resp.json()["allocation_run_id"]
    
    # 2. Get timeline
    response = await client.get(f"/api/v1/admin/agent_timeline?allocation_run_id={run_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert "timeline" in data
    timeline = data["timeline"]
    
    # 3. Verify content
    assert len(timeline) >= 5 # ML, Route, Fairness, Liaison, Explainability, (Learning)
    
    # Check first step (new format with short_message and details)
    assert timeline[0]["agent_name"] == "ML_EFFORT"
    assert timeline[0]["step_type"] == "MATRIX_GENERATION"
    assert "short_message" in timeline[0]
    assert "details" in timeline[0]
    
    # Check sequence
    agents = [step["agent_name"] for step in timeline]
    assert "ROUTE_PLANNER" in agents
    assert "FAIRNESS_MANAGER" in agents
    assert "EXPLAINABILITY" in agents

@pytest.mark.asyncio
async def test_driver_allocation_story(client, allocation_request, db_session, active_config, sample_drivers):
    """Phase 5: Driver-specific explanation story."""
    
    # 1. Run allocation
    alloc_resp = await client.post("/api/v1/allocate", json=allocation_request)
    assert alloc_resp.status_code == 200
    date_str = allocation_request["allocation_date"]
    
    # 2. Pick a driver who was assigned
    assignments = alloc_resp.json()["assignments"]
    assert len(assignments) > 0
    assigned_driver_id = assignments[0]["driver_id"]
    
    # 3. Get story
    response = await client.get(
        f"/api/v1/admin/driver_allocation_story?driver_id={assigned_driver_id}&date={date_str}"
    )
    assert response.status_code == 200
    
    story = response.json()
    
    # 4. Verify story structure
    assert "today" in story
    assert "driver" in story
    assert str(story["driver"]["id"]) == assigned_driver_id
    assert "history_last_7_days" in story
    assert isinstance(story["history_last_7_days"], list)
    
    # Verify agent timeline slice
    assert "agent_timeline_slice" in story
    slice_steps = story["agent_timeline_slice"]
    assert len(slice_steps) >= 1
    # Check if slice contains breakdown/decision for this driver
    pass

@pytest.mark.asyncio
async def test_fairness_metrics_series(client, db_session):
    """Test fetching fairness metrics over time."""
    from datetime import date, timedelta
    
    # Assuming some runs exist or we create empty response
    today = date.today()
    start = today - timedelta(days=7)
    
    response = await client.get(f"/api/v1/admin/metrics/fairness?start_date={start}&end_date={today}")
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    assert isinstance(data["points"], list)

@pytest.mark.asyncio
async def test_allocation_runs_list(client, allocation_request, db_session, active_config):
    """Test listing allocation runs."""
    # 1. Create a run
    await client.post("/api/v1/allocate", json=allocation_request)
    
    # 2. List runs
    date_str = allocation_request["allocation_date"]
    response = await client.get(f"/api/v1/admin/allocation_runs?date={date_str}")
    assert response.status_code == 200
    
    data = response.json()
    assert "runs" in data
    assert len(data["runs"]) >= 1
    assert data["runs"][0]["status"] == "SUCCESS"
