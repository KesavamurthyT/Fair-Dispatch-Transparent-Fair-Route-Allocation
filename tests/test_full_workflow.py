
import pytest
from sqlalchemy import select
from app.models import AllocationRun, Assignment, DecisionLog, AllocationRunStatus

@pytest.mark.asyncio
async def test_full_allocation_workflow(client, allocation_request, db_session, active_config):
    """
    E2E: POST /allocate -> verify agents -> check outputs
    """
    # 1. Run allocation
    response = await client.post("/api/v1/allocate", json=allocation_request)
    assert response.status_code == 200, f"Allocation failed: {response.text}"
    
    data = response.json()
    assert "allocation_run_id" in data
    assert "global_fairness" in data
    # assignments might be fewer than requested drivers if not enough packages or other logic, 
    # but our generator matches counts.
    assert "assignments" in data
    assert len(data["assignments"]) == 50 
    
    # 2. Verify AllocationRun created
    result = await db_session.execute(select(AllocationRun))
    runs = result.scalars().all()
    assert len(runs) == 1
    run = runs[0]
    
    assert run.status == AllocationRunStatus.SUCCESS
    assert run.num_drivers == 50
    assert run.finished_at is not None
    
    # 3. Verify DecisionLogs (5 agents fired)
    logs_result = await db_session.execute(
        select(DecisionLog)
        .where(DecisionLog.allocation_run_id == run.id)
        .order_by(DecisionLog.created_at)
    )
    logs = logs_result.scalars().all()
    agent_steps = [log.agent_name for log in logs]
    
    # Note: LEARNING agent is also expected in Phase 8
    expected_agents = ["ML_EFFORT", "ROUTE_PLANNER", "FAIRNESS_MANAGER", 
                      "DRIVER_LIAISON", "EXPLAINABILITY"]
    
    for agent in expected_agents:
        assert agent in agent_steps, f"Missing agent: {agent}"
        
    # 4. Verify Assignments created
    assign_result = await db_session.execute(
        select(Assignment).where(Assignment.allocation_run_id == run.id)
    )
    assignments = assign_result.scalars().all()
    assert len(assignments) == 50
    
    # 5. Verify fairness metrics
    fairness = data["global_fairness"]
    # Relax checks slightly for random test data
    assert fairness["gini_index"] < 0.45 
    assert fairness["std_dev"] < 35.0

@pytest.mark.asyncio
async def test_assignments_have_explanations(client, allocation_request, active_config):
    """Verify that all assignments have explanations."""
    response = await client.post("/api/v1/allocate", json=allocation_request)
    assert response.status_code == 200
    data = response.json()
    
    for assignment in data["assignments"]:
        assert assignment["explanation"] is not None
        assert len(assignment["explanation"]) > 10
        
@pytest.mark.asyncio
async def test_allocation_run_persistence(client, allocation_request, db_session, active_config):
    """Verify detailed persistence of allocation run."""
    response = await client.post("/api/v1/allocate", json=allocation_request)
    run_id = response.json()["allocation_run_id"]
    
    result = await db_session.execute(
        select(AllocationRun).where(AllocationRun.id == run_id)
    )
    run = result.scalar_one()
    
    assert run.global_gini_index is not None
    assert run.global_std_dev is not None
    assert run.num_packages > 0
    assert run.num_routes > 0
