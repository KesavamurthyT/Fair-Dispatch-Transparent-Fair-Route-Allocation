
import pytest
import time
import asyncio

@pytest.mark.performance
@pytest.mark.asyncio
async def test_allocation_performance_sla(client, allocation_request, db_session, active_config):
    """50 drivers/routes completes < 30s."""
    start = time.time()
    
    response = await client.post("/api/v1/allocate", json=allocation_request)
    
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 30.0, f"Allocation took {duration:.2f}s (SLA < 30s)"

@pytest.mark.performance
@pytest.mark.asyncio
async def test_agent_timeline_query_performance(client, allocation_request, db_session, active_config):
    """Timeline query should be fast (< 500ms)."""
    # 1. Run allocation
    resp = await client.post("/api/v1/allocate", json=allocation_request)
    run_id = resp.json()["allocation_run_id"]
    
    # 2. Measure timeline query
    start = time.time()
    resp = await client.get(f"/api/v1/admin/agent_timeline?allocation_run_id={run_id}")
    duration = time.time() - start
    
    assert resp.status_code == 200
    assert duration < 0.5, f"Timeline query took {duration:.3f}s"

@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_allocations(client, allocation_request, db_session, active_config):
    """Test concurrent allocation requests."""
    # Testing concurrency with 3 parallel requests
    async def make_request():
        return await client.post("/api/v1/allocate", json=allocation_request)
    
    # Fire 3 requests
    tasks = [make_request() for _ in range(3)]
    results = await asyncio.gather(*tasks)
    
    for res in results:
        assert res.status_code == 200
