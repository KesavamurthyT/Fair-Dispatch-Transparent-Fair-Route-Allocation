"""
Unit tests for the async allocation API endpoints.

Tests the new HTTP 202 job queue flow and the deprecated sync endpoint.
Uses mocked Celery to avoid needing a running Redis/worker.
"""

import pytest
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import date

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models.allocation_run import AllocationRun, AllocationRunStatus


# === Fixtures ===

@pytest.fixture
def sample_allocation_payload():
    """Minimal valid allocation request payload."""
    return {
        "allocation_date": str(date.today()),
        "drivers": [
            {
                "id": f"D{i:03d}",
                "name": f"Driver {i}",
                "vehicle_capacity_kg": 100.0,
                "preferred_language": "en",
            }
            for i in range(3)
        ],
        "packages": [
            {
                "id": f"P{j:03d}",
                "weight_kg": 5.0,
                "fragility_level": 1,
                "address": f"{j} Test Street",
                "latitude": 12.97 + j * 0.01,
                "longitude": 77.59 + j * 0.01,
                "priority": "standard",
            }
            for j in range(6)
        ],
        "warehouse": {"lat": 12.97, "lng": 77.59},
    }


# === Tests ===

class TestAsyncAllocationEndpoint:
    """Test the new async POST /allocate/langgraph endpoint."""

    @pytest.mark.asyncio
    async def test_returns_202_with_job_id(self, client, sample_allocation_payload):
        """POST should return 202 Accepted with a job_id."""
        with patch("app.api.allocation_langgraph.run_allocation_job") as mock_task:
            mock_task.apply_async = MagicMock()

            response = await client.post(
                "/api/v1/allocate/langgraph",
                json=sample_allocation_payload,
            )

            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "queued"
            assert "message" in data

    @pytest.mark.asyncio
    async def test_empty_packages_returns_400(self, client):
        """Empty packages list should return 400."""
        response = await client.post(
            "/api/v1/allocate/langgraph",
            json={
                "allocation_date": str(date.today()),
                "drivers": [{"id": "D1", "name": "A", "vehicle_capacity_kg": 100, "preferred_language": "en"}],
                "packages": [],
                "warehouse": {"lat": 12.97, "lng": 77.59},
            },
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_drivers_returns_400(self, client):
        """Empty drivers list should return 400."""
        response = await client.post(
            "/api/v1/allocate/langgraph",
            json={
                "allocation_date": str(date.today()),
                "drivers": [],
                "packages": [{"id": "P1", "weight_kg": 5, "fragility_level": 1, "address": "x", "latitude": 12.97, "longitude": 77.59, "priority": "standard"}],
                "warehouse": {"lat": 12.97, "lng": 77.59},
            },
        )
        assert response.status_code == 400


class TestJobStatusEndpoint:
    """Test GET /allocate/status/{job_id} endpoint."""

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, client):
        """Non-existent job ID should return 404."""
        fake_id = str(uuid.uuid4())
        response = await client.get(f"/api/v1/allocate/status/{fake_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_400(self, client):
        """Invalid job ID format should return 400."""
        response = await client.get("/api/v1/allocate/status/not-a-uuid")
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_queued_status(self, client, db_session, sample_allocation_payload):
        """Should return QUEUED status for a newly created run."""
        run = AllocationRun(
            date=date.today(),
            num_drivers=3,
            num_packages=6,
            num_routes=0,
            status=AllocationRunStatus.QUEUED,
        )
        db_session.add(run)
        await db_session.flush()

        response = await client.get(f"/api/v1/allocate/status/{run.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "QUEUED"


class TestDeprecatedSyncEndpoint:
    """Test the deprecated POST /allocate/langgraph_sync endpoint."""

    @pytest.mark.asyncio
    async def test_sync_endpoint_exists(self, client):
        """Sync endpoint should exist (even if it fails due to validation)."""
        response = await client.post(
            "/api/v1/allocate/langgraph_sync",
            json={},
        )
        # Should get 422 (validation error) not 404
        assert response.status_code in (400, 422, 410)

    @pytest.mark.asyncio
    async def test_sync_disabled_returns_410(self, client):
        """When feature flag is off, sync should return 410 Gone."""
        with patch("app.api.allocation_langgraph.settings") as mock_settings:
            mock_settings.sync_allocation_enabled = False

            response = await client.post(
                "/api/v1/allocate/langgraph_sync",
                json={
                    "allocation_date": str(date.today()),
                    "drivers": [{"id": "D1", "name": "A", "vehicle_capacity_kg": 100, "preferred_language": "en"}],
                    "packages": [{"id": "P1", "weight_kg": 5, "fragility_level": 1, "address": "x", "latitude": 12.97, "longitude": 77.59, "priority": "standard"}],
                    "warehouse": {"lat": 12.97, "lng": 77.59},
                },
            )
            assert response.status_code == 410
