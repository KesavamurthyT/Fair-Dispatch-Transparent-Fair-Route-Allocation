"""
Unit tests for Phase 5 Agent Workflow Visualization APIs.
Tests agent timeline and driver allocation story endpoints.
"""

import pytest
from datetime import date, datetime, timedelta
from uuid import uuid4, UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Driver, Route, Assignment, AllocationRun, DecisionLog,
    AllocationRunStatus, VehicleType,
)
from app.services.admin_service import (
    get_agent_timeline,
    get_driver_allocation_story,
    _generate_short_message,
    _extract_details,
)
from app.schemas.admin import AgentTimelineResponse, DriverAllocationStoryResponse


class TestAgentTimelineShortMessages:
    """Tests for short message generation."""
    
    def test_ml_effort_message(self):
        """ML_EFFORT should include driver/route counts."""
        log = type('Log', (), {
            'agent_name': 'ML_EFFORT',
            'step_type': 'MATRIX_GENERATION',
            'input_snapshot': {'num_drivers': 50, 'num_routes': 50},
            'output_snapshot': {},
        })()
        
        message = _generate_short_message(log)
        assert "50 drivers" in message
        assert "50 routes" in message
    
    def test_route_planner_proposal(self):
        """ROUTE_PLANNER proposals should describe appropriately."""
        log = type('Log', (), {
            'agent_name': 'ROUTE_PLANNER',
            'step_type': 'PROPOSAL_1',
            'input_snapshot': {},
            'output_snapshot': {},
        })()
        
        message = _generate_short_message(log)
        assert "initial" in message.lower()
    
    def test_route_planner_resolution(self):
        """FINAL_RESOLUTION should include swap count."""
        log = type('Log', (), {
            'agent_name': 'ROUTE_PLANNER',
            'step_type': 'FINAL_RESOLUTION',
            'input_snapshot': {},
            'output_snapshot': {'swaps_applied': 4},
        })()
        
        message = _generate_short_message(log)
        assert "4 swaps" in message
    
    def test_fairness_manager_reoptimize(self):
        """FAIRNESS_MANAGER with REOPTIMIZE status."""
        log = type('Log', (), {
            'agent_name': 'FAIRNESS_MANAGER',
            'step_type': 'FAIRNESS_CHECK_PROPOSAL_1',
            'input_snapshot': {},
            'output_snapshot': {'status': 'REOPTIMIZE'},
        })()
        
        message = _generate_short_message(log)
        assert "re-optimization" in message.lower()
    
    def test_driver_liaison_counts(self):
        """DRIVER_LIAISON should include decision counts."""
        log = type('Log', (), {
            'agent_name': 'DRIVER_LIAISON',
            'step_type': 'NEGOTIATION_DECISIONS',
            'input_snapshot': {},
            'output_snapshot': {'num_accept': 32, 'num_counter': 10, 'num_force_accept': 8},
        })()
        
        message = _generate_short_message(log)
        assert "32 ACCEPT" in message
        assert "10 COUNTER" in message
        assert "8 FORCE_ACCEPT" in message
    
    def test_explainability_categories(self):
        """EXPLAINABILITY should include explanation count."""
        log = type('Log', (), {
            'agent_name': 'EXPLAINABILITY',
            'step_type': 'EXPLANATIONS_GENERATED',
            'input_snapshot': {},
            'output_snapshot': {'total_explanations': 50, 'category_counts': {'NEAR_AVG': 20, 'HEAVY': 10}},
        })()
        
        message = _generate_short_message(log)
        assert "50" in message
        assert "2 categories" in message


class TestExtractDetails:
    """Tests for details extraction from logs."""
    
    def test_extracts_relevant_keys(self):
        """Should extract whitelisted keys from snapshots."""
        log = type('Log', (), {
            'input_snapshot': {'num_drivers': 50, 'irrelevant_key': 'ignored'},
            'output_snapshot': {'gini_index': 0.15, 'std_dev': 12.0, 'also_irrelevant': []},
        })()
        
        details = _extract_details(log)
        
        assert details.get('num_drivers') == 50
        assert details.get('gini_index') == 0.15
        assert details.get('std_dev') == 12.0
        assert 'irrelevant_key' not in details
        assert 'also_irrelevant' not in details
    
    def test_prefers_output_over_input(self):
        """Output snapshot should take precedence."""
        log = type('Log', (), {
            'input_snapshot': {'gini_index': 0.20},
            'output_snapshot': {'gini_index': 0.15},
        })()
        
        details = _extract_details(log)
        assert details['gini_index'] == 0.15


@pytest.fixture
async def test_data(db_session: AsyncSession):
    """Create test data for visualization APIs."""
    # Create driver
    driver = Driver(
        external_id="VIZ-001",
        name="Visualization Test Driver",
        vehicle_type=VehicleType.ICE,
        vehicle_capacity_kg=100.0,
    )
    db_session.add(driver)
    
    # Create route
    route = Route(
        date=date.today(),
        cluster_id=1,
        num_packages=25,
        total_weight_kg=60.0,
        num_stops=15,
        route_difficulty_score=2.5,
        estimated_time_minutes=180,
    )
    db_session.add(route)
    await db_session.flush()
    
    # Create allocation run
    allocation_run = AllocationRun(
        date=date.today(),
        num_drivers=10,
        num_routes=10,
        num_packages=100,
        global_gini_index=0.15,
        global_std_dev=12.0,
        global_max_gap=25.0,
        status=AllocationRunStatus.SUCCESS,
        started_at=datetime.utcnow() - timedelta(minutes=5),
        finished_at=datetime.utcnow(),
    )
    db_session.add(allocation_run)
    await db_session.flush()
    
    # Create assignment
    assignment = Assignment(
        date=date.today(),
        driver_id=driver.id,
        route_id=route.id,
        workload_score=65.0,
        fairness_score=0.85,
        explanation="Test explanation",
        allocation_run_id=allocation_run.id,
    )
    db_session.add(assignment)
    
    # Create decision logs
    logs = [
        DecisionLog(
            allocation_run_id=allocation_run.id,
            agent_name="ML_EFFORT",
            step_type="MATRIX_GENERATION",
            input_snapshot={"num_drivers": 10, "num_routes": 10},
            output_snapshot={"avg_effort": 60.0},
        ),
        DecisionLog(
            allocation_run_id=allocation_run.id,
            agent_name="ROUTE_PLANNER",
            step_type="PROPOSAL_1",
            input_snapshot={},
            output_snapshot={"total_effort": 600.0},
        ),
        DecisionLog(
            allocation_run_id=allocation_run.id,
            agent_name="FAIRNESS_MANAGER",
            step_type="FAIRNESS_CHECK_PROPOSAL_1",
            input_snapshot={},
            output_snapshot={"status": "ACCEPT", "gini_index": 0.15},
        ),
        DecisionLog(
            allocation_run_id=allocation_run.id,
            agent_name="EXPLAINABILITY",
            step_type="EXPLANATIONS_GENERATED",
            input_snapshot={},
            output_snapshot={"total_explanations": 10, "category_counts": {"NEAR_AVG": 8}},
        ),
    ]
    for log in logs:
        db_session.add(log)
    
    await db_session.commit()
    
    return {
        "driver_id": driver.id,
        "route_id": route.id,
        "allocation_run_id": allocation_run.id,
        "assignment_id": assignment.id,
        "date": date.today(),
    }


class TestAgentTimelineEndpoint:
    """Tests for GET /admin/agent_timeline."""
    
    @pytest.mark.asyncio
    async def test_timeline_returns_allocation_run_info(self, db_session: AsyncSession, test_data):
        """Timeline should include allocation run info."""
        result = await get_agent_timeline(db_session, test_data["allocation_run_id"])
        
        assert result.allocation_run.id == test_data["allocation_run_id"]
        assert result.allocation_run.num_drivers == 10
        assert result.allocation_run.num_routes == 10
        assert result.allocation_run.status == "SUCCESS"
        assert "gini_index" in result.allocation_run.global_metrics
    
    @pytest.mark.asyncio
    async def test_timeline_contains_all_logs(self, db_session: AsyncSession, test_data):
        """Timeline should contain all decision logs."""
        result = await get_agent_timeline(db_session, test_data["allocation_run_id"])
        
        assert len(result.timeline) == 4
        agents = [e.agent_name for e in result.timeline]
        assert "ML_EFFORT" in agents
        assert "ROUTE_PLANNER" in agents
        assert "FAIRNESS_MANAGER" in agents
        assert "EXPLAINABILITY" in agents
    
    @pytest.mark.asyncio
    async def test_timeline_events_have_short_messages(self, db_session: AsyncSession, test_data):
        """Each timeline event should have a short_message."""
        result = await get_agent_timeline(db_session, test_data["allocation_run_id"])
        
        for event in result.timeline:
            assert event.short_message
            assert len(event.short_message) > 0
    
    @pytest.mark.asyncio
    async def test_timeline_events_sorted_by_time(self, db_session: AsyncSession, test_data):
        """Events should be sorted by timestamp."""
        result = await get_agent_timeline(db_session, test_data["allocation_run_id"])
        
        timestamps = [e.timestamp for e in result.timeline]
        assert timestamps == sorted(timestamps)
    
    @pytest.mark.asyncio
    async def test_nonexistent_run_returns_empty(self, db_session: AsyncSession):
        """Non-existent allocation run should return empty timeline."""
        fake_id = uuid4()
        result = await get_agent_timeline(db_session, fake_id)
        
        assert result.allocation_run.status == "NOT_FOUND"
        assert len(result.timeline) == 0


class TestDriverAllocationStoryEndpoint:
    """Tests for GET /admin/driver_allocation_story."""
    
    @pytest.mark.asyncio
    async def test_story_returns_driver_info(self, db_session: AsyncSession, test_data):
        """Story should include driver info."""
        result = await get_driver_allocation_story(

            db_session, test_data["driver_id"], test_data["date"]
        )
        
        assert result is not None
        assert result.driver.id == test_data["driver_id"]
        assert result.driver.name == "Visualization Test Driver"
    
    @pytest.mark.asyncio
    async def test_story_returns_today_info(self, db_session: AsyncSession, test_data):
        """Story should include today's assignment info."""
        result = await get_driver_allocation_story(

            db_session, test_data["driver_id"], test_data["date"]
        )
        
        assert result.today.assignment_id == test_data["assignment_id"]
        assert result.today.route.id == test_data["route_id"]
        assert result.today.effort.value == 65.0
        assert result.today.fairness_score == 0.85
    
    @pytest.mark.asyncio
    async def test_story_returns_global_metrics(self, db_session: AsyncSession, test_data):
        """Story should include allocation run metrics."""
        result = await get_driver_allocation_story(

            db_session, test_data["driver_id"], test_data["date"]
        )
        
        assert result.allocation_run.id == test_data["allocation_run_id"]
        assert result.allocation_run.global_metrics.gini_index == 0.15
        assert result.allocation_run.global_metrics.std_dev == 12.0
    
    @pytest.mark.asyncio
    async def test_story_includes_timeline_slice(self, db_session: AsyncSession, test_data):
        """Story should include agent timeline slice."""
        result = await get_driver_allocation_story(

            db_session, test_data["driver_id"], test_data["date"]
        )
        
        assert len(result.agent_timeline_slice) >= 1
        for event in result.agent_timeline_slice:
            assert event.agent_name
            assert event.description
    
    @pytest.mark.asyncio
    async def test_story_returns_none_for_no_assignment(self, db_session: AsyncSession, test_data):
        """Should return None when no assignment exists."""
        fake_driver_id = uuid4()
        result = await get_driver_allocation_story(

            db_session, fake_driver_id, test_data["date"]
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_story_returns_none_for_wrong_date(self, db_session: AsyncSession, test_data):
        """Should return None when no assignment for date."""
        wrong_date = date.today() - timedelta(days=30)
        result = await get_driver_allocation_story(

            db_session, test_data["driver_id"], wrong_date
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_story_includes_recovery_info(self, db_session: AsyncSession, test_data):
        """Story should include recovery information."""
        result = await get_driver_allocation_story(

            db_session, test_data["driver_id"], test_data["date"]
        )
        
        assert hasattr(result.recovery, 'is_recovery_day')
        assert hasattr(result.recovery, 'recent_hard_days')
    
    @pytest.mark.asyncio
    async def test_story_includes_negotiation_info(self, db_session: AsyncSession, test_data):
        """Story should include negotiation information."""
        result = await get_driver_allocation_story(

            db_session, test_data["driver_id"], test_data["date"]
        )
        
        assert hasattr(result.negotiation, 'swap_applied')
        assert hasattr(result.negotiation, 'manual_override')
        assert hasattr(result.negotiation.manual_override, 'affected')
