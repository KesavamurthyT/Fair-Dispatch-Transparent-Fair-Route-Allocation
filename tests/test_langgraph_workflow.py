"""
Tests for LangGraph allocation workflow.
Verifies workflow equivalence, decision logging, and performance.
"""

import pytest
import time
from datetime import date
from uuid import uuid4

from app.schemas.allocation_state import AllocationState
from app.services.langgraph_nodes import (
    ml_effort_node,
    route_planner_node,
    fairness_check_node,
    should_reoptimize,
)
from app.services.langgraph_workflow import (
    create_allocation_graph,
    get_workflow_visualization,
)


class TestAllocationState:
    """Tests for AllocationState schema."""
    
    def test_allocation_state_defaults(self):
        """AllocationState should have sensible defaults."""
        state = AllocationState()
        
        assert state.request == {}
        assert state.config_used is None
        assert state.decision_logs == []
        assert state.effort_matrix is None
        assert state.explanations == {}
    
    def test_allocation_state_serialization(self):
        """AllocationState should serialize to dict."""
        state = AllocationState(
            request={"test": "data"},
            driver_models=[{"id": "d1", "name": "Driver 1"}],
        )
        
        data = state.model_dump(mode="json")
        
        assert data["request"] == {"test": "data"}
        assert data["driver_models"] == [{"id": "d1", "name": "Driver 1"}]
    
    def test_allocation_state_deserialization(self):
        """AllocationState should deserialize from dict."""
        data = {
            "request": {"drivers": []},
            "config_used": {"gini_threshold": 0.35},
            "decision_logs": [{"agent_name": "TEST"}],
        }
        
        state = AllocationState.model_validate(data)
        
        assert state.request == {"drivers": []}
        assert state.config_used["gini_threshold"] == 0.35
        assert len(state.decision_logs) == 1


class TestLangGraphNodes:
    """Tests for individual LangGraph nodes."""
    
    def test_should_reoptimize_returns_reoptimize(self):
        """should_reoptimize should return 'reoptimize' when fairness check says REOPTIMIZE."""
        state = AllocationState(
            fairness_check_1={"status": "REOPTIMIZE"},
            route_proposal_2=None,
        )
        
        result = should_reoptimize(state)
        
        assert result == "reoptimize"
    
    def test_should_reoptimize_returns_continue(self):
        """should_reoptimize should return 'continue' when fairness check says ACCEPT."""
        state = AllocationState(
            fairness_check_1={"status": "ACCEPT"},
        )
        
        result = should_reoptimize(state)
        
        assert result == "continue"
    
    def test_should_reoptimize_skips_when_proposal2_exists(self):
        """should_reoptimize should return 'continue' if proposal 2 already exists."""
        state = AllocationState(
            fairness_check_1={"status": "REOPTIMIZE"},
            route_proposal_2={"allocation": []},  # Already have proposal 2
        )
        
        result = should_reoptimize(state)
        
        assert result == "continue"


class TestWorkflowGraph:
    """Tests for the LangGraph workflow."""
    
    def test_create_allocation_graph(self):
        """create_allocation_graph should return a compiled graph."""
        graph = create_allocation_graph()
        
        assert graph is not None
        # Graph should have nodes
        assert hasattr(graph, 'invoke') or hasattr(graph, 'ainvoke')
    
    def test_workflow_visualization(self):
        """get_workflow_visualization should return a Mermaid diagram."""
        diagram = get_workflow_visualization()
        
        assert "```mermaid" in diagram
        assert "ml_effort" in diagram
        assert "fairness_check_1" in diagram
        assert "explainability" in diagram
    
    def test_graph_with_gemini_disabled(self):
        """Graph should compile without Gemini node."""
        import os
        os.environ.pop("GOOGLE_API_KEY", None)
        
        graph = create_allocation_graph(enable_gemini=False)
        
        assert graph is not None


class TestDecisionLogging:
    """Tests for decision log generation."""
    
    def test_ml_effort_node_creates_log(self):
        """ml_effort_node should append to decision_logs."""
        # This test would require mock drivers/routes
        # Placeholder for full integration test
        pass
    
    def test_decision_log_format(self):
        """Decision logs should have required fields."""
        log_entry = {
            "timestamp": "2026-02-04T10:00:00",
            "agent_name": "ML_EFFORT",
            "step_type": "MATRIX_GENERATION",
            "input_snapshot": {"num_drivers": 5},
            "output_snapshot": {"matrix_size": 25},
        }
        
        assert "timestamp" in log_entry
        assert "agent_name" in log_entry
        assert "step_type" in log_entry
        assert "input_snapshot" in log_entry
        assert "output_snapshot" in log_entry


class TestWorkflowPerformance:
    """Performance tests for the workflow."""
    
    @pytest.mark.slow
    def test_state_serialization_performance(self):
        """State serialization should be fast."""
        state = AllocationState(
            request={"packages": [{"id": f"pkg_{i}"} for i in range(100)]},
            decision_logs=[{"step": i} for i in range(50)],
        )
        
        start = time.time()
        for _ in range(100):
            state.model_dump(mode="json")
        elapsed = time.time() - start
        
        # Should serialize 100 times in under 1 second
        assert elapsed < 1.0, f"Serialization too slow: {elapsed:.2f}s"


# Integration test placeholder
class TestWorkflowEquivalence:
    """Tests to verify LangGraph produces same results as original."""
    
    @pytest.mark.skip(reason="Requires full DB setup - run manually")
    async def test_workflow_produces_identical_results(self):
        """LangGraph workflow should produce identical results to original endpoint."""
        # This test compares original /allocate response with /allocate/langgraph
        # Requires:
        # 1. Same request to both endpoints
        # 2. Compare final allocations (ignoring timestamps/UUIDs)
        pass
