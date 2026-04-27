"""
LangGraph workflow definition for Fair Dispatch allocation.
Orchestrates all agents in a graph with conditional edges and checkpointing.
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END

from app.schemas.allocation_state import AllocationState
from app.services.langgraph_nodes import (
    orchestrator_node,
    ml_effort_node,
    route_planner_node,
    fairness_check_node,
    route_planner_reoptimize_node,
    select_final_proposal_node,
    driver_liaison_node,
    final_resolution_node,
    explainability_node,
    learning_agent_node,
    should_reoptimize,
    has_counter_decisions,
)


def create_allocation_graph(
    checkpointer: Optional[Any] = None,
    enable_gemini: bool = False,
) -> StateGraph:
    """
    Create the allocation workflow graph.
    
    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence
        enable_gemini: If True, add Gemini explainability node
        
    Returns:
        Compiled StateGraph ready for invocation
    """
    # Create the graph with AllocationState
    workflow = StateGraph(AllocationState)
    
    # ==========================================================================
    # Add Nodes
    # ==========================================================================
    
    # Phase 0: Central Orchestrator (Workflow Initialization)
    workflow.add_node("orchestrator", orchestrator_node)
    
    # Phase 1: ML Effort Agent
    workflow.add_node("ml_effort", ml_effort_node)
    
    # Phase 2: Route Planner Agent (Proposal 1)
    workflow.add_node("route_planner_1", route_planner_node)
    
    # Phase 3: Fairness Manager Agent (Check 1)
    workflow.add_node("fairness_check_1", fairness_check_node)
    
    # Phase 3b: Route Planner Re-optimization (Proposal 2)
    workflow.add_node("route_planner_2", route_planner_reoptimize_node)
    
    # Phase 3c: Fairness Manager Agent (Check 2)
    workflow.add_node("fairness_check_2", fairness_check_node)
    
    # Phase 3d: Select Final Proposal
    workflow.add_node("select_final", select_final_proposal_node)
    
    # Phase 4: Driver Liaison Agent
    workflow.add_node("driver_liaison", driver_liaison_node)
    
    # Phase 5: Final Resolution Agent
    workflow.add_node("final_resolution", final_resolution_node)
    
    # Phase 6: Explainability Agent
    workflow.add_node("explainability", explainability_node)
    
    # Phase 7: Learning Agent
    workflow.add_node("learning", learning_agent_node)
    
    # Optional: Gemini Explainability Node
    if enable_gemini and os.getenv("GOOGLE_API_KEY"):
        try:
            from app.services.gemini_explain_node import gemini_explain_node
            workflow.add_node("gemini_explain", gemini_explain_node)
        except ImportError:
            pass  # Gemini not available, skip
    
    # ==========================================================================
    # Add Edges
    # ==========================================================================
    
    # Entry point
    workflow.set_entry_point("orchestrator")
    
    # Orchestrator -> ML Effort
    workflow.add_edge("orchestrator", "ml_effort")
    
    # Linear flow: ML Effort -> Route Planner 1
    workflow.add_edge("ml_effort", "route_planner_1")
    
    # Route Planner 1 -> Fairness Check 1
    workflow.add_edge("route_planner_1", "fairness_check_1")
    
    # Conditional: Fairness Check 1 -> Reoptimize or Select Final
    workflow.add_conditional_edges(
        "fairness_check_1",
        should_reoptimize,
        {
            "reoptimize": "route_planner_2",
            "continue": "select_final",
        }
    )
    
    # Reoptimize path: Route Planner 2 -> Fairness Check 2 -> Select Final
    workflow.add_edge("route_planner_2", "fairness_check_2")
    workflow.add_edge("fairness_check_2", "select_final")
    
    # Select Final -> Driver Liaison
    workflow.add_edge("select_final", "driver_liaison")
    
    # Conditional: Driver Liaison -> Final Resolution or Explainability
    workflow.add_conditional_edges(
        "driver_liaison",
        has_counter_decisions,
        {
            "resolve": "final_resolution",
            "skip": "explainability",
        }
    )
    
    # Final Resolution -> Explainability
    workflow.add_edge("final_resolution", "explainability")
    
    # Explainability -> Learning Agent
    workflow.add_edge("explainability", "learning")
    
    # Learning -> Gemini or END
    if enable_gemini and os.getenv("GOOGLE_API_KEY"):
        try:
            from app.services.gemini_explain_node import gemini_explain_node
            workflow.add_edge("learning", "gemini_explain")
            workflow.add_edge("gemini_explain", END)
        except ImportError:
            workflow.add_edge("learning", END)
    else:
        workflow.add_edge("learning", END)
    
    # ==========================================================================
    # Compile
    # ==========================================================================
    
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


# Global graph instance (lazy initialization)
_allocation_graph = None


def get_allocation_graph(
    checkpointer: Optional[Any] = None,
    enable_gemini: bool = None,
) -> StateGraph:
    """
    Get or create the allocation graph singleton.
    
    Args:
        checkpointer: Optional checkpointer for persistence
        enable_gemini: Override Gemini setting (defaults to env var)
        
    Returns:
        Compiled allocation graph
    """
    global _allocation_graph
    
    if enable_gemini is None:
        enable_gemini = os.getenv("ENABLE_GEMINI_EXPLAIN", "false").lower() == "true"
    
    if _allocation_graph is None:
        _allocation_graph = create_allocation_graph(
            checkpointer=checkpointer,
            enable_gemini=enable_gemini,
        )
    
    return _allocation_graph


async def invoke_allocation_workflow(
    request_dict: Dict[str, Any],
    config_used: Optional[Dict[str, Any]] = None,
    driver_models: list = None,
    route_models: list = None,
    route_dicts: list = None,
    driver_contexts: Dict[str, Dict[str, Any]] = None,
    recovery_targets: Dict[str, Optional[float]] = None,
    allocation_run_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> AllocationState:
    """
    Invoke the allocation workflow with the given inputs.
    
    This is the main entry point for running the LangGraph workflow.
    
    Args:
        request_dict: AllocationRequest.dict()
        config_used: Active FairnessConfig snapshot
        driver_models: List of driver model data
        route_models: List of route model data
        route_dicts: List of route dictionaries with packages
        driver_contexts: Dict of driver contexts for liaison agent
        recovery_targets: Recovery effort targets per driver
        allocation_run_id: ID of the AllocationRun for persistence
        thread_id: Thread ID for checkpointing (defaults to allocation_run_id)
        
    Returns:
        Final AllocationState with all agent outputs
    """
    graph = get_allocation_graph()
    
    # Build initial state
    initial_state = AllocationState(
        request=request_dict,
        config_used=config_used or {},
        driver_models=driver_models or [],
        route_models=route_models or [],
        route_dicts=route_dicts or [],
        driver_contexts=driver_contexts or {},
        recovery_targets=recovery_targets or {},
        allocation_run_id=allocation_run_id,
        workflow_start=datetime.utcnow(),
    )
    
    # Prepare config for graph invocation
    config = {}
    if thread_id or allocation_run_id:
        config["configurable"] = {"thread_id": thread_id or allocation_run_id}
    
    # Invoke the graph
    # Note: LangGraph's invoke returns the final state
    final_state_dict = await graph.ainvoke(initial_state.model_dump(), config=config)
    
    # Convert back to AllocationState
    return AllocationState.model_validate(final_state_dict)


def get_workflow_visualization() -> str:
    """
    Get a Mermaid diagram of the workflow for documentation.
    
    Returns:
        Mermaid diagram string
    """
    return """
```mermaid
graph TD
    A[Entry: ml_effort] --> B[route_planner_1]
    B --> C[fairness_check_1]
    C --> D{should_reoptimize?}
    D -->|reoptimize| E[route_planner_2]
    D -->|continue| G[select_final]
    E --> F[fairness_check_2]
    F --> G
    G --> H[driver_liaison]
    H --> I{has_counter_decisions?}
    I -->|resolve| J[final_resolution]
    I -->|skip| K[explainability]
    J --> K
    K --> L{gemini_enabled?}
    L -->|yes| M[gemini_explain]
    L -->|no| N[END]
    M --> N
```
"""
