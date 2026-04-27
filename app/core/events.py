"""
Agent Event Bus for real-time SSE synchronization.

Provides pub/sub mechanism for agent events that can be consumed
by multiple SSE clients (visualization UI, demo page, etc.).
"""

from typing import AsyncIterator, Dict, Any, List, Optional
from uuid import UUID
import asyncio
import time


class AgentEventBus:
    """
    Simple in-process pub/sub for agent events.
    
    Multiple listeners (SSE connections) can subscribe and receive
    events published by LangGraph nodes during allocation runs.
    """

    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self._recent_events: List[Dict[str, Any]] = []
        self._max_recent = 100  # Keep last 100 events for late joiners

    async def subscribe(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator yielding events. Callers iterate and send as SSE.
        
        Yields:
            Agent event dictionaries
        """
        queue: asyncio.Queue = asyncio.Queue()
        
        async with self._lock:
            self._subscribers.append(queue)
            # Send recent events to catch up
            for event in self._recent_events[-20:]:
                await queue.put(event)

        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)

    async def publish(self, event: Dict[str, Any]) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Agent event dictionary
        """
        async with self._lock:
            # Store in recent events
            self._recent_events.append(event)
            if len(self._recent_events) > self._max_recent:
                self._recent_events = self._recent_events[-self._max_recent:]
            
            # Broadcast to all subscribers
            for queue in self._subscribers:
                try:
                    await queue.put(event)
                except Exception:
                    pass  # Ignore failed subscribers

    def get_recent_events(
        self, 
        allocation_run_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get recent events, optionally filtered by allocation run.
        
        Args:
            allocation_run_id: Filter by specific run (optional)
            limit: Maximum events to return
            
        Returns:
            List of recent events
        """
        events = self._recent_events
        if allocation_run_id:
            events = [
                e for e in events 
                if e.get("allocation_run_id") == allocation_run_id
            ]
        return events[-limit:]


# Global singleton
agent_event_bus = AgentEventBus()


def make_agent_event(
    allocation_run_id: str,
    agent_name: str,
    step_type: str,
    state: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a standardized agent event dictionary.
    
    Args:
        allocation_run_id: UUID string of the allocation run
        agent_name: Agent identifier (e.g., "ML_EFFORT", "ROUTE_PLANNER")
        step_type: Step identifier (e.g., "MATRIX_GENERATION", "PROPOSAL_1")
        state: Event state - "STARTED", "COMPLETED", or "ERROR"
        payload: Optional additional data for the event
        
    Returns:
        Formatted event dictionary
    """
    return {
        "allocation_run_id": str(allocation_run_id),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent_name": agent_name,
        "step_type": step_type,
        "state": state,
        "payload": payload or {},
    }


async def publish_agent_event(
    allocation_run_id: str,
    agent_name: str,
    step_type: str,
    state: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience function to publish an agent event.
    
    Args:
        allocation_run_id: UUID string of the allocation run
        agent_name: Agent identifier
        step_type: Step identifier
        state: Event state
        payload: Optional additional data
    """
    event = make_agent_event(
        allocation_run_id=allocation_run_id,
        agent_name=agent_name,
        step_type=step_type,
        state=state,
        payload=payload,
    )
    await agent_event_bus.publish(event)
