"""
Agent Events SSE Endpoint.

Provides Server-Sent Events stream for real-time agent status updates
to visualization frontends.
"""

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import json
import asyncio

from app.core.events import agent_event_bus


router = APIRouter(prefix="/api/v1", tags=["agent-events"])


@router.get("/agent-events/stream")
async def agent_events_stream(
    run_id: Optional[str] = Query(None, description="Filter by allocation run ID")
):
    """
    Server-Sent Events endpoint for agent events.
    
    Returns a continuous stream of agent events for real-time visualization.
    Optionally filter by allocation_run_id via query parameter.
    
    Args:
        run_id: Optional allocation run ID to filter events
        
    Returns:
        SSE stream of agent events
    """

    async def event_generator():
        # Send initial connection event
        init_event = {
            "type": "connected",
            "message": "SSE connection established",
            "filter_run_id": run_id,
        }
        yield f"data: {json.dumps(init_event)}\n\n"
        
        # Send any recent events for this run
        if run_id:
            recent = agent_event_bus.get_recent_events(allocation_run_id=run_id)
            for event in recent:
                yield f"data: {json.dumps(event)}\n\n"
        
        # Stream live events
        async for event in agent_event_bus.subscribe():
            # Filter by run_id if specified
            if run_id and event.get("allocation_run_id") != run_id:
                continue
            
            # SSE format: "data: {...}\n\n"
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/agent-events/recent")
async def get_recent_events(
    run_id: Optional[str] = Query(None, description="Filter by allocation run ID"),
    limit: int = Query(50, ge=1, le=200, description="Maximum events to return"),
):
    """
    Get recent agent events (non-streaming).
    
    Useful for initial page load or debugging.
    
    Args:
        run_id: Optional allocation run ID to filter events
        limit: Maximum number of events to return
        
    Returns:
        List of recent agent events
    """
    events = agent_event_bus.get_recent_events(
        allocation_run_id=run_id,
        limit=limit,
    )
    return {"events": events, "count": len(events)}
