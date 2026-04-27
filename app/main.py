"""
Fair Dispatch System - FastAPI Application
Main entry point for the API server.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.api import (
    allocation_router,
    drivers_router,
    routes_router,
    feedback_router,
    driver_api_router,
    admin_router,
    admin_learning_router,
    allocation_langgraph_router,
    api_keys_router,
)
from app.api.agent_events import router as agent_events_router
from app.api.runs import router as runs_router


settings = get_settings()

# Path to frontend directory
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    print(f"Starting {settings.app_title} v{settings.app_version}")
    
    # Initialize database tables (important for SQLite)
    from app.database import init_db, get_db
    await init_db()
    print("Database tables initialized")
    
    # Seed historical data for mock drivers (idempotent)
    from app.services.history_seed import seed_mock_history
    async for db in get_db():
        seeded_count = await seed_mock_history(db)
        if seeded_count is not None and seeded_count > 0:
            print(f"Seeded {seeded_count} mock drivers with 7-day history")
        else:
            print("Mock driver history already seeded (skipped)")
        break
    
    yield
    # Shutdown
    print("Shutting down...")



# Create FastAPI application
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="""
    ## Fair Dispatch System API
    
    A fairness-focused route allocation system for delivery operations.
    
    ### Features
    - **Route Clustering**: Groups packages using K-Means for efficient routes
    - **Workload Scoring**: Calculates balanced workload metrics
    - **Fairness Metrics**: Computes Gini index and fairness scores
    - **Explainability**: Provides human-readable explanations for allocations
    - **LangGraph Workflow**: Multi-agent orchestration with LangSmith tracing
    
    ### Main Endpoints
    - `POST /api/v1/allocate` - Allocate packages to drivers (original)
    - `POST /api/v1/allocate/langgraph` - Allocate with LangGraph workflow
    - `GET /api/v1/drivers/{id}` - Get driver details and stats
    - `GET /api/v1/routes/{id}` - Get route details
    - `POST /api/v1/feedback` - Submit driver feedback
    - `GET /api/v1/agent-events/stream` - SSE stream for agent events
    """,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(allocation_router, prefix=settings.api_prefix)
app.include_router(allocation_langgraph_router, prefix=settings.api_prefix)
app.include_router(drivers_router, prefix=settings.api_prefix)
app.include_router(routes_router, prefix=settings.api_prefix)
app.include_router(feedback_router, prefix=settings.api_prefix)
app.include_router(driver_api_router, prefix=settings.api_prefix)
app.include_router(admin_router, prefix=settings.api_prefix)
app.include_router(admin_learning_router, prefix=settings.api_prefix)

# Include SSE agent events router (no prefix - it defines its own)
app.include_router(agent_events_router)

# Include run-scoped endpoints
app.include_router(runs_router, prefix=settings.api_prefix)
app.include_router(api_keys_router, prefix=settings.api_prefix)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "service": settings.app_title,
        "version": settings.app_version,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",  # TODO: Add actual DB check
    }


# Mount static files for frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/demo/allocate", tags=["Demo"])
async def demo_allocate():
    """Serve the API demo page for testing allocation endpoint."""
    demo_path = FRONTEND_DIR / "demo.html"
    return FileResponse(demo_path, media_type="text/html")


@app.get("/demo/visualization", tags=["Demo"])
async def demo_visualization():
    """Serve the agent visualization page."""
    viz_path = FRONTEND_DIR / "visualization.html"
    return FileResponse(viz_path, media_type="text/html")


@app.get("/demo/api-keys", tags=["Demo"])
async def demo_api_keys():
    """Serve the API key management page."""
    api_keys_path = FRONTEND_DIR / "api_keys.html"
    return FileResponse(api_keys_path, media_type="text/html")
