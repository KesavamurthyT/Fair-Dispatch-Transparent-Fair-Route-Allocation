"""API routers package initialization."""

from app.api.allocation import router as allocation_router
from app.api.drivers import router as drivers_router
from app.api.routes import router as routes_router
from app.api.feedback import router as feedback_router
from app.api.driver_api import router as driver_api_router
from app.api.admin import router as admin_router
from app.api.admin_learning import router as admin_learning_router
from app.api.allocation_langgraph import router as allocation_langgraph_router
from app.api.api_keys import router as api_keys_router

__all__ = [
    "allocation_router",
    "drivers_router",
    "routes_router",
    "feedback_router",
    "driver_api_router",
    "admin_router",
    "admin_learning_router",
    "allocation_langgraph_router",
    "api_keys_router",
]

