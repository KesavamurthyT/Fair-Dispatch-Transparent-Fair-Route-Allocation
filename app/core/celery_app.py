"""
Celery application factory for Fair Dispatch System.
Configures the broker (Redis), result backend, serialization,
and reliability settings for the async worker infrastructure.
"""

from celery import Celery

from app.config import get_settings


def create_celery_app() -> Celery:
    """
    Create and configure the Celery application instance.

    Uses Redis as both broker and result backend.
    Configured for reliability: acks_late + prefetch_multiplier=1
    ensures tasks are not lost on worker crashes.
    """
    settings = get_settings()

    app = Celery(
        "fair_dispatch",
        broker=settings.effective_broker_url,
        backend=settings.effective_result_backend,
    )

    app.conf.update(
        # Serialization — JSON only (no pickle anywhere)
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",

        # Reliability
        task_acks_late=True,
        worker_prefetch_multiplier=1,

        # Timeouts (5 min soft, 6 min hard)
        task_soft_time_limit=300,
        task_time_limit=360,

        # Result expiry (24 hours)
        result_expires=86400,

        # Task routing
        task_default_queue="allocation",

        # Retry on connection errors during startup
        broker_connection_retry_on_startup=True,

        # Beat schedule for periodic tasks (Phase 4 - async training)
        beat_schedule={
            "retrain-xgboost-daily": {
                "task": "retrain_xgboost_models",
                "schedule": 86400.0,  # 24 hours in seconds
                "options": {"queue": "allocation"},
            },
        },
    )

    # Auto-discover tasks in app.tasks module
    app.autodiscover_tasks(["app"])

    return app


# Singleton instance used by workers and task decorators
celery_app = create_celery_app()
