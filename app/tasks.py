"""
Celery task definitions for the Fair Dispatch System.
Tasks are discovered by the Celery app via autodiscover.
"""

import logging

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="run_allocation_job",
    queue="allocation",
    max_retries=2,
    default_retry_delay=30,
)
def run_allocation_job(self, allocation_run_id: str) -> dict:
    """
    Execute the full LangGraph allocation pipeline in a worker process.

    This is the main async task that replaces synchronous allocation.
    It runs clustering, effort computation, CVRP solving, fairness checks,
    negotiation, and persists final assignments to the database.

    Args:
        allocation_run_id: UUID string of the AllocationRun to process.

    Returns:
        dict with status and summary metrics.
    """
    from app.worker.langgraph_runner import execute_allocation

    logger.info(f"Starting allocation job: {allocation_run_id}")

    try:
        result = execute_allocation(allocation_run_id)
        logger.info(f"Allocation job completed: {allocation_run_id}")
        return {
            "status": "completed",
            "allocation_run_id": allocation_run_id,
            "num_assignments": result.get("num_assignments", 0),
        }
    except Exception as exc:
        logger.error(f"Allocation job failed: {allocation_run_id} — {exc}")
        # Mark the run as failed in the DB before retrying
        try:
            from app.worker.langgraph_runner import mark_run_failed
            mark_run_failed(allocation_run_id, str(exc))
        except Exception:
            pass
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    name="retrain_xgboost_models",
    queue="allocation",
    max_retries=1,
    default_retry_delay=60,
)
def retrain_xgboost_models(self) -> dict:
    """
    Periodic task: retrain all per-driver XGBoost models.
    Scheduled by Celery Beat (daily).
    Uses safe JSON model storage — NO pickle.
    """
    from app.trainer.train_xgboost import retrain_all_drivers

    logger.info("Starting scheduled XGBoost retraining")
    try:
        result = retrain_all_drivers()
        logger.info(f"Retraining complete: {result}")
        return result
    except Exception as exc:
        logger.error(f"Retraining failed: {exc}")
        raise self.retry(exc=exc)
