"""
Offline XGBoost model trainer for per-driver effort prediction.

This trainer runs as a scheduled background job (Celery Beat) rather than
inline during allocation. It trains per-driver XGBoost models using
historical assignment data and driver feedback.

Key safety decisions:
  - Models are saved using xgboost.Booster.save_model() (JSON format)
  - NO pickle is used anywhere
  - Models are versioned in the database with checksums

Usage:
    from app.trainer.train_xgboost import retrain_all_drivers

    # Called by Celery Beat daily, or manually
    retrain_all_drivers()
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

# Minimum samples needed to train a useful model
MIN_TRAINING_SAMPLES = 10
MAX_TRAINING_SAMPLES = 500

# XGBoost hyperparameters for effort prediction
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",
}

FEATURE_NAMES = [
    "route_difficulty_score",
    "num_packages",
    "total_weight_kg",
    "num_stops",
    "estimated_time_minutes",
    "fatigue_score",
    "recent_avg_effort",
    "recent_hard_days",
]


def _get_sync_session() -> Session:
    """Create a synchronous DB session for the trainer."""
    settings = get_settings()
    sync_url = os.environ.get("SYNC_DATABASE_URL")
    if not sync_url:
        sync_url = settings.database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )
    engine = create_engine(sync_url, echo=False)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def _ensure_model_dir() -> Path:
    """Ensure the model storage directory exists and return its path."""
    settings = get_settings()
    model_dir = Path(settings.model_storage_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _compute_checksum(filepath: str) -> str:
    """Compute SHA-256 checksum of a model file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def train_driver_model(
    driver_id: str,
    X: np.ndarray,
    y: np.ndarray,
    model_dir: Path,
) -> Optional[dict]:
    """
    Train an XGBoost model for a single driver.

    Args:
        driver_id: UUID string of the driver.
        X: Feature matrix (n_samples, n_features).
        y: Target vector (actual effort values).
        model_dir: Directory to save the model file.

    Returns:
        dict with model metadata, or None if training failed.
    """
    if len(X) < MIN_TRAINING_SAMPLES:
        logger.debug(
            f"Driver {driver_id}: only {len(X)} samples, need {MIN_TRAINING_SAMPLES}"
        )
        return None

    # Cap training samples
    if len(X) > MAX_TRAINING_SAMPLES:
        # Use most recent samples
        X = X[-MAX_TRAINING_SAMPLES:]
        y = y[-MAX_TRAINING_SAMPLES:]

    try:
        dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES[:X.shape[1]])

        # Train
        bst = xgb.train(
            {
                "objective": XGBOOST_PARAMS["objective"],
                "max_depth": XGBOOST_PARAMS["max_depth"],
                "learning_rate": XGBOOST_PARAMS["learning_rate"],
                "min_child_weight": XGBOOST_PARAMS["min_child_weight"],
                "subsample": XGBOOST_PARAMS["subsample"],
                "colsample_bytree": XGBOOST_PARAMS["colsample_bytree"],
                "eval_metric": XGBOOST_PARAMS["eval_metric"],
            },
            dtrain,
            num_boost_round=XGBOOST_PARAMS["n_estimators"],
        )

        # Predict and compute metrics
        predictions = bst.predict(dtrain)
        mse = float(np.mean((predictions - y) ** 2))
        rmse = float(np.sqrt(mse))

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Save model using safe JSON format (NO PICKLE)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"driver_{driver_id}_v{timestamp}.json"
        model_path = str(model_dir / model_filename)

        bst.save_model(model_path)

        checksum = _compute_checksum(model_path)

        logger.info(
            f"Trained model for driver {driver_id}: "
            f"samples={len(X)}, RMSE={rmse:.3f}, R²={r2:.3f}"
        )

        return {
            "driver_id": driver_id,
            "model_path": model_path,
            "model_checksum": checksum,
            "training_samples": len(X),
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "feature_names": FEATURE_NAMES[:X.shape[1]],
            "trained_at": datetime.utcnow(),
        }

    except Exception as e:
        logger.error(f"Failed to train model for driver {driver_id}: {e}")
        return None


def load_driver_model(model_path: str) -> Optional[xgb.Booster]:
    """
    Safely load an XGBoost model from a JSON file.
    NO PICKLE ANYWHERE.

    Args:
        model_path: Path to the .json model file.

    Returns:
        xgb.Booster instance, or None if loading failed.
    """
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return None

    try:
        bst = xgb.Booster()
        bst.load_model(model_path)
        return bst
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None


def retrain_all_drivers():
    """
    Retrain XGBoost models for all drivers with sufficient historical data.

    This is the main entry point invoked by Celery Beat.
    It:
      1. Queries all drivers with enough assignment history
      2. Builds feature matrices from historical data
      3. Trains per-driver models
      4. Saves models as JSON artifacts (NO pickle)
      5. Updates the driver_effort_models table with new paths and metadata
    """
    from app.models.assignment import Assignment
    from app.models.driver import Driver, DriverStatsDaily, DriverFeedback
    from app.models.driver_effort_model import DriverEffortModel

    session = _get_sync_session()
    model_dir = _ensure_model_dir()

    try:
        # Get all drivers
        drivers = session.query(Driver).all()
        logger.info(f"Starting retraining for {len(drivers)} drivers")

        trained_count = 0
        skipped_count = 0

        for driver in drivers:
            driver_id_str = str(driver.id)

            # Get historical assignments for this driver
            assignments = (
                session.query(Assignment)
                .filter(Assignment.driver_id == driver.id)
                .order_by(Assignment.date.desc())
                .limit(MAX_TRAINING_SAMPLES)
                .all()
            )

            if len(assignments) < MIN_TRAINING_SAMPLES:
                skipped_count += 1
                continue

            # Build feature matrix from assignment + route data
            X_list = []
            y_list = []

            for asgn in assignments:
                # Get corresponding route
                from app.models.route import Route
                route = session.query(Route).filter(Route.id == asgn.route_id).first()
                if not route:
                    continue

                # Get driver stats for that date
                stats = (
                    session.query(DriverStatsDaily)
                    .filter(
                        DriverStatsDaily.driver_id == driver.id,
                        DriverStatsDaily.date == asgn.date,
                    )
                    .first()
                )

                # Get recent feedback
                feedback = (
                    session.query(DriverFeedback)
                    .filter(DriverFeedback.driver_id == driver.id)
                    .order_by(DriverFeedback.created_at.desc())
                    .first()
                )

                fatigue = float(feedback.tiredness_level) if feedback else 3.0
                recent_avg = stats.avg_workload_score if stats else 60.0
                hard_days = 0  # simplified; could compute from history

                features = [
                    route.route_difficulty_score,
                    route.num_packages,
                    route.total_weight_kg,
                    route.num_stops,
                    route.estimated_time_minutes,
                    fatigue,
                    recent_avg,
                    hard_days,
                ]

                X_list.append(features)
                y_list.append(asgn.workload_score)

            if len(X_list) < MIN_TRAINING_SAMPLES:
                skipped_count += 1
                continue

            X = np.array(X_list, dtype=np.float64)
            y = np.array(y_list, dtype=np.float64)

            result = train_driver_model(driver_id_str, X, y, model_dir)

            if result:
                # Upsert into driver_effort_models
                existing = (
                    session.query(DriverEffortModel)
                    .filter(DriverEffortModel.driver_id == driver.id)
                    .first()
                )

                if existing:
                    existing.model_version += 1
                    existing.model_path = result["model_path"]
                    existing.model_checksum = result["model_checksum"]
                    existing.training_samples = result["training_samples"]
                    existing.current_mse = result["mse"]
                    existing.r2_score = result["r2_score"]
                    existing.feature_names = result["feature_names"]
                    existing.last_trained_at = result["trained_at"]
                    existing.active = True
                else:
                    model_record = DriverEffortModel(
                        driver_id=driver.id,
                        model_version=1,
                        model_path=result["model_path"],
                        model_checksum=result["model_checksum"],
                        training_samples=result["training_samples"],
                        current_mse=result["mse"],
                        r2_score=result["r2_score"],
                        feature_names=result["feature_names"],
                        last_trained_at=result["trained_at"],
                        active=True,
                    )
                    session.add(model_record)

                trained_count += 1

        session.commit()
        logger.info(
            f"Retraining complete: {trained_count} trained, {skipped_count} skipped"
        )
        return {"trained": trained_count, "skipped": skipped_count}

    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        session.rollback()
        raise
    finally:
        session.close()
