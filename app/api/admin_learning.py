"""
Admin Learning API endpoints for Phase 8.
Provides monitoring, control, and debugging for the Learning Agent.
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import (
    LearningEpisode,
    DriverEffortModel,
    FairnessConfig,
    Driver,
)
from app.services.learning_agent import LearningAgent, hash_config
from app.schemas.learning_schemas import (
    LearningStatusResponse,
    LearningEpisodeResponse,
    LearningEpisodesListResponse,
    DriverModelStatusResponse,
    ForceConfigRequest,
    ForceConfigResponse,
    TriggerLearningRequest,
    TriggerLearningResponse,
    DriverModelUpdateRequest,
    DriverModelUpdateResponse,
    AllDriverModelsListResponse,
    BanditStatistics,
)


router = APIRouter(prefix="/admin/learning", tags=["Admin - Learning Agent"])


@router.get(
    "/status",
    response_model=LearningStatusResponse,
    summary="Get learning agent status",
    description="Returns overall learning status including bandit statistics, "
                "active driver models, and top performing configurations.",
)
async def get_learning_status(
    db: AsyncSession = Depends(get_db),
) -> LearningStatusResponse:
    """Get current learning agent status."""
    agent = LearningAgent(db)
    status_dict = await agent.get_learning_status()
    
    return LearningStatusResponse(
        current_config=status_dict.get("current_config"),
        top_performing_configs=status_dict.get("top_performing_configs", []),
        driver_models_active=status_dict.get("driver_models_active", 0),
        avg_prediction_mse=status_dict.get("avg_prediction_mse", 0.0),
        recent_episodes_7d=status_dict.get("recent_episodes_7d", 0),
        total_arms=status_dict.get("total_arms", 81),
        bandit_statistics=BanditStatistics(
            total_samples=status_dict.get("bandit_statistics", {}).get("total_samples", 0),
            explored_arms=status_dict.get("bandit_statistics", {}).get("explored_arms", 0),
            total_arms=status_dict.get("total_arms", 81),
        ),
    )


@router.get(
    "/episodes",
    response_model=LearningEpisodesListResponse,
    summary="List learning episodes",
    description="Returns paginated list of learning episodes with rewards.",
)
async def list_episodes(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    has_reward: Optional[bool] = Query(None),
    is_experimental: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> LearningEpisodesListResponse:
    """List learning episodes with optional filters."""
    # Build query
    query = select(LearningEpisode).order_by(LearningEpisode.created_at.desc())
    count_query = select(func.count(LearningEpisode.id))
    
    # Apply filters
    if has_reward is True:
        query = query.where(LearningEpisode.episode_reward.isnot(None))
        count_query = count_query.where(LearningEpisode.episode_reward.isnot(None))
    elif has_reward is False:
        query = query.where(LearningEpisode.episode_reward.is_(None))
        count_query = count_query.where(LearningEpisode.episode_reward.is_(None))
    
    if is_experimental is not None:
        query = query.where(LearningEpisode.is_experimental == is_experimental)
        count_query = count_query.where(LearningEpisode.is_experimental == is_experimental)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    
    result = await db.execute(query)
    episodes = result.scalars().all()
    
    return LearningEpisodesListResponse(
        episodes=[LearningEpisodeResponse.model_validate(ep) for ep in episodes],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + len(episodes)) < total,
    )


@router.get(
    "/episodes/{episode_id}",
    response_model=LearningEpisodeResponse,
    summary="Get learning episode details",
)
async def get_episode(
    episode_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> LearningEpisodeResponse:
    """Get details of a specific learning episode."""
    result = await db.execute(
        select(LearningEpisode).where(LearningEpisode.id == episode_id)
    )
    episode = result.scalar_one_or_none()
    
    if not episode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Episode {episode_id} not found",
        )
    
    return LearningEpisodeResponse.model_validate(episode)


@router.post(
    "/force_config",
    response_model=ForceConfigResponse,
    summary="Force a specific fairness config",
    description="Override bandit selection with a specific config. "
                "Use for emergency rollbacks or testing.",
)
async def force_config(
    request: ForceConfigRequest,
    db: AsyncSession = Depends(get_db),
) -> ForceConfigResponse:
    """Force a specific fairness configuration."""
    # Get current active config
    result = await db.execute(
        select(FairnessConfig).where(FairnessConfig.is_active == True).limit(1)
    )
    current = result.scalar_one_or_none()
    
    previous_config = None
    if current:
        previous_config = {
            "gini_threshold": current.gini_threshold,
            "stddev_threshold": current.stddev_threshold,
            "recovery_lightening_factor": current.recovery_lightening_factor,
            "ev_charging_penalty_weight": current.ev_charging_penalty_weight,
        }
        current.is_active = False
    
    # Create new config
    new_config = FairnessConfig(
        is_active=True,
        gini_threshold=request.gini_threshold,
        stddev_threshold=request.stddev_threshold,
        recovery_lightening_factor=request.recovery_lightening_factor,
        ev_charging_penalty_weight=request.ev_charging_penalty_weight,
        max_gap_threshold=request.max_gap_threshold,
    )
    db.add(new_config)
    await db.commit()
    
    return ForceConfigResponse(
        status="success",
        message=f"Config forced. Reason: {request.reason}",
        config_applied={
            "gini_threshold": request.gini_threshold,
            "stddev_threshold": request.stddev_threshold,
            "recovery_lightening_factor": request.recovery_lightening_factor,
            "ev_charging_penalty_weight": request.ev_charging_penalty_weight,
            "max_gap_threshold": request.max_gap_threshold,
        },
        previous_config=previous_config,
    )


@router.post(
    "/trigger",
    response_model=TriggerLearningResponse,
    summary="Manually trigger learning pipeline",
    description="Trigger the daily learning pipeline manually. "
                "Useful for testing or forcing immediate updates.",
)
async def trigger_learning(
    request: TriggerLearningRequest,
    db: AsyncSession = Depends(get_db),
) -> TriggerLearningResponse:
    """Manually trigger learning pipeline."""
    from cron.daily_learning import DailyLearningPipeline
    
    start_time = datetime.utcnow()
    metrics = {
        "episodes_processed": 0,
        "rewards_computed": 0,
        "models_updated": 0,
        "config_selection": None,
        "errors": [],
    }
    
    try:
        pipeline = DailyLearningPipeline(db)
        
        if request.process_episodes:
            await pipeline._process_pending_episodes()
            metrics["episodes_processed"] = pipeline.metrics.get("episodes_processed", 0)
            metrics["rewards_computed"] = pipeline.metrics.get("rewards_computed", 0)
        
        if request.select_config:
            await pipeline._select_todays_config()
            metrics["config_selection"] = pipeline.metrics.get("config_selection")
        
        if request.update_models:
            await pipeline._update_driver_models()
            metrics["models_updated"] = pipeline.metrics.get("models_updated", 0)
        
        await db.commit()
        
        return TriggerLearningResponse(
            status="success",
            episodes_processed=metrics["episodes_processed"],
            rewards_computed=metrics["rewards_computed"],
            models_updated=metrics["models_updated"],
            config_selection=metrics["config_selection"],
            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            errors=metrics.get("errors", []),
        )
        
    except Exception as e:
        return TriggerLearningResponse(
            status="error",
            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            errors=[str(e)],
        )


@router.get(
    "/models",
    response_model=AllDriverModelsListResponse,
    summary="List all driver effort models",
)
async def list_driver_models(
    active_only: bool = Query(True),
    db: AsyncSession = Depends(get_db),
) -> AllDriverModelsListResponse:
    """List all driver effort models."""
    query = select(DriverEffortModel)
    if active_only:
        query = query.where(DriverEffortModel.active == True)
    query = query.order_by(DriverEffortModel.last_trained_at.desc().nullsfirst())
    
    result = await db.execute(query)
    models = result.scalars().all()
    
    # Calculate stats
    active_count = sum(1 for m in models if m.active)
    mse_values = [m.current_mse for m in models if m.current_mse is not None]
    avg_mse = sum(mse_values) / len(mse_values) if mse_values else None
    
    return AllDriverModelsListResponse(
        models=[
            DriverModelStatusResponse(
                driver_id=m.driver_id,
                model_version=m.model_version,
                training_samples=m.training_samples,
                current_mse=m.current_mse,
                r2_score=m.r2_score,
                mse_history=m.mse_history,
                active=m.active,
                last_trained_at=m.last_trained_at,
            )
            for m in models
        ],
        total=len(models),
        active_count=active_count,
        avg_mse=avg_mse,
    )


@router.get(
    "/models/{driver_id}",
    response_model=DriverModelStatusResponse,
    summary="Get driver model status",
)
async def get_driver_model(
    driver_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DriverModelStatusResponse:
    """Get status of a specific driver's effort model."""
    result = await db.execute(
        select(DriverEffortModel).where(DriverEffortModel.driver_id == driver_id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model found for driver {driver_id}",
        )
    
    return DriverModelStatusResponse(
        driver_id=model.driver_id,
        model_version=model.model_version,
        training_samples=model.training_samples,
        current_mse=model.current_mse,
        r2_score=model.r2_score,
        mse_history=model.mse_history,
        active=model.active,
        last_trained_at=model.last_trained_at,
    )


@router.post(
    "/models/{driver_id}/retrain",
    response_model=DriverModelUpdateResponse,
    summary="Retrain a driver's model",
)
async def retrain_driver_model(
    driver_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DriverModelUpdateResponse:
    """Manually trigger retraining of a driver's effort model."""
    # Check driver exists
    result = await db.execute(
        select(Driver).where(Driver.id == driver_id)
    )
    driver = result.scalar_one_or_none()
    
    if not driver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Driver {driver_id} not found",
        )
    
    agent = LearningAgent(db)
    result = await agent.effort_learner.update_model(driver_id)
    
    await db.commit()
    
    return DriverModelUpdateResponse(
        status=result.get("status", "unknown"),
        driver_id=driver_id,
        model_version=result.get("model_version"),
        training_samples=result.get("training_samples"),
        mse=result.get("mse"),
        r2_score=result.get("r2_score"),
        reason=result.get("reason"),
    )
