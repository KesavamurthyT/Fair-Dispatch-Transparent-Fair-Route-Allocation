"""
Historical Feature Service - Computes historical features for drivers.
Used by ML Effort Agent and Driver Liaison Agent for personalized decisions.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.assignment import Assignment
from app.models.driver import DriverFeedback


class DriverHistoryFeatures(BaseModel):
    """Historical features computed for a driver."""
    driver_id: str = Field(..., description="Driver ID (string UUID)")
    recent_avg_effort: float = Field(default=0.0, description="Average workload over window")
    recent_std_effort: float = Field(default=0.0, description="Std dev of workload over window")
    recent_hard_days: int = Field(default=0, description="Days where effort > threshold")
    avg_stress_level: float = Field(default=5.0, description="Average stress (1-10)")
    avg_fairness_rating: float = Field(default=3.0, description="Average fairness rating (1-5)")
    avg_tiredness_level: float = Field(default=2.5, description="Average tiredness (1-5)")
    total_assignments: int = Field(default=0, description="Number of assignments in window")
    total_feedback_count: int = Field(default=0, description="Number of feedback entries")
    
    # Derived metrics
    fatigue_score: float = Field(default=3.0, description="Computed fatigue (1-5)")
    is_high_stress_driver: bool = Field(default=False, description="Avg stress >= 7")
    is_frequent_hard_days: bool = Field(default=False, description="Hard days >= 3 in window")


async def compute_history_features_for_drivers(
    db: AsyncSession,
    driver_ids: List[str],
    window_days: int = 7,
    hard_threshold_factor: float = 1.2,
) -> Dict[str, DriverHistoryFeatures]:
    """
    Compute historical features for a list of drivers from the last window_days.
    
    Args:
        db: Database session
        driver_ids: List of driver IDs (as strings)
        window_days: Number of days to look back
        hard_threshold_factor: Multiplier for avg to define "hard" day
        
    Returns:
        Dict mapping driver_id to DriverHistoryFeatures
    """
    if not driver_ids:
        return {}
    
    end_date = date.today()
    start_date = end_date - timedelta(days=window_days)
    
    # Convert string IDs to UUIDs for querying
    driver_uuids = []
    for d_id in driver_ids:
        try:
            driver_uuids.append(UUID(d_id))
        except (ValueError, TypeError):
            continue
    
    if not driver_uuids:
        return {d_id: DriverHistoryFeatures(driver_id=d_id) for d_id in driver_ids}
    
    # 1) Load assignments for all drivers in the window
    assignments_query = select(Assignment).where(
        and_(
            Assignment.driver_id.in_(driver_uuids),
            Assignment.date >= start_date,
            Assignment.date <= end_date,
        )
    )
    result = await db.execute(assignments_query)
    assignments = result.scalars().all()
    
    # Group by driver
    by_driver_assign: Dict[str, List[Assignment]] = {d_id: [] for d_id in driver_ids}
    for a in assignments:
        d_id_str = str(a.driver_id)
        if d_id_str in by_driver_assign:
            by_driver_assign[d_id_str].append(a)
    
    # 2) Load feedback for all drivers in the window
    feedbacks_query = select(DriverFeedback).where(
        and_(
            DriverFeedback.driver_id.in_(driver_uuids),
            DriverFeedback.created_at >= start_date,
        )
    )
    result = await db.execute(feedbacks_query)
    feedbacks = result.scalars().all()
    
    # Group by driver
    by_driver_fb: Dict[str, List[DriverFeedback]] = {d_id: [] for d_id in driver_ids}
    for f in feedbacks:
        d_id_str = str(f.driver_id)
        if d_id_str in by_driver_fb:
            by_driver_fb[d_id_str].append(f)
    
    # 3) Compute global average effort to define "hard" days
    all_efforts = [a.workload_score for a in assignments if a.workload_score is not None]
    global_avg_effort = (sum(all_efforts) / len(all_efforts)) if all_efforts else 40.0
    hard_threshold = global_avg_effort * hard_threshold_factor
    
    # 4) Compute features for each driver
    features: Dict[str, DriverHistoryFeatures] = {}
    
    for d_id in driver_ids:
        d_assigns = by_driver_assign.get(d_id, [])
        d_fbs = by_driver_fb.get(d_id, [])
        
        # Assignment-based metrics
        if d_assigns:
            efforts = [a.workload_score for a in d_assigns if a.workload_score is not None]
            if efforts:
                avg_effort = sum(efforts) / len(efforts)
                # Standard deviation
                if len(efforts) > 1:
                    variance = sum((e - avg_effort) ** 2 for e in efforts) / len(efforts)
                    std_effort = variance ** 0.5
                else:
                    std_effort = 0.0
                hard_days = sum(1 for e in efforts if e >= hard_threshold)
            else:
                avg_effort = 0.0
                std_effort = 0.0
                hard_days = 0
            total_assignments = len(d_assigns)
        else:
            avg_effort = 0.0
            std_effort = 0.0
            hard_days = 0
            total_assignments = 0
        
        # Feedback-based metrics
        if d_fbs:
            stress_levels = [f.stress_level for f in d_fbs if f.stress_level is not None]
            fairness_ratings = [f.fairness_rating for f in d_fbs if f.fairness_rating is not None]
            tiredness_levels = [f.tiredness_level for f in d_fbs if f.tiredness_level is not None]
            
            avg_stress = sum(stress_levels) / len(stress_levels) if stress_levels else 5.0
            avg_fairness = sum(fairness_ratings) / len(fairness_ratings) if fairness_ratings else 3.0
            avg_tiredness = sum(tiredness_levels) / len(tiredness_levels) if tiredness_levels else 2.5
            feedback_count = len(d_fbs)
        else:
            avg_stress = 5.0
            avg_fairness = 3.0
            avg_tiredness = 2.5
            feedback_count = 0
        
        # Compute derived fatigue score (1-5 scale)
        # Based on: recent hard days, stress level, tiredness
        fatigue_base = avg_tiredness
        if hard_days >= 3:
            fatigue_base += 1.0
        if avg_stress >= 7:
            fatigue_base += 0.5
        fatigue_score = min(5.0, max(1.0, fatigue_base))
        
        features[d_id] = DriverHistoryFeatures(
            driver_id=d_id,
            recent_avg_effort=round(avg_effort, 2),
            recent_std_effort=round(std_effort, 2),
            recent_hard_days=hard_days,
            avg_stress_level=round(avg_stress, 2),
            avg_fairness_rating=round(avg_fairness, 2),
            avg_tiredness_level=round(avg_tiredness, 2),
            total_assignments=total_assignments,
            total_feedback_count=feedback_count,
            fatigue_score=round(fatigue_score, 2),
            is_high_stress_driver=avg_stress >= 7,
            is_frequent_hard_days=hard_days >= 3,
        )
    
    return features


async def get_driver_history_summary(
    db: AsyncSession,
    driver_id: str,
    window_days: int = 7,
) -> Optional[DriverHistoryFeatures]:
    """
    Get history features for a single driver.
    Convenience wrapper around compute_history_features_for_drivers.
    """
    features = await compute_history_features_for_drivers(
        db=db,
        driver_ids=[driver_id],
        window_days=window_days,
    )
    return features.get(driver_id)


class HistoryConfig(BaseModel):
    """Configuration for history feature computation."""
    window_days: int = Field(default=7, ge=1, le=30, description="Days to look back")
    hard_threshold_factor: float = Field(default=1.2, ge=1.0, le=2.0, description="Multiplier for hard day threshold")
    # Weights for effort adjustment based on history
    w_hard_days: float = Field(default=2.0, ge=0, description="Effort penalty per hard day")
    w_stress: float = Field(default=3.0, ge=0, description="Effort penalty for high stress")
    w_fairness: float = Field(default=1.0, ge=0, description="Effort discount for good fairness rating")


def compute_history_effort_adjustment(
    history: Optional[DriverHistoryFeatures],
    config: Optional[HistoryConfig] = None,
) -> float:
    """
    Compute effort adjustment based on driver's historical features.
    
    Returns a value to ADD to base effort:
    - Positive = harder perceived effort (driver has had rough days)
    - Negative = easier perceived effort (driver is in good state)
    
    Args:
        history: Driver's history features
        config: Configuration for adjustment weights
        
    Returns:
        Adjustment value to add to base effort score
    """
    if history is None:
        return 0.0
    
    config = config or HistoryConfig()
    adjustment = 0.0
    
    # Penalize if driver had many hard days recently
    if history.recent_hard_days > 0:
        adjustment += config.w_hard_days * history.recent_hard_days
    
    # Penalize if driver reports high stress (above 6)
    if history.avg_stress_level > 6.0:
        adjustment += config.w_stress * (history.avg_stress_level - 6.0)
    
    # Small discount if driver has good fairness rating (above 3)
    # This represents a resilient driver who handles routes well
    if history.avg_fairness_rating > 3.0:
        adjustment -= config.w_fairness * (history.avg_fairness_rating - 3.0)
    
    return round(adjustment, 2)


def compute_history_features_for_drivers_sync(
    driver_ids: List[str],
    window_days: int = 7,
) -> Dict[str, DriverHistoryFeatures]:
    """
    Synchronous version that returns empty features for now.
    
    In LangGraph nodes (which are sync), we don't have easy DB access.
    This function returns default features. Full history lookup requires
    async DB session.
    
    For full history features with DB:
      - Use compute_history_features_for_drivers() with async session
      - Or pass pre-computed features through AllocationState
    
    Args:
        driver_ids: List of driver IDs
        window_days: Not used in sync version
    
    Returns:
        Dict mapping driver_id to default DriverHistoryFeatures
    """
    features = {}
    for d_id in driver_ids:
        # Return default features (neutral history)
        features[d_id] = DriverHistoryFeatures(
            driver_id=d_id,
            recent_avg_effort=0.0,  # Will use global stats
            recent_std_effort=0.0,
            recent_hard_days=0,
            avg_stress_level=5.0,  # Neutral
            avg_fairness_rating=3.0,  # Neutral
            avg_tiredness_level=2.5,  # Neutral
            total_assignments=0,
            total_feedback_count=0,
            fatigue_score=2.5,  # Neutral
            is_high_stress_driver=False,
            is_frequent_hard_days=False,
        )
    return features
