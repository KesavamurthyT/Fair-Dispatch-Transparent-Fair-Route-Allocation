"""
Recovery Service for Phase 7.
Handles complexity debt tracking and recovery day calculations.
"""

from datetime import date, timedelta
from typing import Dict, Optional, List
from uuid import UUID
import statistics

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Driver, Assignment, DriverStatsDaily, AllocationRun, FairnessConfig


async def get_driver_recovery_targets(
    db: AsyncSession,
    driver_ids: List[UUID],
    target_date: date,
    config: Optional[FairnessConfig] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute recovery effort targets for drivers based on complexity debt.
    
    Args:
        db: Database session
        driver_ids: List of driver IDs to check
        target_date: Date for the allocation
        config: FairnessConfig (fetched if not provided)
    
    Returns:
        Dict mapping driver_id (str) to max effort target (None if no recovery needed)
    """
    # Get active config if not provided
    if config is None:
        config_result = await db.execute(
            select(FairnessConfig)
            .where(FairnessConfig.is_active == True)
            .limit(1)
        )
        config = config_result.scalar_one_or_none()
    
    if not config or not config.recovery_mode_enabled:
        return {str(did): None for did in driver_ids}
    
    recovery_targets: Dict[str, Optional[float]] = {}
    lookback_days = 7
    
    for driver_id in driver_ids:
        # Get recent stats
        stats_result = await db.execute(
            select(DriverStatsDaily)
            .where(
                and_(
                    DriverStatsDaily.driver_id == driver_id,
                    DriverStatsDaily.date >= target_date - timedelta(days=lookback_days),
                    DriverStatsDaily.date < target_date,
                )
            )
            .order_by(DriverStatsDaily.date.desc())
        )
        recent_stats = list(stats_result.scalars().all())
        
        if not recent_stats:
            recovery_targets[str(driver_id)] = None
            continue
        
        # Get latest complexity debt
        latest_debt = recent_stats[0].complexity_debt if recent_stats else 0.0
        
        # Calculate recent average effort
        recent_efforts = [s.avg_workload_score for s in recent_stats if s.avg_workload_score > 0]
        
        if not recent_efforts:
            recovery_targets[str(driver_id)] = None
            continue
        
        recent_avg = statistics.mean(recent_efforts)
        
        # Check if driver needs recovery
        if latest_debt >= config.complexity_debt_hard_threshold:
            # Set recovery target as a fraction of recent average
            recovery_targets[str(driver_id)] = recent_avg * config.recovery_lightening_factor
        else:
            recovery_targets[str(driver_id)] = None
    
    return recovery_targets


async def update_daily_stats_for_run(
    db: AsyncSession,
    allocation_run_id: UUID,
    target_date: date,
    config: Optional[FairnessConfig] = None,
) -> None:
    """
    Update DriverStatsDaily after an allocation run.
    Computes is_hard_day, complexity_debt, and is_recovery_day.
    
    Args:
        db: Database session
        allocation_run_id: The allocation run that just completed
        target_date: Date of the allocation
        config: FairnessConfig (fetched if not provided)
    """
    # Get active config if not provided
    if config is None:
        config_result = await db.execute(
            select(FairnessConfig)
            .where(FairnessConfig.is_active == True)
            .limit(1)
        )
        config = config_result.scalar_one_or_none()
    
    # Get all assignments for this run
    assignments_result = await db.execute(
        select(Assignment)
        .where(Assignment.allocation_run_id == allocation_run_id)
    )
    assignments = list(assignments_result.scalars().all())
    
    if not assignments:
        return
    
    # Calculate effort statistics
    all_efforts = [a.workload_score for a in assignments]
    avg_effort = statistics.mean(all_efforts)
    std_effort = statistics.stdev(all_efforts) if len(all_efforts) > 1 else 0.0
    
    # Hard day threshold: effort > avg + k * std
    k = 0.5  # Configurable threshold factor
    hard_day_threshold = avg_effort + k * std_effort
    
    for assignment in assignments:
        driver_id = assignment.driver_id
        effort = assignment.workload_score
        
        # Get yesterday's stats for complexity debt
        yesterday = target_date - timedelta(days=1)
        yesterday_result = await db.execute(
            select(DriverStatsDaily)
            .where(
                and_(
                    DriverStatsDaily.driver_id == driver_id,
                    DriverStatsDaily.date == yesterday,
                )
            )
            .limit(1)
        )
        yesterday_stats = yesterday_result.scalar_one_or_none()
        previous_debt = yesterday_stats.complexity_debt if yesterday_stats else 0.0
        
        # Determine if hard day
        is_hard = effort > hard_day_threshold
        
        # Update complexity debt
        if is_hard:
            new_debt = previous_debt + 1.0
        else:
            new_debt = max(0.0, previous_debt - 0.5)
        
        # Determine if recovery day
        # Recovery if debt was high but effort is significantly below recent average
        is_recovery = False
        if config and config.recovery_mode_enabled:
            if previous_debt >= config.complexity_debt_hard_threshold:
                # Get recent average
                recent_result = await db.execute(
                    select(DriverStatsDaily.avg_workload_score)
                    .where(
                        and_(
                            DriverStatsDaily.driver_id == driver_id,
                            DriverStatsDaily.date >= target_date - timedelta(days=7),
                            DriverStatsDaily.date < target_date,
                        )
                    )
                )
                recent_efforts = [e for (e,) in recent_result.all() if e > 0]
                if recent_efforts:
                    recent_avg = statistics.mean(recent_efforts)
                    recovery_threshold = recent_avg * config.recovery_lightening_factor
                    is_recovery = effort <= recovery_threshold
                    
                    # Bonus debt reduction for recovery
                    if is_recovery:
                        new_debt = max(0.0, previous_debt - 1.0)
        
        # Check if stats already exist for today
        existing_result = await db.execute(
            select(DriverStatsDaily)
            .where(
                and_(
                    DriverStatsDaily.driver_id == driver_id,
                    DriverStatsDaily.date == target_date,
                )
            )
            .limit(1)
        )
        existing_stats = existing_result.scalar_one_or_none()
        
        if existing_stats:
            # Update existing
            existing_stats.avg_workload_score = effort
            existing_stats.is_hard_day = is_hard
            existing_stats.complexity_debt = new_debt
            existing_stats.is_recovery_day = is_recovery
        else:
            # Create new
            new_stats = DriverStatsDaily(
                driver_id=driver_id,
                date=target_date,
                avg_workload_score=effort,
                total_routes=1,
                is_hard_day=is_hard,
                complexity_debt=new_debt,
                is_recovery_day=is_recovery,
            )
            db.add(new_stats)
    
    await db.flush()


def calculate_recovery_penalty(
    effort: float,
    recovery_target: Optional[float],
    penalty_weight: float = 3.0,
) -> float:
    """
    Calculate penalty for exceeding recovery target effort.
    
    Args:
        effort: Proposed effort score
        recovery_target: Max effort target for recovery (None if not in recovery)
        penalty_weight: Weight for penalty calculation
    
    Returns:
        Additional penalty to add to cost function
    """
    if recovery_target is None:
        return 0.0
    
    if effort <= recovery_target:
        return 0.0
    
    # Penalty increases sharply for effort above target
    excess = effort - recovery_target
    return penalty_weight * excess


async def get_driver_context_for_recovery(
    db: AsyncSession,
    driver_id: UUID,
    target_date: date,
) -> dict:
    """
    Get driver's recovery context for explainability.
    
    Returns dict with:
        - recent_avg_effort
        - recent_hard_days
        - complexity_debt
        - is_in_recovery_mode
    """
    yesterday = target_date - timedelta(days=1)
    
    # Get recent stats
    stats_result = await db.execute(
        select(DriverStatsDaily)
        .where(
            and_(
                DriverStatsDaily.driver_id == driver_id,
                DriverStatsDaily.date >= target_date - timedelta(days=7),
                DriverStatsDaily.date < target_date,
            )
        )
        .order_by(DriverStatsDaily.date.desc())
    )
    recent_stats = list(stats_result.scalars().all())
    
    if not recent_stats:
        return {
            "recent_avg_effort": 0.0,
            "recent_hard_days": 0,
            "complexity_debt": 0.0,
            "is_in_recovery_mode": False,
        }
    
    efforts = [s.avg_workload_score for s in recent_stats if s.avg_workload_score > 0]
    hard_days = sum(1 for s in recent_stats if s.is_hard_day)
    latest_debt = recent_stats[0].complexity_debt
    
    return {
        "recent_avg_effort": statistics.mean(efforts) if efforts else 0.0,
        "recent_hard_days": hard_days,
        "complexity_debt": latest_debt,
        "is_in_recovery_mode": latest_debt >= 2.0,
    }
