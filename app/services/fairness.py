"""
Fairness metrics calculation service.
Implements Gini index, standard deviation, and per-driver fairness scores.
"""

from dataclasses import dataclass
from typing import List
import statistics


@dataclass
class FairnessMetrics:
    """Global fairness metrics for an allocation."""
    avg_workload: float
    std_dev: float
    gini_index: float


def gini_index(workloads: List[float]) -> float:
    """
    Calculate the Gini index for a distribution of workloads.
    
    The Gini index measures inequality in a distribution.
    - 0 = perfect equality (all workloads equal)
    - 1 = maximum inequality (one driver has all work)
    
    Formula: G = (2 * Σ(i * x_i)) / (n * Σx_i) - (n + 1) / n
    where x_i is sorted in ascending order and i starts from 1.
    
    Args:
        workloads: List of workload scores for each driver
    
    Returns:
        Gini index between 0 and 1
    """
    n = len(workloads)
    
    if n == 0:
        return 0.0
    
    total = sum(workloads)
    
    if total == 0:
        return 0.0
    
    # Sort workloads in ascending order
    sorted_workloads = sorted(workloads)
    
    # Calculate cumulative sum with weights
    cumulative = sum((i + 1) * w for i, w in enumerate(sorted_workloads))
    
    # Gini formula
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    
    # Ensure result is in valid range [0, 1]
    return max(0.0, min(1.0, round(gini, 4)))


def calculate_fairness_score(
    workload: float,
    avg_workload: float,
) -> float:
    """
    Calculate individual fairness score for a driver.
    
    Formula: fairness = 1 - |workload - avg| / max(avg, 1)
    
    A score of 1.0 means the workload is exactly average.
    Lower scores indicate deviation from average (either over or under-loaded).
    
    Args:
        workload: Individual driver's workload score
        avg_workload: Average workload across all drivers
    
    Returns:
        Fairness score between 0 and 1
    """
    if avg_workload <= 0:
        avg_workload = 1.0
    
    deviation = abs(workload - avg_workload)
    fairness = 1.0 - (deviation / max(avg_workload, 1.0))
    
    # Clamp to valid range [0, 1]
    return max(0.0, min(1.0, round(fairness, 4)))


def calculate_global_fairness(workloads: List[float]) -> FairnessMetrics:
    """
    Calculate global fairness metrics for an allocation.
    
    Args:
        workloads: List of workload scores for each driver
    
    Returns:
        FairnessMetrics with avg_workload, std_dev, and gini_index
    """
    if not workloads:
        return FairnessMetrics(
            avg_workload=0.0,
            std_dev=0.0,
            gini_index=0.0,
        )
    
    avg = statistics.mean(workloads)
    
    # Standard deviation (population, not sample)
    if len(workloads) > 1:
        std = statistics.pstdev(workloads)
    else:
        std = 0.0
    
    gini = gini_index(workloads)
    
    return FairnessMetrics(
        avg_workload=round(avg, 2),
        std_dev=round(std, 2),
        gini_index=gini,
    )
