"""Services package initialization."""

from app.services.clustering import cluster_packages, ClusterResult
from app.services.workload import calculate_workload, calculate_route_difficulty, estimate_route_time
from app.services.fairness import (
    gini_index,
    calculate_fairness_score,
    calculate_global_fairness,
    FairnessMetrics,
)
from app.services.allocation import allocate_routes, AllocationResult
from app.services.explainability import generate_explanation

__all__ = [
    "cluster_packages",
    "ClusterResult",
    "calculate_workload",
    "calculate_route_difficulty",
    "estimate_route_time",
    "gini_index",
    "calculate_fairness_score",
    "calculate_global_fairness",
    "FairnessMetrics",
    "allocate_routes",
    "AllocationResult",
    "generate_explanation",
]
