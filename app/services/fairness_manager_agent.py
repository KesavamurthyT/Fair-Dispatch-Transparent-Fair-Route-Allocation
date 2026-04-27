"""
Fairness Manager Agent - Evaluates fairness metrics and decides on re-optimization.
Phase 4.1 implementation with configurable thresholds.
"""

import statistics
from typing import Dict, List, Literal, Optional

from app.schemas.agent_schemas import (
    FairnessCheckResult,
    FairnessMetrics,
    FairnessRecommendations,
    FairnessThresholds,
    RoutePlanResult,
)


class FairnessManagerAgent:
    """
    Fairness Manager Agent evaluates allocation fairness and decides
    whether to accept or request re-optimization.
    
    Metrics computed:
    - Gini index (inequality measure)
    - Standard deviation  
    - Max gap (max - min effort)
    - Outlier count
    """
    
    def __init__(self, thresholds: Optional[FairnessThresholds] = None):
        """
        Initialize agent with configurable thresholds.
        
        Args:
            thresholds: Custom thresholds. Uses defaults if not provided.
        """
        self.thresholds = thresholds or FairnessThresholds()
    
    def check(
        self,
        plan_result: RoutePlanResult,
        proposal_number: int = 1,
    ) -> FairnessCheckResult:
        """
        Check fairness of a route plan and decide ACCEPT or REOPTIMIZE.
        
        Args:
            plan_result: Result from RoutePlannerAgent
            proposal_number: Which proposal is being checked (1 or 2)
            
        Returns:
            FairnessCheckResult with status, metrics, and recommendations
        """
        per_driver_effort = plan_result.per_driver_effort
        
        if not per_driver_effort:
            return FairnessCheckResult(
                status="ACCEPT",
                metrics=FairnessMetrics(
                    avg_effort=0.0,
                    std_dev=0.0,
                    max_gap=0.0,
                    gini_index=0.0,
                    min_effort=0.0,
                    max_effort=0.0,
                    outlier_count=0,
                    pct_above_avg=0.0,
                ),
                proposal_number=proposal_number,
                thresholds_used=self.thresholds.model_dump(),
            )
        
        efforts = list(per_driver_effort.values())
        
        # Compute metrics
        metrics = self._compute_metrics(efforts)
        
        # Make decision
        status = self._decide(metrics)
        
        # Generate recommendations if re-optimization needed
        recommendations = None
        if status == "REOPTIMIZE":
            recommendations = self._generate_recommendations(
                per_driver_effort,
                metrics,
            )
        
        return FairnessCheckResult(
            status=status,
            metrics=metrics,
            recommendations=recommendations,
            proposal_number=proposal_number,
            thresholds_used=self.thresholds.model_dump(),
        )
    
    def _compute_metrics(self, efforts: List[float]) -> FairnessMetrics:
        """Compute all fairness metrics from effort values."""
        if not efforts:
            return FairnessMetrics(
                avg_effort=0.0,
                std_dev=0.0,
                max_gap=0.0,
                gini_index=0.0,
                min_effort=0.0,
                max_effort=0.0,
            )
        
        n = len(efforts)
        
        # Basic stats
        avg_effort = statistics.mean(efforts)
        min_effort = min(efforts)
        max_effort = max(efforts)
        max_gap = max_effort - min_effort
        
        # Standard deviation
        std_dev = statistics.stdev(efforts) if n > 1 else 0.0
        
        # Gini index
        gini_index = self._compute_gini(efforts)
        
        # Outliers (above avg + 2 * std_dev)
        threshold = avg_effort + 2 * std_dev if std_dev > 0 else avg_effort * 1.5
        outlier_count = sum(1 for e in efforts if e > threshold)
        
        # Percentage above average
        pct_above_avg = (sum(1 for e in efforts if e > avg_effort) / n) * 100 if n > 0 else 0.0
        
        return FairnessMetrics(
            avg_effort=round(avg_effort, 2),
            std_dev=round(std_dev, 2),
            max_gap=round(max_gap, 2),
            gini_index=round(gini_index, 4),
            min_effort=round(min_effort, 2),
            max_effort=round(max_effort, 2),
            outlier_count=outlier_count,
            pct_above_avg=round(pct_above_avg, 1),
        )
    
    def _compute_gini(self, values: List[float]) -> float:
        """
        Compute Gini coefficient.
        
        Formula: Gini = sum(|xi - xj|) / (2 * n^2 * mean)
        
        Returns value between 0 (perfect equality) and 1 (perfect inequality).
        """
        if not values:
            return 0.0
        
        n = len(values)
        if n == 1:
            return 0.0
        
        mean = statistics.mean(values)
        if mean == 0:
            return 0.0
        
        # Compute sum of absolute differences
        total_diff = 0.0
        for i in range(n):
            for j in range(n):
                total_diff += abs(values[i] - values[j])
        
        gini = total_diff / (2 * n * n * mean)
        
        return min(gini, 1.0)  # Cap at 1.0
    
    def _decide(self, metrics: FairnessMetrics) -> Literal["ACCEPT", "REOPTIMIZE"]:
        """Decide whether to accept or request re-optimization."""
        thresholds = self.thresholds
        
        # Check all three thresholds
        gini_ok = metrics.gini_index <= thresholds.gini_threshold
        stddev_ok = metrics.std_dev <= thresholds.stddev_threshold
        maxgap_ok = metrics.max_gap <= thresholds.max_gap_threshold
        
        if gini_ok and stddev_ok and maxgap_ok:
            return "ACCEPT"
        else:
            return "REOPTIMIZE"
    
    def _generate_recommendations(
        self,
        per_driver_effort: Dict[str, float],
        metrics: FairnessMetrics,
    ) -> FairnessRecommendations:
        """Generate recommendations for re-optimization."""
        avg = metrics.avg_effort
        std = metrics.std_dev if metrics.std_dev > 0 else avg * 0.15
        
        # Identify high-effort drivers (above avg + 1.0 * std_dev)
        threshold = avg + 1.0 * std
        high_effort_drivers = [
            driver_id for driver_id, effort in per_driver_effort.items()
            if effort > threshold
        ]
        
        # Calculate penalty factor based on how far above threshold we are
        gini_ratio = metrics.gini_index / self.thresholds.gini_threshold if self.thresholds.gini_threshold > 0 else 1.0
        penalty_factor = min(2.0, 1.0 + (gini_ratio - 1.0) * 0.5)
        penalty_factor = max(1.2, penalty_factor)  # At least 1.2x penalty
        
        return FairnessRecommendations(
            penalize_high_effort_drivers=len(high_effort_drivers) > 0,
            high_effort_driver_ids=high_effort_drivers,
            penalty_factor=round(penalty_factor, 2),
            target_max_gap=self.thresholds.max_gap_threshold,
        )
    
    def get_input_snapshot(
        self,
        plan_result: RoutePlanResult,
    ) -> dict:
        """Generate input snapshot for DecisionLog."""
        efforts = list(plan_result.per_driver_effort.values())
        return {
            "proposal_number": plan_result.proposal_number,
            "num_drivers": len(plan_result.per_driver_effort),
            "total_effort": plan_result.total_effort,
            "avg_effort": plan_result.avg_effort,
            "effort_range": [min(efforts), max(efforts)] if efforts else [0, 0],
            "thresholds": self.thresholds.model_dump(),
        }
    
    def get_output_snapshot(self, result: FairnessCheckResult) -> dict:
        """Generate output snapshot for DecisionLog."""
        output = {
            "status": result.status,
            "gini_index": result.metrics.gini_index,
            "std_dev": result.metrics.std_dev,
            "max_gap": result.metrics.max_gap,
            "outlier_count": result.metrics.outlier_count,
        }
        
        if result.recommendations:
            output["num_high_effort_drivers"] = len(result.recommendations.high_effort_driver_ids)
            output["penalty_factor"] = result.recommendations.penalty_factor
        
        return output
