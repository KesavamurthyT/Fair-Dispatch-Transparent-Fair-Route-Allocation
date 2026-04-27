"""
Driver Liaison Agent - Per-driver negotiation logic.
Phase 4.2 implementation for reviewing and countering route assignments.
Now enhanced with historical features for personalized decision-making.
"""

from typing import Dict, List, Optional, Tuple

from app.models.driver import Driver
from app.models.route import Route
from app.schemas.agent_schemas import (
    DriverAssignmentProposal,
    DriverContext,
    DriverLiaisonDecision,
    EffortMatrixResult,
    NegotiationResult,
)
from app.services.history_features import DriverHistoryFeatures


class DriverLiaisonAgent:
    """
    Driver Liaison Agent reviews proposed assignments on behalf of each driver.
    
    Decision rules:
    - ACCEPT: Effort is within driver's comfort band
    - COUNTER: Effort too high but a better alternative exists
    - FORCE_ACCEPT: Effort too high but no fair alternative available
    
    The agent considers:
    - Driver's recent average effort and variability
    - Hard day streaks and fatigue
    - Global team average for context
    - Historical stress level and fairness ratings
    """
    
    # Default thresholds (can be configured)
    HIGH_STREAK_THRESHOLD_DAYS: int = 3
    MINIMUM_IMPROVEMENT_PCT: float = 0.10  # 10% lighter to counter
    FAIRNESS_EPSILON: float = 0.05  # Tolerance for swap acceptance
    
    def __init__(
        self,
        high_streak_threshold: int = 3,
        minimum_improvement_pct: float = 0.10,
    ):
        """
        Initialize agent with configurable thresholds.
        
        Args:
            high_streak_threshold: Days of hard work before tightening comfort band
            minimum_improvement_pct: Required improvement % for counter suggestion
        """
        self.HIGH_STREAK_THRESHOLD_DAYS = high_streak_threshold
        self.MINIMUM_IMPROVEMENT_PCT = minimum_improvement_pct
    
    def decide_for_driver(
        self,
        proposal: DriverAssignmentProposal,
        context: DriverContext,
        global_avg_effort: float,
        global_std_effort: float,
        alternative_routes_sorted: List[Tuple[str, float]],
    ) -> DriverLiaisonDecision:
        """
        Make a decision for a single driver's proposed assignment.
        
        Args:
            proposal: The proposed assignment (driver_id, route_id, effort, rank)
            context: Driver's historical context and preferences
            global_avg_effort: Team average effort for this run
            global_std_effort: Team std dev of effort for this run
            alternative_routes_sorted: List of (route_id, effort) sorted by effort ascending,
                                       excluding the currently assigned route
        
        Returns:
            DriverLiaisonDecision with ACCEPT, COUNTER, or FORCE_ACCEPT
        """
        # Compute comfort upper bound
        # Base: recent_avg + max(global_std, recent_std)
        std_to_use = max(global_std_effort, context.recent_std_effort)
        comfort_upper = context.recent_avg_effort + std_to_use
        
        # Tighten threshold for fatigued or streaking drivers
        if context.recent_hard_days >= self.HIGH_STREAK_THRESHOLD_DAYS:
            comfort_upper -= 0.3 * global_std_effort
        
        if context.fatigue_score >= 4.0:
            comfort_upper -= 0.2 * global_std_effort
        
        # Ensure comfort_upper doesn't go below a reasonable minimum
        min_comfort = context.recent_avg_effort * 0.7
        comfort_upper = max(comfort_upper, min_comfort)
        
        # Decision 1: ACCEPT if within comfort band
        if proposal.effort <= comfort_upper:
            return DriverLiaisonDecision(
                driver_id=proposal.driver_id,
                decision="ACCEPT",
                reason=f"Effort {proposal.effort:.1f} within comfort band (≤{comfort_upper:.1f})"
            )
        
        # Effort is above comfort - look for alternatives
        required_max_effort = proposal.effort * (1.0 - self.MINIMUM_IMPROVEMENT_PCT)
        
        best_alternative: Optional[str] = None
        best_effort: Optional[float] = None
        
        for route_id, route_effort in alternative_routes_sorted:
            # Route must be at least X% lighter
            if route_effort <= required_max_effort:
                # Avoid giving extremely easy routes to drivers who often get easy ones
                # Simple heuristic: don't suggest routes much below global avg for high-rank drivers
                if proposal.rank_in_team <= 2 and route_effort < global_avg_effort * 0.5:
                    continue  # Skip extremely easy routes for drivers with hard assignments
                
                best_alternative = route_id
                best_effort = route_effort
                break  # Take first valid alternative (sorted by ascending effort)
        
        # Decision 2: COUNTER if valid alternative exists
        if best_alternative is not None:
            return DriverLiaisonDecision(
                driver_id=proposal.driver_id,
                decision="COUNTER",
                preferred_route_id=best_alternative,
                reason=f"Effort {proposal.effort:.1f} exceeds comfort ({comfort_upper:.1f}); "
                       f"prefer route with effort {best_effort:.1f}"
            )
        
        # Decision 3: FORCE_ACCEPT if no fair alternative
        return DriverLiaisonDecision(
            driver_id=proposal.driver_id,
            decision="FORCE_ACCEPT",
            reason=f"Effort {proposal.effort:.1f} exceeds comfort ({comfort_upper:.1f}) "
                   f"but no fair alternative available"
        )
    
    def run_for_all_drivers(
        self,
        proposals: List[DriverAssignmentProposal],
        driver_contexts: Dict[str, DriverContext],
        effort_matrix: List[List[float]],
        driver_ids: List[str],
        route_ids: List[str],
        global_avg_effort: float,
        global_std_effort: float,
    ) -> NegotiationResult:
        """
        Run liaison decisions for all drivers.
        
        Args:
            proposals: List of DriverAssignmentProposal for each driver
            driver_contexts: Dict mapping driver_id -> DriverContext
            effort_matrix: 2D effort matrix [driver_idx][route_idx]
            driver_ids: List of driver IDs in matrix row order
            route_ids: List of route IDs in matrix column order
            global_avg_effort: Team average effort
            global_std_effort: Team std dev of effort
        
        Returns:
            NegotiationResult with all decisions
        """
        decisions: List[DriverLiaisonDecision] = []
        
        # Build index maps
        driver_idx_map = {did: idx for idx, did in enumerate(driver_ids)}
        
        # Process each proposal
        for proposal in proposals:
            driver_idx = driver_idx_map.get(proposal.driver_id)
            if driver_idx is None:
                # Driver not found in matrix - skip
                continue
            
            # Build alternative routes for this driver (sorted by effort ascending)
            alternatives: List[Tuple[str, float]] = []
            for route_idx, rid in enumerate(route_ids):
                if rid != proposal.route_id:
                    alternatives.append((rid, effort_matrix[driver_idx][route_idx]))
            
            alternatives.sort(key=lambda x: x[1])  # Sort by effort ascending
            
            # Get driver context (use defaults if missing)
            context = driver_contexts.get(
                proposal.driver_id,
                DriverContext(
                    driver_id=proposal.driver_id,
                    recent_avg_effort=global_avg_effort,
                    recent_std_effort=global_std_effort,
                    recent_hard_days=0,
                    fatigue_score=3.0,
                    preferences={},
                )
            )
            
            # Make decision
            decision = self.decide_for_driver(
                proposal=proposal,
                context=context,
                global_avg_effort=global_avg_effort,
                global_std_effort=global_std_effort,
                alternative_routes_sorted=alternatives,
            )
            decisions.append(decision)
        
        # Count decision types
        num_accept = sum(1 for d in decisions if d.decision == "ACCEPT")
        num_counter = sum(1 for d in decisions if d.decision == "COUNTER")
        num_force_accept = sum(1 for d in decisions if d.decision == "FORCE_ACCEPT")
        
        return NegotiationResult(
            decisions=decisions,
            num_accept=num_accept,
            num_counter=num_counter,
            num_force_accept=num_force_accept,
        )
    
    def get_input_snapshot(
        self,
        proposals: List[DriverAssignmentProposal],
        global_avg_effort: float,
        global_std_effort: float,
    ) -> dict:
        """Generate input snapshot for DecisionLog."""
        efforts = [p.effort for p in proposals]
        return {
            "num_drivers": len(proposals),
            "global_avg_effort": round(global_avg_effort, 2),
            "global_std_effort": round(global_std_effort, 2),
            "effort_range": [round(min(efforts), 2), round(max(efforts), 2)] if efforts else [0, 0],
            "thresholds": {
                "high_streak_days": self.HIGH_STREAK_THRESHOLD_DAYS,
                "min_improvement_pct": self.MINIMUM_IMPROVEMENT_PCT,
            },
        }
    
    def get_output_snapshot(self, result: NegotiationResult) -> dict:
        """Generate output snapshot for DecisionLog."""
        # Include sample of decisions (first 3)
        sample_decisions = [
            {
                "driver_id": d.driver_id,
                "decision": d.decision,
                "preferred_route": d.preferred_route_id,
            }
            for d in result.decisions[:3]
        ]
        
        return {
            "num_accept": result.num_accept,
            "num_counter": result.num_counter,
            "num_force_accept": result.num_force_accept,
            "sample_decisions": sample_decisions,
        }
    
    def decide_with_history(
        self,
        proposal: DriverAssignmentProposal,
        history: Optional[DriverHistoryFeatures],
        global_avg_effort: float,
        global_std_effort: float,
        alternative_routes_sorted: List[Tuple[str, float]],
    ) -> DriverLiaisonDecision:
        """
        Make a decision using historical features directly.
        
        This is an alternative interface that takes DriverHistoryFeatures
        instead of DriverContext, using feedback-based stress and fairness.
        
        Args:
            proposal: The proposed assignment
            history: Driver's historical features (can be None for new drivers)
            global_avg_effort: Team average effort for this run
            global_std_effort: Team std dev of effort for this run
            alternative_routes_sorted: List of (route_id, effort) sorted by effort ascending
        
        Returns:
            DriverLiaisonDecision with ACCEPT, COUNTER, or FORCE_ACCEPT
        """
        # If no history, use defaults
        if history is None:
            recent_avg = global_avg_effort
            recent_std = global_std_effort
            hard_days = 0
            fatigue = 3.0
            avg_stress = 5.0
        else:
            recent_avg = history.recent_avg_effort if history.recent_avg_effort > 0 else global_avg_effort
            recent_std = history.recent_std_effort if history.recent_std_effort > 0 else global_std_effort
            hard_days = history.recent_hard_days
            fatigue = history.fatigue_score
            avg_stress = history.avg_stress_level
        
        # Compute comfort upper bound
        std_to_use = max(global_std_effort, recent_std)
        comfort_upper = recent_avg + std_to_use
        
        # Tighten threshold for drivers with many hard days
        if hard_days >= self.HIGH_STREAK_THRESHOLD_DAYS:
            comfort_upper -= 0.3 * global_std_effort
        
        # Tighten for fatigued drivers
        if fatigue >= 4.0:
            comfort_upper -= 0.2 * global_std_effort
        
        # Tighten for high stress drivers
        if avg_stress >= 7.0:
            comfort_upper -= 0.2 * global_std_effort
        
        # Ensure comfort_upper doesn't go below a reasonable minimum
        min_comfort = recent_avg * 0.6
        comfort_upper = max(comfort_upper, min_comfort)
        
        # Decision 1: ACCEPT if within comfort band
        if proposal.effort <= comfort_upper:
            reason = f"Effort {proposal.effort:.1f} within comfort band (≤{comfort_upper:.1f})"
            if history and hard_days >= 2:
                reason += f". Note: driver had {hard_days} hard days recently."
            return DriverLiaisonDecision(
                driver_id=proposal.driver_id,
                decision="ACCEPT",
                reason=reason
            )
        
        # Effort is above comfort - look for alternatives
        required_max_effort = proposal.effort * (1.0 - self.MINIMUM_IMPROVEMENT_PCT)
        
        best_alternative: Optional[str] = None
        best_effort: Optional[float] = None
        
        for route_id, route_effort in alternative_routes_sorted:
            if route_effort <= required_max_effort:
                # Avoid giving extremely easy routes to drivers with hard assignments
                if proposal.rank_in_team <= 2 and route_effort < global_avg_effort * 0.5:
                    continue
                best_alternative = route_id
                best_effort = route_effort
                break
        
        # Decision 2: COUNTER if valid alternative exists
        if best_alternative is not None:
            reason = f"Effort {proposal.effort:.1f} exceeds comfort ({comfort_upper:.1f}); prefer route with effort {best_effort:.1f}"
            if history and avg_stress >= 7:
                reason += f" (high stress history: {avg_stress:.1f}/10)"
            return DriverLiaisonDecision(
                driver_id=proposal.driver_id,
                decision="COUNTER",
                preferred_route_id=best_alternative,
                reason=reason
            )
        
        # Decision 3: FORCE_ACCEPT if no fair alternative
        reason = f"Effort {proposal.effort:.1f} exceeds comfort ({comfort_upper:.1f}) but no fair alternative available"
        if history and hard_days >= 3:
            reason += f" (warning: {hard_days} hard days in past week)"
        return DriverLiaisonDecision(
            driver_id=proposal.driver_id,
            decision="FORCE_ACCEPT",
            reason=reason
        )
