"""
Explainability Agent v2 (Phase 4.3).
Generates deterministic, template-based explanations for route assignments.

Features:
- Driver-facing explanations: Short, simple language (1-3 sentences)
- Admin-facing explanations: Detailed with metrics and context
- Category classification for explanation templates
"""

from typing import Dict, Any, List
from app.schemas.explainability import DriverExplanationInput, DriverExplanationOutput


class ExplainabilityAgent:
    """
    Generates explanations for driver assignments using template-based logic.
    
    Categories:
    - NEAR_AVG: Effort close to team average
    - HEAVY: Above average effort
    - HEAVY_WITH_SWAP: Above average, but swap was applied
    - HEAVY_NO_SWAP: Above average, counter requested but no swap possible
    - RECOVERY: Intentional light day for recovery
    - LIGHT_RECOVERY: Light day after hard streak
    - LIGHT: Below average effort
    - LEARNING_OPTIMIZED: Assignment uses personalized ML model (Phase 8)
    """
    
    def build_explanation_for_driver(
        self,
        data: DriverExplanationInput
    ) -> DriverExplanationOutput:
        """
        Build both driver and admin explanations for a single driver.
        
        Args:
            data: Complete input data including effort, history, and negotiation context
        
        Returns:
            DriverExplanationOutput with driver_explanation, admin_explanation, category
        """
        category = self._classify_category(data)
        driver_text = self._build_driver_text(data, category)
        admin_text = self._build_admin_text(data, category)
        
        return DriverExplanationOutput(
            driver_explanation=driver_text,
            admin_explanation=admin_text,
            category=category,
        )
    
    def _classify_category(self, data: DriverExplanationInput) -> str:
        """
        Classify the assignment into a category for template selection.
        
        Categories drive which text template is used.
        """
        # Compute effort band
        delta_vs_avg = data.today_effort - data.global_avg_effort
        percent_vs_avg = (delta_vs_avg / max(data.global_avg_effort, 1.0)) * 100.0
        
        if abs(percent_vs_avg) <= 10:
            band = "NEAR_AVG"
        elif percent_vs_avg > 10:
            band = "ABOVE_AVG"
        else:
            band = "BELOW_AVG"
        
        # Priority-based classification
        # Phase 8: Check for personalized learning model first
        if data.personalized_model_version is not None and data.personalized_model_version > 0:
            # If model has low MSE (good predictions), highlight learning
            if data.personalized_model_mse is not None and data.personalized_model_mse < 15.0:
                return "LEARNING_OPTIMIZED"
        
        if data.is_recovery_day:
            return "RECOVERY"
        
        if band == "ABOVE_AVG":
            if data.swap_applied:
                return "HEAVY_WITH_SWAP"
            if data.liaison_decision in ("COUNTER", "FORCE_ACCEPT"):
                return "HEAVY_NO_SWAP"
            return "HEAVY"
        
        if band == "BELOW_AVG":
            if data.history_hard_days_last_7 >= 2:
                return "LIGHT_RECOVERY"
            return "LIGHT"
        
        return "NEAR_AVG"
    
    def _build_driver_text(self, data: DriverExplanationInput, category: str) -> str:
        """
        Build driver-facing explanation (1-3 sentences, simple language).
        """
        # Extract route summary values
        num_packages = data.route_summary.get("num_packages", 0)
        total_weight_kg = data.route_summary.get("total_weight_kg", 0.0)
        num_stops = data.route_summary.get("num_stops", 0)
        eta_minutes = data.route_summary.get("estimated_time_minutes", 60)
        
        # Format time
        if eta_minutes >= 60:
            hours = eta_minutes // 60
            mins = eta_minutes % 60
            time_str = f"{hours}h {mins}m" if mins > 0 else f"{hours}h"
        else:
            time_str = f"{eta_minutes} minutes"
        
        # Category-specific templates
        if category == "NEAR_AVG":
            text = (
                f"Today you have a moderate route with {num_packages} packages "
                f"({total_weight_kg:.1f} kg) across {num_stops} stops, "
                f"estimated about {time_str}. "
                f"Your effort score is close to the team average, keeping workloads balanced."
            )
        
        elif category == "HEAVY_WITH_SWAP":
            text = (
                f"You received one of the heavier routes today with {num_packages} packages "
                f"({total_weight_kg:.1f} kg) and {num_stops} stops, taking around {time_str}. "
                f"The system adjusted other routes to keep overall fairness, "
                f"and your effort remains within agreed team limits."
            )
        
        elif category == "HEAVY_NO_SWAP":
            text = (
                f"Your route today is on the heavier side with {num_packages} packages "
                f"({total_weight_kg:.1f} kg) and {num_stops} stops, about {time_str}. "
                f"We couldn't find a lighter alternative without overloading teammates, "
                f"so this will be considered when planning tomorrow's route."
            )
        
        elif category == "HEAVY":
            text = (
                f"Today's route is heavier than average with {num_packages} packages "
                f"({total_weight_kg:.1f} kg) and {num_stops} stops, around {time_str}. "
                f"This will be factored into future allocations to maintain fairness."
            )
        
        elif category in ("RECOVERY", "LIGHT_RECOVERY"):
            text = (
                f"Today's route is intentionally lighter to help you recover after several busy days. "
                f"You have {num_packages} packages ({total_weight_kg:.1f} kg) and {num_stops} stops, "
                f"giving you a more balanced workload this week."
            )
        
        elif category == "LIGHT":
            text = (
                f"You have a lighter route today with {num_packages} packages "
                f"({total_weight_kg:.1f} kg) and {num_stops} stops, around {time_str}. "
                f"This helps balance out previous days and keeps the team's workload fair."
            )
        
        elif category == "LEARNING_OPTIMIZED":
            model_version = data.personalized_model_version or 1
            text = (
                f"Today's route uses your personalized workload model (v{model_version}), "
                f"tuned from your recent performance. "
                f"You have {num_packages} packages ({total_weight_kg:.1f} kg) and {num_stops} stops, "
                f"estimated at {time_str}. The system learns and adapts to your preferences over time."
            )
        
        else:
            # Fallback
            text = (
                f"Your route has {num_packages} packages ({total_weight_kg:.1f} kg) "
                f"and {num_stops} stops, estimated at {time_str}."
            )
        
        return text
    
    def _build_admin_text(self, data: DriverExplanationInput, category: str) -> str:
        """
        Build admin-facing explanation (detailed with metrics).
        """
        # Compute derived values
        percent_vs_avg = ((data.today_effort - data.global_avg_effort) / 
                          max(data.global_avg_effort, 1.0)) * 100.0
        
        # Effort breakdown percentages
        pe = data.effort_breakdown.get("physical_effort", 0.0)
        rc = data.effort_breakdown.get("route_complexity", 0.0)
        tp = data.effort_breakdown.get("time_pressure", 0.0)
        total_breakdown = max(pe + rc + tp, 0.001)
        pe_pct = round(pe / total_breakdown * 100)
        rc_pct = round(rc / total_breakdown * 100)
        tp_pct = round(tp / total_breakdown * 100)
        
        # Route summary
        num_packages = data.route_summary.get("num_packages", 0)
        num_stops = data.route_summary.get("num_stops", 0)
        difficulty = data.route_summary.get("difficulty_score", 
                     data.route_summary.get("route_difficulty_score", 0.0))
        
        # Base text
        lines = [
            f"Driver {data.driver_name} received route with effort {data.today_effort:.1f}, "
            f"which is {percent_vs_avg:+.1f}% relative to fleet average ({data.global_avg_effort:.1f}), "
            f"ranked {data.today_rank}/{data.num_drivers} in difficulty.",
            
            f"Route: {num_packages} packages, {num_stops} stops, difficulty {difficulty:.1f}.",
        ]
        
        # Add breakdown if available
        if total_breakdown > 0.01:
            lines.append(
                f"Effort composition: ~{pe_pct}% physical load, "
                f"~{rc_pct}% route complexity, ~{tp_pct}% time pressure."
            )
        
        # Global metrics
        lines.append(
            f"Global fairness: Gini {data.global_gini_index:.3f}, "
            f"std dev {data.global_std_effort:.1f}, max gap {data.global_max_gap:.1f}."
        )
        
        # Category-specific additions
        if category == "RECOVERY":
            lines.append(
                f"Recovery day: intentionally lighter route after "
                f"{data.history_hard_days_last_7} hard days in the last week."
            )
        
        elif category == "LIGHT_RECOVERY":
            lines.append(
                f"Light assignment following {data.history_hard_days_last_7} above-average days recently."
            )
        
        elif category == "HEAVY_WITH_SWAP":
            lines.append(
                "A swap was applied during negotiation to reduce this driver's effort "
                "while maintaining fairness thresholds."
            )
        
        elif category == "HEAVY_NO_SWAP":
            lines.append(
                "Driver requested lighter route, but no alternative met fairness constraints "
                "without significantly overloading others. Flagged for future planning."
            )
        
        elif category == "HEAVY":
            if data.liaison_decision == "ACCEPT":
                lines.append(
                    "Driver accepted this heavier assignment within their comfort threshold."
                )
        
        # Manual override note
        if data.had_manual_override:
            lines.append("Note: This assignment includes a manual admin override.")
        
        # EV context (Phase 7)
        if data.is_ev_driver and data.ev_charging_overhead > 0:
            lines.append(
                f"EV driver: effort includes {data.ev_charging_overhead:.1f} points overhead "
                f"from battery range/charging constraints."
            )
        
        # Complexity debt note
        if data.complexity_debt >= 2.0:
            lines.append(
                f"Driver has complexity debt of {data.complexity_debt:.1f} "
                f"(threshold 2.0), eligible for recovery scheduling."
            )
        
        # Learning model note (Phase 8)
        if category == "LEARNING_OPTIMIZED" or data.personalized_model_version:
            version = data.personalized_model_version or "N/A"
            mse = f"{data.personalized_model_mse:.1f}" if data.personalized_model_mse else "N/A"
            lines.append(
                f"Personalized ML model v{version} used for effort prediction (MSE: {mse})."
            )
        
        return " ".join(lines)
    
    def get_input_snapshot(
        self,
        num_drivers: int,
        avg_effort: float,
        std_effort: float,
        gini_index: float,
        category_counts: Dict[str, int],
    ) -> dict:
        """Generate input snapshot for DecisionLog."""
        return {
            "num_drivers": num_drivers,
            "avg_effort": round(avg_effort, 2),
            "std_effort": round(std_effort, 2),
            "gini_index": round(gini_index, 4),
        }
    
    def get_output_snapshot(
        self,
        total_explanations: int,
        category_counts: Dict[str, int],
    ) -> dict:
        """Generate output snapshot for DecisionLog."""
        return {
            "total_explanations": total_explanations,
            "category_counts": category_counts,
        }


# ==================== Legacy Functions (Backward Compatibility) ====================

def generate_explanation(
    driver_name: str,
    route: Dict[str, Any],
    workload_score: float,
    avg_workload: float,
    gini_index: float,
) -> str:
    """
    Generate a plain English explanation for a driver's assignment.
    
    Legacy function maintained for backward compatibility.
    For new code, use ExplainabilityAgent.build_explanation_for_driver().
    """
    num_packages = route.get("num_packages", 0)
    total_weight = route.get("total_weight_kg", 0)
    num_stops = route.get("num_stops", 0)
    difficulty = route.get("route_difficulty_score", 1.0)
    time_minutes = route.get("estimated_time_minutes", 60)
    
    # Classify difficulty
    if difficulty < 1.5:
        difficulty_desc = "low"
    elif difficulty < 2.5:
        difficulty_desc = "moderate"
    elif difficulty < 3.5:
        difficulty_desc = "high"
    else:
        difficulty_desc = "very high"
    
    # Compare to average
    if avg_workload > 0:
        diff_from_avg = workload_score - avg_workload
        pct_diff = abs(diff_from_avg) / avg_workload * 100
    else:
        diff_from_avg = 0
        pct_diff = 0
    
    if abs(pct_diff) < 10:
        comparison = "close to the team average"
    elif diff_from_avg > 0:
        comparison = f"about {pct_diff:.0f}% above the team average"
    else:
        comparison = f"about {pct_diff:.0f}% below the team average"
    
    # Classify Gini
    if gini_index < 0.2:
        fairness_desc = "very well balanced"
    elif gini_index < 0.35:
        fairness_desc = "well balanced"
    elif gini_index < 0.5:
        fairness_desc = "reasonably balanced"
    else:
        fairness_desc = "less balanced than ideal"
    
    # Format time
    hours = time_minutes // 60
    mins = time_minutes % 60
    if hours > 0:
        time_str = f"{hours}h {mins}m"
    else:
        time_str = f"{mins} minutes"
    
    # Build explanation
    lines = [
        f"Your route has {num_packages} packages ({total_weight:.1f}kg), "
        f"{num_stops} stops, and {difficulty_desc} difficulty.",
        
        f"Estimated completion time is {time_str}.",
        
        f"Your workload score of {workload_score:.1f} is {comparison} "
        f"(team avg: {avg_workload:.1f}).",
        
        f"Today's overall fairness (Gini {gini_index:.2f}) indicates "
        f"loads are {fairness_desc}.",
    ]
    
    return " ".join(lines)


def generate_brief_explanation(
    workload_score: float,
    avg_workload: float,
    fairness_score: float,
) -> str:
    """
    Generate a brief one-line explanation.
    
    Legacy function maintained for backward compatibility.
    """
    if fairness_score >= 0.9:
        return f"Workload ({workload_score:.0f}) is very close to average ({avg_workload:.0f}). Fair assignment."
    elif fairness_score >= 0.7:
        return f"Workload ({workload_score:.0f}) is reasonably close to average ({avg_workload:.0f}). Good balance."
    elif workload_score > avg_workload:
        return f"Workload ({workload_score:.0f}) is above average ({avg_workload:.0f}). This will be balanced in future allocations."
    else:
        return f"Workload ({workload_score:.0f}) is below average ({avg_workload:.0f}). Lighter day today."
