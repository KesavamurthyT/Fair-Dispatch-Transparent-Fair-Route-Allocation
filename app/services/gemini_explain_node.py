"""
Gemini 1.5 Flash Explainability Node for LangGraph Integration.
Generates personalized Tamil/English explanations using LLM.
"""

import os
from typing import Dict, Any
from datetime import datetime

from app.schemas.allocation_state import AllocationState


async def gemini_explain_node(state: AllocationState) -> Dict[str, Any]:
    """
    LangGraph Node: Gemini 1.5 Flash personalized explanations.
    
    Generates natural language explanations in Tamil/English based on
    driver context, recovery status, EV considerations, and fairness metrics.
    
    Input: Full workflow state (effort/fairness/recovery/EV)
    Output: {"driver_id": {"driver_explanation": "...", "admin_explanation": "..."}}
    
    Falls back to template-based explanations on API error.
    """
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # No API key, return existing explanations unchanged
        return {}
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.prompts import PromptTemplate
    except ImportError:
        # LangChain Google GenAI not installed
        return {}
    
    # Initialize Gemini 3 Flash Preview
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.2,  # Consistent tone
        max_tokens=100,   # Keep explanations concise (<50 words)
    )
    
    # Rich prompt template with Tamil/English support
    prompt_template = PromptTemplate.from_template("""
Generate a friendly, personalized delivery route explanation.

DRIVER: {driver_name} ({exp_years} years experience{ev_status})
ROUTE TODAY: {stops} stops | {distance}km | {weight}kg load
EFFORT SCORE: Team average {team_avg:.0f} → Your route {today_effort:.0f} ({delta_pct:+.0f}%)
{recovery_note}
{fairness_note}
{ev_note}

LANGUAGE: {language}

Guidelines:
- Friendly & natural tone
- Maximum 50 words
- Actionable advice if needed
- No technical jargon
- End on a positive note

Generate the explanation:
""")
    
    final_proposal = state.final_proposal or state.route_proposal_1
    final_fairness = state.final_fairness or state.fairness_check_1
    metrics = final_fairness["metrics"]
    
    updated_explanations = state.explanations.copy()
    
    for alloc in final_proposal["allocation"]:
        driver_id = str(alloc["driver_id"])
        
        # Get existing explanation to enhance
        existing = state.explanations.get(driver_id, {})
        
        # Find driver info
        driver = next(
            (d for d in state.driver_models if str(d.get("id")) == driver_id),
            {}
        )
        
        # Find route info
        route_id = str(alloc["route_id"])
        route = next(
            (r for r in state.route_models if str(r.get("id")) == route_id),
            {}
        )
        
        # Get driver context
        driver_context = state.driver_contexts.get(driver_id, {})
        
        # Determine language preference
        preferred_lang = driver.get("preferred_language", "en")
        language = "Tamil" if preferred_lang == "ta" else "English"
        
        # Check EV status
        is_ev = driver.get("vehicle_type") == "EV" or driver.get("is_ev", False)
        
        # Check recovery status
        recovery_target = state.recovery_targets.get(driver_id)
        is_recovery = recovery_target is not None
        
        # Build context for prompt
        today_effort = alloc["effort"]
        team_avg = metrics["avg_effort"]
        delta_pct = ((today_effort / team_avg) - 1) * 100 if team_avg > 0 else 0
        
        context = {
            "driver_name": driver.get("name", "Driver"),
            "exp_years": driver.get("experience_years", 2),
            "ev_status": " - EV Driver" if is_ev else "",
            "stops": route.get("num_stops", 12),
            "distance": route.get("total_distance_km", 45),
            "weight": route.get("total_weight_kg", 48),
            "team_avg": team_avg,
            "today_effort": today_effort,
            "delta_pct": delta_pct,
            
            # Recovery note
            "recovery_note": (
                "🔋 RECOVERY DAY - Lighter route after a tough week."
                if is_recovery else ""
            ),
            
            # Fairness note
            "fairness_note": (
                "✅ Team workload perfectly balanced today!"
                if metrics["gini_index"] < 0.25
                else "Team fairness optimized."
            ),
            
            # EV note
            "ev_note": (
                "⚡ EV battery range verified - you're good to go!"
                if is_ev else ""
            ),
            
            "language": language,
        }
        
        try:
            # Generate explanation using Gemini
            chain = prompt_template | llm
            response = await chain.ainvoke(context)
            
            generated_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # Update explanation
            updated_explanations[driver_id] = {
                "driver_explanation": generated_text,
                "admin_explanation": f"Gemini 1.5 Flash ({language}, {len(generated_text)} chars) - {existing.get('category', 'NEAR_AVG')}",
                "category": existing.get("category", "NEAR_AVG"),
                "gemini_generated": True,
            }
            
        except Exception as e:
            # Fallback to existing template-based explanation
            updated_explanations[driver_id] = {
                **existing,
                "admin_explanation": f"{existing.get('admin_explanation', '')} [Gemini fallback: {str(e)[:50]}]",
                "gemini_generated": False,
            }
    
    # Create decision log entry
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "agent_name": "GEMINI_1_5_FLASH",
        "step_type": "PERSONALIZED_EXPLANATIONS",
        "input_snapshot": {
            "num_drivers": len(final_proposal["allocation"]),
            "languages": list(set(
                d.get("preferred_language", "en") 
                for d in state.driver_models
            )),
        },
        "output_snapshot": {
            "generated_count": sum(
                1 for e in updated_explanations.values() 
                if e.get("gemini_generated", False)
            ),
            "fallback_count": sum(
                1 for e in updated_explanations.values() 
                if not e.get("gemini_generated", True)
            ),
        },
    }
    
    return {
        "explanations": updated_explanations,
        "decision_logs": state.decision_logs + [log_entry],
    }


def template_fallback(effort: float, avg_effort: float, is_recovery: bool) -> str:
    """
    Fallback template-based explanation when Gemini is unavailable.
    
    Args:
        effort: Today's effort score
        avg_effort: Team average effort
        is_recovery: Whether driver is in recovery mode
        
    Returns:
        Simple explanation string
    """
    if is_recovery:
        return "Recovery route today - lighter load after a busy week. Take it easy!"
    
    delta = effort - avg_effort
    
    if delta < -10:
        return "Light route today! Great opportunity for a smooth day."
    elif delta > 10:
        return "Moderate-heavy route - team balance achieved. You've got this!"
    else:
        return "Perfectly balanced route for you today. Standard workload."
