# Fair Dispatch System - Architecture & Source Files

## Overview

The Fair Dispatch System is a multi-agent AI system for fair route allocation to delivery drivers. It uses LangGraph for workflow orchestration and implements fairness-aware optimization using OR-Tools.

---

## Agents & Core Components

### 1. Central Orchestrator
**Purpose:** Initializes the allocation workflow, validates inputs, and coordinates the agent execution sequence.

| Component | Source File |
|-----------|-------------|
| Orchestrator Node | `app/services/langgraph_nodes.py` → `orchestrator_node()` |
| Workflow Graph | `app/services/langgraph_workflow.py` → `create_allocation_graph()` |

**Key Features:**
- Workflow initialization and configuration
- Input validation (drivers, routes, packages)
- Agent sequence coordination
- Fleet composition analysis (ICE/EV/BICYCLE)

---

### 2. Workload Optimization Engine (ML Effort Agent)
**Purpose:** Computes effort matrix for all driver-route pairs using weighted formulas and historical adjustments.

| Component | Source File |
|-----------|-------------|
| ML Effort Agent Class | `app/services/ml_effort_agent.py` → `MLEffortAgent` |
| LangGraph Node | `app/services/langgraph_nodes.py` → `ml_effort_node()` |
| Effort Weights Schema | `app/schemas/agent_schemas.py` → `EffortWeights` |

**Key Features:**
- Effort calculation: `α*packages + β*weight + γ*difficulty + δ*time`
- EV range feasibility checking
- Historical effort adjustment based on driver fatigue
- Capacity utilization penalties

---

### 3. Route Planner Agent
**Purpose:** Generates optimal driver-route assignments using OR-Tools constraint programming.

| Component | Source File |
|-----------|-------------|
| Route Planner Agent Class | `app/services/route_planner_agent.py` → `RoutePlannerAgent` |
| LangGraph Node (Proposal 1) | `app/services/langgraph_nodes.py` → `route_planner_node()` |
| LangGraph Node (Proposal 2) | `app/services/langgraph_nodes.py` → `route_planner_reoptimize_node()` |

**Key Features:**
- OR-Tools CP-SAT solver integration
- 1:1 driver-route matching
- Fairness penalty injection for re-optimization
- Infeasible pair handling (EV range constraints)

---

### 4. Fairness Manager Agent
**Purpose:** Evaluates fairness metrics and decides if re-optimization is needed.

| Component | Source File |
|-----------|-------------|
| Fairness Manager Agent Class | `app/services/fairness_manager_agent.py` → `FairnessManagerAgent` |
| LangGraph Node | `app/services/langgraph_nodes.py` → `fairness_check_node()` |
| Fairness Calculation | `app/services/fairness.py` → `calculate_fairness_score()` |
| Fairness Thresholds Schema | `app/schemas/agent_schemas.py` → `FairnessThresholds` |

**Key Features:**
- Gini index calculation (target: < 0.15)
- Standard deviation analysis
- Max gap detection (max - min effort)
- APPROVE/REOPTIMIZE decision logic

---

### 5. Driver Liaison Agent
**Purpose:** Reviews proposed assignments from the driver's perspective and negotiates alternatives.

| Component | Source File |
|-----------|-------------|
| Driver Liaison Agent Class | `app/services/driver_liaison_agent.py` → `DriverLiaisonAgent` |
| LangGraph Node | `app/services/langgraph_nodes.py` → `driver_liaison_node()` |
| Driver Context Schema | `app/schemas/agent_schemas.py` → `DriverContext` |
| Liaison Decision Schema | `app/schemas/agent_schemas.py` → `DriverLiaisonDecision` |

**Key Features:**
- Comfort band calculation (avg ± std)
- ACCEPT/COUNTER/FORCE_ACCEPT decisions
- Historical stress-based adjustments
- Alternative route suggestions

---

### 6. Final Resolution Agent
**Purpose:** Resolves COUNTER decisions through driver-driver route swaps.

| Component | Source File |
|-----------|-------------|
| Final Resolution Agent Class | `app/services/final_resolution.py` → `FinalResolutionAgent` |
| LangGraph Node | `app/services/langgraph_nodes.py` → `final_resolution_node()` |

**Key Features:**
- Swap pair identification
- Effort improvement validation
- Fairness preservation during swaps
- Cascade swap prevention

---

### 7. Explainability Agent
**Purpose:** Generates human-readable explanations for route assignments.

| Component | Source File |
|-----------|-------------|
| Explainability Agent Class | `app/services/explainability.py` → `ExplainabilityAgent` |
| LangGraph Node | `app/services/langgraph_nodes.py` → `explainability_node()` |
| Explanation Input Schema | `app/schemas/explainability.py` → `DriverExplanationInput` |
| Explanation Output Schema | `app/schemas/explainability.py` → `DriverExplanationOutput` |

**Key Features:**
- Driver-facing explanations (friendly tone)
- Admin-facing explanations (technical details)
- Category classification (EASY_DAY, AVERAGE, HARD_DAY, RECOVERY)
- Effort breakdown visualization

---

### 8. Learning Agent
**Purpose:** Analyzes historical data and records learning episodes for future optimization.

| Component | Source File |
|-----------|-------------|
| LangGraph Node | `app/services/langgraph_nodes.py` → `learning_agent_node()` |
| Learning Agent Class | `app/services/learning_agent.py` → `LearningAgent` |
| Fairness Bandit | `app/services/learning_agent.py` → `FairnessBandit` |
| Driver Effort Learner | `app/services/learning_agent.py` → `DriverEffortLearner` |
| Reward Computer | `app/services/learning_agent.py` → `RewardComputer` |

**Key Features:**
- Historical feature analysis (stress, fatigue, hard days)
- Online/cold-start learning mode detection
- Thompson Sampling for config selection
- Episode reward computation from feedback

---

## Historical Data Layer

### History Seeding Service
**Purpose:** Seeds 50 mock drivers with 7 days of historical assignments and feedback.

| Component | Source File |
|-----------|-------------|
| Seed Function | `app/services/history_seed.py` → `seed_mock_history()` |

**Key Features:**
- 50 mock drivers with Indian names
- 7 days of past assignments per driver
- Realistic stress/fairness/tiredness ratings
- Idempotent (safe to run multiple times)

---

### History Features Service
**Purpose:** Computes driver historical features from past assignments and feedback.

| Component | Source File |
|-----------|-------------|
| Features Computation | `app/services/history_features.py` → `compute_history_features_for_drivers()` |
| History Features Model | `app/services/history_features.py` → `DriverHistoryFeatures` |
| History Config | `app/services/history_features.py` → `HistoryConfig` |
| Effort Adjustment | `app/services/history_features.py` → `compute_history_effort_adjustment()` |

**Key Features:**
- Recent average/std effort calculation
- Hard day detection (effort > 1.2× average)
- Fatigue score computation
- Stress and fairness rating aggregation

---

## API Endpoints

### Allocation API
| Endpoint | Source File |
|----------|-------------|
| `POST /api/v1/allocate/langgraph` | `app/api/allocation_langgraph.py` |
| `POST /api/v1/allocate` | `app/api/allocation.py` |

### Admin API
| Endpoint | Source File |
|----------|-------------|
| `GET /api/v1/admin/allocation_runs` | `app/api/admin.py` |
| `GET /api/v1/admin/agent_timeline` | `app/api/admin.py` |
| `GET /api/v1/admin/decision_logs` | `app/api/admin.py` |
| `GET /api/v1/admin/fairness_summary` | `app/api/admin.py` |

### Driver API
| Endpoint | Source File |
|----------|-------------|
| `GET /api/v1/driver/{id}/assignments` | `app/api/driver_api.py` |
| `POST /api/v1/driver/{id}/feedback` | `app/api/driver_api.py` |
| `GET /api/v1/driver/{id}/history` | `app/api/driver_api.py` |

### Feedback API
| Endpoint | Source File |
|----------|-------------|
| `POST /api/v1/feedback` | `app/api/feedback.py` |
| `GET /api/v1/feedback/{driver_id}` | `app/api/feedback.py` |

---

## Database Models

| Model | Source File | Description |
|-------|-------------|-------------|
| `Driver` | `app/models/driver.py` | Driver profile with vehicle info |
| `DriverFeedback` | `app/models/driver.py` | Post-shift feedback records |
| `Route` | `app/models/route.py` | Route with packages and difficulty |
| `Assignment` | `app/models/assignment.py` | Driver-route assignment |
| `AllocationRun` | `app/models/allocation_run.py` | Allocation execution record |
| `DecisionLog` | `app/models/decision_log.py` | Agent decision audit trail |
| `LearningEpisode` | `app/models/learning_episode.py` | Learning agent episodes |
| `DriverStatsDaily` | `app/models/driver.py` | Daily driver statistics |

---

## Schemas (Pydantic Models)

| Schema | Source File | Description |
|--------|-------------|-------------|
| `AllocationState` | `app/schemas/allocation_state.py` | LangGraph workflow state |
| `AllocationRequest` | `app/schemas/allocation.py` | API request payload |
| `AllocationResponse` | `app/schemas/allocation.py` | API response payload |
| `DriverHistoryFeatures` | `app/services/history_features.py` | Historical features model |
| `EffortWeights` | `app/schemas/agent_schemas.py` | Effort calculation weights |
| `FairnessThresholds` | `app/schemas/agent_schemas.py` | Fairness evaluation thresholds |

---

## Frontend

| Component | Source File | Description |
|-----------|-------------|-------------|
| Agent Visualization | `frontend/visualization.html` | Real-time agent workflow UI |
| API Client | `frontend/api.js` | Backend API integration |
| Main App | `frontend/app.js` | Dashboard application |
| Demo Page | `frontend/demo.html` | Quick demo interface |

---

## Testing

| Test File | Coverage |
|-----------|----------|
| `tests/test_history_layer.py` | History seeding, features, adjustments |
| `tests/test_ml_effort_agent.py` | Effort matrix computation |
| `tests/test_driver_liaison_agent.py` | Liaison decisions |
| `tests/test_fairness_manager_agent.py` | Fairness evaluation |
| `tests/test_route_planner_agent.py` | OR-Tools optimization |
| `tests/test_explainability_agent.py` | Explanation generation |
| `tests/test_learning_agent.py` | Learning & bandit algorithms |
| `tests/test_langgraph_workflow.py` | End-to-end workflow |
| `tests/test_full_workflow.py` | Integration tests |

---

## Configuration

| File | Purpose |
|------|---------|
| `app/config.py` | Application settings |
| `app/database.py` | Database connection |
| `alembic.ini` | Database migrations config |
| `requirements.txt` | Python dependencies |
| `pytest.ini` | Test configuration |

---

## Workflow Sequence

```
┌─────────────────────┐
│ Central Orchestrator│  ← Initializes workflow
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ ML Effort Agent     │  ← Computes effort matrix
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Route Planner       │  ← Proposal 1 (pure effort)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Fairness Manager    │  ← Check fairness
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
 APPROVE    REOPTIMIZE
    │           │
    │     ┌─────▼─────┐
    │     │Route Plan 2│  ← Proposal 2 (with penalties)
    │     └─────┬─────┘
    │           ▼
    │     ┌───────────┐
    │     │ Fairness 2│
    │     └─────┬─────┘
    └─────┬─────┘
          ▼
┌─────────────────────┐
│ Select Final        │  ← Pick best proposal
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Driver Liaison      │  ← Review per driver
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
  SKIP      RESOLVE
    │           │
    │     ┌─────▼─────┐
    │     │ Final Res │  ← Swap resolution
    │     └─────┬─────┘
    └─────┬─────┘
          ▼
┌─────────────────────┐
│ Explainability      │  ← Generate explanations
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Learning Agent      │  ← Record episode
└─────────┬───────────┘
          ▼
        END
```

---

## Key Formulas

### Effort Calculation
```
effort = α × packages + β × weight_kg + γ × difficulty + δ × time_minutes
```
Default weights: α=1.0, β=0.5, γ=10.0, δ=0.2

### History Adjustment
```
adjustment = w_hard_days × recent_hard_days 
           + w_stress × (avg_stress - 6.0)  [if stress > 6]
           - w_fairness × (avg_fairness - 3.0)  [if fairness > 3]
```
Default weights: w_hard_days=2.0, w_stress=3.0, w_fairness=1.0

### Gini Index
```
gini = Σ|effort_i - effort_j| / (2 × n × Σ effort)
```
Target: gini < 0.15

### Comfort Band
```
comfort_upper = recent_avg_effort + max(global_std, recent_std)
```
Tightened for high-stress or fatigued drivers.

---

*Generated: February 5, 2026*
