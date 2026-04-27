# Fair Dispatch System — Project Overview

---

## 1. Project Title and Purpose

**Fair Dispatch System** is a backend API and multi-agent AI platform for **fairness-focused delivery route allocation**. It accepts a set of packages and available drivers each day, clusters packages geographically, scores workloads mathematically, and then runs a multi-agent LangGraph pipeline to ensure that routes are distributed as **equitably** as possible among drivers.

**Key goals:**
- Minimize Gini inequality in per-driver workload
- Support driver negotiation (counter-proposals, recovery days for burned-out drivers)
- Provide transparent, human-readable explanations for every assignment
- Learn continuously from driver feedback using a Multi-Armed Bandit + per-driver XGBoost model
- Support EV drivers with range/charging-aware constraints (Phase 7)
- Operate as a production-ready async service with Celery workers and Redis

---

## 2. Architecture Overview

The system follows a **layered, multi-agent architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                  Frontend (Vanilla JS/HTML)              │
│   demo.html | visualization.html | index.html | app.js  │
└────────────────────────┬────────────────────────────────┘
                         │  REST / SSE
┌────────────────────────▼────────────────────────────────┐
│                 FastAPI Application Layer                 │
│  main.py → API Routers (allocation, drivers, feedback,  │
│            admin, langgraph, runs, agent_events)        │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│          LangGraph Multi-Agent Workflow Engine           │
│                                                         │
│  orchestrator → ml_effort → route_planner_1 →           │
│  fairness_check_1 → [reoptimize?] → route_planner_2 → │
│  fairness_check_2 → select_final → driver_liaison →    │
│  [counter?] → final_resolution → explainability →      │
│  learning → [gemini_explain?] → END                    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                     Service Layer                        │
│  clustering | workload | fairness | explainability |    │
│  ev_utils | recovery_service | history_features |       │
│  history_seed | admin_service | driver_service          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Persistence Layer                      │
│  SQLAlchemy Async ORM → PostgreSQL (prod) / SQLite (dev)│
│  Alembic Migrations                                     │
└─────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│             Async Worker Infrastructure                  │
│  Celery (allocation queue) + Celery Beat (cron)         │
│  Redis (broker + result backend)                        │
└─────────────────────────────────────────────────────────┘
```

**Data store:** SQLite during local/dev runs (auto-created via `init_db()`), PostgreSQL for production (Docker Compose).

---

## 3. Core Components

### 3.1 API Layer (`app/api/`)

| File | Router Prefix | Purpose |
|------|---------------|---------|
| `allocation.py` | `POST /api/v1/allocate` | Synchronous allocation (original multi-agent pipeline) |
| `allocation_langgraph.py` | `POST /api/v1/allocate/langgraph` | LangGraph-orchestrated allocation with SSE events |
| `drivers.py` | `GET /api/v1/drivers/{id}` | Driver details and recent statistics |
| `routes.py` | `GET /api/v1/routes/{id}` | Route details and assignment info |
| `feedback.py` | `POST /api/v1/feedback` | Driver feedback submission |
| `driver_api.py` | `GET /api/v1/driver-api/...` | Driver-facing sub-API (appeals, route swap, stop issues) |
| `admin.py` | `/api/v1/admin/...` | Admin dashboard: runs, config, overrides, stats |
| `admin_learning.py` | `/api/v1/admin/learning/...` | Learning agent status, episode rewards, model retraining |
| `agent_events.py` | `/api/v1/agent-events/stream` | Server-Sent Events (SSE) for real-time agent progress |
| `runs.py` | `/api/v1/runs/{run_id}/...` | Run-scoped endpoints (timeline, assignments, status) |

### 3.2 Multi-Agent Pipeline (`app/services/langgraph_nodes.py`, `langgraph_workflow.py`)

The LangGraph `StateGraph` orchestrates **9 sequential/conditional nodes**:

| Node | Agent Class | Responsibility |
|------|-------------|---------------|
| `orchestrator` | Inline | Validates request, initializes state, seeds DB record |
| `ml_effort` | `MLEffortAgent` | Computes effort matrix (drivers × routes) |
| `route_planner_1` | `RoutePlannerAgent` | OR-Tools/Hungarian optimal assignment (Proposal 1) |
| `fairness_check_1` | `FairnessManagerAgent` | Evaluates Gini / std-dev / max-gap. Decides: ACCEPT or REOPTIMIZE |
| `route_planner_2` | `RoutePlannerAgent` | Re-runs with fairness penalties applied (Proposal 2) |
| `fairness_check_2` | `FairnessManagerAgent` | Second fairness check |
| `select_final` | Inline | Picks the better of Proposal 1 and 2 |
| `driver_liaison` | `DriverLiaisonAgent` | Per-driver ACCEPT / COUNTER / FORCE_ACCEPT negotiation |
| `final_resolution` | `FinalResolutionAgent` | Resolves counter-proposals via legitimate route swaps |
| `explainability` | `ExplainabilityAgent` | Template-based plain-language explanations per driver |
| `learning` | `LearningAgent` | Records episode, stores allocation history for bandit/XGBoost |
| `gemini_explain` *(optional)* | `GeminiExplainNode` | LLM-enhanced explanations via Google Gemini API |

### 3.3 Core Services (`app/services/`)

| Service | Description |
|---------|-------------|
| `clustering.py` | K-Means geographic clustering of packages into routes; nearest-neighbor TSP stop ordering |
| `workload.py` | `workload = a·packages + b·weight + c·difficulty + d·time` formula |
| `fairness.py` | Gini index, per-driver fairness score, standard deviation |
| `ml_effort_agent.py` | Builds `N_drivers × N_routes` effort matrix with physical effort, route complexity, time pressure, capacity penalty, EV overhead, and historical adjustment |
| `route_planner_agent.py` | Solves linear assignment using OR-Tools SCIP (→ GLOP → Hungarian → Greedy fallbacks) |
| `fairness_manager_agent.py` | Evaluates Gini, std-dev, max-gap against configurable thresholds; generates penalty recommendations |
| `driver_liaison_agent.py` | Per-driver comfort-band analysis; outputs ACCEPT, COUNTER, or FORCE_ACCEPT |
| `final_resolution.py` | Resolves COUNTER decisions through valid swaps that don't over-burden other drivers |
| `explainability.py` | `ExplainabilityAgent` with 8 category templates (NEAR_AVG, HEAVY, RECOVERY, LEARNING_OPTIMIZED, etc.) + legacy functions |
| `learning_agent.py` | `FairnessBandit` (Thompson Sampling MAB), `DriverEffortLearner` (XGBoost per-driver), `RewardComputer` |
| `ev_utils.py` | EV feasibility check (battery range vs. route distance) and charging overhead penalty |
| `recovery_service.py` | Complexity debt tracking; enforces recovery (light) days after hard day streaks |
| `history_features.py` | `DriverHistoryFeatures` from 7-day rolling statistics; used in effort, liaison, and explainability |
| `history_seed.py` | Idempotent seeder for mock driver 7-day histories on startup |
| `admin_service.py` | Admin queries: allocation run summaries, override management, fairness config updates |
| `driver_service.py` | Driver CRUD, stats lookup, appeal and swap management |
| `gemini_explain_node.py` | LLM node using `langchain-google-genai` for richer explanation text |

### 3.4 Database Models (`app/models/`)

| Model | Table | Key Fields |
|-------|-------|-----------|
| `Driver` | `drivers` | id, external_id, name, vehicle_type, battery_range_km, is_ev |
| `DriverStatsDaily` | `driver_stats_daily` | date, avg_workload_score, is_hard_day, complexity_debt, is_recovery_day, predicted_effort, actual_effort |
| `DriverFeedback` | `driver_feedback` | fairness_rating (1-5), stress_level (1-10), tiredness_level, hardest_aspect |
| `Route` | `routes` | num_packages, total_weight_kg, num_stops, route_difficulty_score, estimated_time_minutes, total_distance_km |
| `Package` | `packages` | weight_kg, fragility_level, latitude, longitude, priority |
| `Assignment` | `assignments` | driver_id, route_id, workload_score, fairness_score, effort |
| `AllocationRun` | `allocation_runs` | status (PENDING/SUCCESS/FAILED), global_gini_index, global_std_dev, global_max_gap |
| `DecisionLog` | `decision_logs` | agent_name, step_type, input_snapshot (JSON), output_snapshot (JSON) |
| `FairnessConfig` | `fairness_configs` | gini_threshold, stddev_threshold, max_gap_threshold, recovery_mode_enabled |
| `DriverEffortModel` | `driver_effort_models` | model_path (JSON), model_version, mse_history, r2_score |
| `LearningEpisode` | `learning_episodes` | config_hash, arm_idx, episode_reward, alpha_prior, beta_prior |
| `Appeal` | `appeals` | assignment_id, reason, status |
| `RouteSwap` | `route_swaps` | from_driver_id, to_driver_id, status |
| `ManualOverride` | `manual_overrides` | by admin, before/after snapshot |

### 3.5 Pydantic Schemas (`app/schemas/`)

| Schema File | Key Schemas |
|-------------|-------------|
| `allocation.py` | `AllocationRequest`, `AllocationResponse`, `AssignmentResult`, `RouteSummary`, `GlobalFairness` |
| `agent_schemas.py` | `EffortMatrixResult`, `RoutePlanResult`, `FairnessCheckResult`, `NegotiationResult`, `DriverLiaisonDecision` |
| `allocation_state.py` | `AllocationState` — the shared LangGraph typed state dict threading through all nodes |
| `driver_api.py` | Driver appeal, stop issue, route swap, delivery log schemas |
| `admin.py` | Admin report schemas, fairness config update payloads |
| `learning_schemas.py` | Learning status, episode summary, bandit statistics |

### 3.6 Frontend (`frontend/`)

| File | Purpose |
|------|---------|
| `index.html` | Main live supervision dashboard (driver map, agent status panel) |
| `app.js` | Dashboard JavaScript logic: SSE listener, chart updates, map rendering (Leaflet) |
| `demo.html` | JSON-based API demo page (pre-filled AllocationRequest, live JSON output) |
| `visualization.html` | Real-time agent pipeline visualization with timeline and Mermaid workflow |
| `api.js` | Frontend API client abstraction layer |
| `styles.css` | Global CSS |

### 3.7 Celery Infrastructure (`app/core/`, `cron/`)

| Component | Purpose |
|-----------|---------|
| `app/core/celery_app.py` | Celery application factory with Redis broker |
| `app/tasks.py` | Celery task definitions (async allocation, model retraining) |
| `cron/daily_learning.py` | Daily cron: processes reward for all episodes, retrains per-driver XGBoost models |

---

## 4. Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Core language |
| **FastAPI** | 0.109.0 | REST API framework, auto OpenAPI docs |
| **Uvicorn** | 0.27.0 | ASGI server |
| **Pydantic v2** | 2.5.3 | Data validation and serialization |
| **SQLAlchemy (async)** | 2.0.25 | ORM with async session support |
| **Alembic** | 1.13.1 | Database schema migrations |
| **asyncpg** | 0.29.0 | Async PostgreSQL driver |
| **aiosqlite** | ≥0.19.0 | Async SQLite driver (dev/test) |
| **scikit-learn** | 1.4.0 | K-Means clustering |
| **SciPy** | 1.12.0 | Hungarian algorithm (`linear_sum_assignment`) |
| **NumPy** | 1.26.3 | Matrix operations, distance calculations |
| **OR-Tools** | 9.8.3296 | SCIP/GLOP ILP solver for optimal assignment |
| **XGBoost** | 2.0.3 | Per-driver effort prediction (regression) |
| **Pandas** | 2.1.4 | Feature engineering for XGBoost |
| **LangGraph** | ≥0.2.0 | Multi-agent workflow state machine |
| **LangChain Core** | ≥0.2.0 | LLM chain utilities |
| **langchain-google-genai** | ≥1.0.0 | Gemini API integration |
| **LangSmith** | ≥0.1.0 | Workflow tracing and observability |
| **Celery** | ≥5.3.0 | Distributed async task queue |
| **Redis** | ≥5.0.0 | Message broker and result backend |
| **Flower** | ≥2.0.0 | Celery worker monitoring dashboard |
| **PostgreSQL** | 15 (Docker) | Production relational database |
| **Docker / Docker Compose** | — | Containerized deployment |
| **Pytest + pytest-asyncio** | — | Unit and integration testing |

---

## 5. Inter-Module Communication

### 5.1 HTTP / REST
Clients (frontend, curl, external apps) communicate with FastAPI via standard HTTP REST endpoints. All responses are JSON (Pydantic serialized).

### 5.2 Server-Sent Events (SSE)
The `GET /api/v1/agent-events/stream` endpoint pushes real-time agent progress events to the frontend dashboard as each LangGraph node completes. These events carry node name, status, timing, and snapshot data.

### 5.3 LangGraph State Threading
Within the pipeline, agents communicate through **`AllocationState`** — a Pydantic typed dictionary. Each node reads from and writes to shared state fields (e.g., `effort_result`, `route_plan_1`, `fairness_result_1`, `negotiation_result`, `final_assignments`, `explanations`). LangGraph guarantees sequential node execution with conditional routing.

### 5.4 SQLAlchemy Async ORM
All service functions receive an `AsyncSession` (dependency-injected by FastAPI) and communicate with the database through async ORM queries. The session lifecycle is per-request.

### 5.5 Celery Task Queue (Redis)
The `allocation` Celery queue handles offloaded heavy computation. The FastAPI web layer submits tasks; Celery workers pick them up from Redis, execute the LangGraph pipeline, and write results to PostgreSQL.

### 5.6 Direct Function Calls (Intra-service)
Agents are instantiated and called directly within `langgraph_nodes.py` — no network boundary between agent classes. All inter-agent data is passed as Python objects within the same process.

---

## 6. Data Flow and Lifecycle

```
CLIENT REQUEST (POST /api/v1/allocate)
  │
  ├─ 1. Validate AllocationRequest (Pydantic)
  ├─ 2. Persist AllocationRun (status=PENDING)
  ├─ 3. Upsert Driver records
  ├─ 4. cluster_packages() → K-Means on lat/lng → ClusterResult[]
  ├─ 5. Persist Route records
  ├─ 6. Load driver history (DriverHistoryFeatures per driver)
  ├─ 7. Load/select FairnessConfig (bandit or default)
  ├─ 8. Execute LangGraph Workflow:
  │     ├─ ml_effort.compute_effort_matrix()
  │     │     → physical effort + route complexity + time pressure +
  │     │       capacity penalty + EV overhead + history adjustment
  │     ├─ route_planner_agent.plan()  [Proposal 1]
  │     │     → OR-Tools SCIP solver → assignments
  │     ├─ fairness_manager_agent.check()
  │     │     → Gini, std_dev, max_gap vs thresholds
  │     │     → ACCEPT or REOPTIMIZE
  │     ├─ [if REOPTIMIZE] route_planner_agent.plan()  [Proposal 2]
  │     │     → with fairness penalty multipliers
  │     ├─ select_final: pick better proposal
  │     ├─ driver_liaison_agent.run_for_all_drivers()
  │     │     → per-driver ACCEPT / COUNTER / FORCE_ACCEPT
  │     ├─ [if COUNTER] final_resolution_agent → route swaps
  │     ├─ explainability_agent.build_explanation_for_driver()
  │     │     → category classification + template text
  │     └─ learning_agent.create_episode()
  │           → records experiment for bandit update
  ├─ 9. Persist Assignments + DecisionLogs
  ├─ 10. Update AllocationRun (status=SUCCESS, fairness metrics)
  └─ 11. Return AllocationResponse (assignments + explanations + global_fairness)

DRIVER FEEDBACK (POST /api/v1/feedback)
  └─ Stored in driver_feedback table
  └─ Nightly cron: RewardComputer computes episode_reward
     → FairnessBandit.update(config_hash, reward)
     → DriverEffortLearner.update_model(driver_id) [XGBoost retrain]
```

---

## 7. Deployment Details

### 7.1 Local Development
```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```
SQLite (`fair_dispatch.db`) is auto-created on startup via `init_db()`. No migration needed for dev.

### 7.2 Docker Compose (Production-like)
`docker-compose.yml` defines 5 services using a single shared Dockerfile (`python:3.12-slim`):

| Service | Command | Port |
|---------|---------|------|
| `postgres` | PostgreSQL 15 | 5432 |
| `redis` | Redis 7 | 6379 |
| `web` | `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` | 8000 |
| `worker` | `celery -A app.core.celery_app:celery_app worker --concurrency=4 -Q allocation` | — |
| `beat` | `celery beat` (periodic task scheduler) | — |
| `flower` | Celery monitoring dashboard | 5555 |

**Health checks** on postgres and redis ensure the web/worker services wait for dependencies. Volume mounts persist `pgdata`, `redisdata`, and a `model_storage` volume for XGBoost `.json` model artifacts.

### 7.3 CI / Testing (`Makefile`, `pytest.ini`)
```bash
make test          # all unit + E2E + integration
make test-parallel # pytest-xdist parallel run
make test-cov      # with coverage report
make ci            # full CI verification
```
Tests use `aiosqlite` in-memory databases and `httpx` async test client. A separate `tests/docker-compose.test.yml` can spin up a PostgreSQL test instance.

### 7.4 GitHub Actions
`.github/` directory is present (content not expanded), indicating CI/CD is configured.

---

## 8. Key Algorithms and Models

### 8.1 Geographic Clustering (K-Means)
- Input: package lat/lng coordinates
- `num_routes = min(num_drivers, num_packages)`
- `sklearn.cluster.KMeans(n_clusters=num_routes, random_state=42, n_init=10)`
- Stop ordering: nearest-neighbor heuristic (TSP approximation using haversine distance)

### 8.2 Workload Scoring Formula
```
workload_score = a·num_packages + b·total_weight_kg + c·route_difficulty_score + d·estimated_time_minutes
```
Default weights: `a=1.0, b=0.5, c=10.0, d=0.2` (configurable via `.env`)

### 8.3 Effort Matrix (MLEffortAgent)
```
effort = α·packages + β·weight + γ·difficulty + δ·time + ε·capacity_mismatch + EV_overhead + history_adjustment
```
Where `history_adjustment` is derived from 7-day rolling `recent_avg_effort`, `recent_hard_days`, `avg_stress_level`, and `avg_fairness_rating`.

### 8.4 Linear Assignment Optimization (RoutePlannerAgent)
- **Primary**: OR-Tools SCIP (ILP) — minimizes `Σ cost[i][j] × x[i][j]`
- **Fallback 1**: OR-Tools GLOP
- **Fallback 2**: SciPy `linear_sum_assignment` (Hungarian algorithm)
- **Fallback 3**: Greedy assignment (sort all pairs by cost, pick greedily)

Infeasible pairs (EV drivers with insufficient range) are set to cost `99999`.

### 8.5 Gini Index (Fairness)
```
Gini = Σ|xi − xj| / (2 · n² · mean(x))
```
Ranges from 0 (perfect equality) to 1 (maximum inequality). Threshold: ≤ 0.33 by default.

### 8.6 Per-Driver Comfort Band (Driver Liaison)
```
comfort_upper = recent_avg_effort + max(global_std, recent_std)
# Tightened if: recent_hard_days ≥ 3 (−0.3·std), fatigue ≥ 4 (−0.2·std), avg_stress ≥ 7 (−0.2·std)
```

### 8.7 Multi-Armed Bandit — FairnessConfig Selection (Phase 8)
- Algorithm: **Thompson Sampling** with Beta(α, β) posteriors
- Arms: Cartesian product of `GINI_OPTIONS × STDDEV_OPTIONS × RECOVERY_OPTIONS × EV_PENALTY_OPTIONS` = 81 arms
- Reward formula per episode:
  ```
  reward = 0.4·avg_fairness_rating + 0.3·(1 − avg_stress/10) + 0.2·completion_rate + 0.1·(1 − avg_tiredness/5)
  ```
- Priors are loaded from last 30 days of `LearningEpisode` records

### 8.8 Per-Driver XGBoost Effort Model (Phase 8)
- Features: `num_packages, total_weight_kg, num_stops, route_difficulty_score, estimated_time_minutes, experience_days, recent_avg_workload, recent_hard_days`
- Architecture: `XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1)`
- Minimum training samples: 10 per driver; max: 100 (most recent)
- Saved as safe **JSON format** (no pickle) with SHA-256 integrity checksum
- Retrained nightly via Celery Beat → `cron/daily_learning.py`

---

## 9. Scalability and Security Considerations

### 9.1 Scalability
- **Celery workers** (concurrency=4) decouple heavy allocation computation from the web tier; horizontally scalable by adding more worker containers
- **Async I/O** throughout FastAPI + SQLAlchemy asyncpg eliminates thread-blocking
- **K-Means** is O(n·k·iterations), efficient for hundreds of packages
- **OR-Tools SCIP** handles large assignment matrices (tested with 20 drivers × 20 routes)
- **Redis** serves as a fast message broker and result store
- **PostgreSQL** with proper indexing on `driver_id`, `date`, `allocation_run_id` columns

### 9.2 Security
- **CORS**: Configured with `allow_origins=["*"]` — must be restricted for production deployments
- **No Pickle**: XGBoost models are stored as JSON to prevent deserialization attacks
- **SHA-256 checksums** verify model file integrity before loading
- **Environment-based secrets**: All credentials (`DATABASE_URL`, `GOOGLE_API_KEY`, `LANGCHAIN_API_KEY`) are loaded from `.env` and never hardcoded
- **Pydantic validation**: All inbound request data is strictly validated before processing
- **DecisionLog audit trail**: Every agent step's input/output is persisted for traceability

### 9.3 Error Handling
- `AllocationRun.status` transitions to `FAILED` on any exception, with error message stored
- OR-Tools/Hungarian/Greedy fallback chain ensures assignment always completes
- EV infeasibility gracefully marks driver-route pairs with a high-cost sentinel (99999)
- `FORCE_ACCEPT` in the liaison agent handles cases where no fair alternative exists

---

## 10. File Structure Summary

```
fair-dispatch-system/
├── app/
│   ├── main.py                      # FastAPI app, middleware, router registration, lifespan
│   ├── config.py                    # Pydantic Settings (env vars, workload weights, feature flags)
│   ├── database.py                  # Async SQLAlchemy engine, Base, session, init_db()
│   ├── tasks.py                     # Celery task definitions
│   ├── api/
│   │   ├── __init__.py              # Router exports
│   │   ├── allocation.py            # POST /allocate (sync pipeline)
│   │   ├── allocation_langgraph.py  # POST /allocate/langgraph (LangGraph + SSE)
│   │   ├── drivers.py               # GET /drivers/{id}
│   │   ├── routes.py                # GET /routes/{id}
│   │   ├── feedback.py              # POST /feedback
│   │   ├── driver_api.py            # Driver-facing sub-API
│   │   ├── admin.py                 # Admin management endpoints
│   │   ├── admin_learning.py        # Learning system admin endpoints
│   │   ├── agent_events.py          # SSE stream for real-time agent updates
│   │   └── runs.py                  # Run-scoped query endpoints
│   ├── models/
│   │   ├── driver.py                # Driver, DriverStatsDaily, DriverFeedback
│   │   ├── route.py                 # Route
│   │   ├── package.py               # Package
│   │   ├── assignment.py            # Assignment
│   │   ├── allocation_run.py        # AllocationRun
│   │   ├── decision_log.py          # DecisionLog (agent audit)
│   │   ├── driver_effort_model.py   # XGBoost model registry
│   │   ├── learning_episode.py      # Bandit episode records
│   │   ├── fairness_config.py       # Active FairnessConfig
│   │   ├── appeal.py                # Driver appeal
│   │   ├── manual_override.py       # Admin override
│   │   ├── route_swap.py            # Route swap records
│   │   ├── delivery_log.py          # Delivery event logs
│   │   └── stop_issue.py            # Stop-level issue reports
│   ├── schemas/
│   │   ├── allocation.py            # AllocationRequest/Response
│   │   ├── allocation_state.py      # LangGraph AllocationState
│   │   ├── agent_schemas.py         # Effort, plan, fairness, liaison schemas
│   │   ├── admin.py                 # Admin request/response schemas
│   │   ├── driver_api.py            # Driver sub-API schemas
│   │   ├── explainability.py        # Explanation input/output
│   │   └── learning_schemas.py      # Learning system schemas
│   ├── services/
│   │   ├── langgraph_workflow.py    # Graph definition and `invoke_allocation_workflow()`
│   │   ├── langgraph_nodes.py       # All LangGraph node implementations
│   │   ├── ml_effort_agent.py       # MLEffortAgent
│   │   ├── route_planner_agent.py   # RoutePlannerAgent (OR-Tools/Hungarian)
│   │   ├── fairness_manager_agent.py # FairnessManagerAgent (Gini)
│   │   ├── driver_liaison_agent.py  # DriverLiaisonAgent (comfort-band negotiation)
│   │   ├── final_resolution.py      # FinalResolutionAgent (swap resolution)
│   │   ├── explainability.py        # ExplainabilityAgent (8 category templates)
│   │   ├── learning_agent.py        # LearningAgent, FairnessBandit, DriverEffortLearner
│   │   ├── clustering.py            # K-Means + nearest-neighbor stop ordering
│   │   ├── workload.py              # Workload score formula
│   │   ├── fairness.py              # Gini index, fairness_score
│   │   ├── ev_utils.py              # EV range feasibility + charging overhead
│   │   ├── recovery_service.py      # Complexity debt + recovery day scheduling
│   │   ├── history_features.py      # 7-day rolling driver history features
│   │   ├── history_seed.py          # Mock history seeder on startup
│   │   ├── admin_service.py         # Admin business logic
│   │   ├── driver_service.py        # Driver profile/stats service
│   │   └── gemini_explain_node.py   # Optional LLM explanation node
│   ├── core/
│   │   └── celery_app.py            # Celery factory
│   ├── solver/                      # (Reserved for future custom solver logic)
│   ├── trainer/                     # (Reserved for ML training utilities)
│   └── utils/                       # General utilities
├── frontend/
│   ├── index.html                   # Live supervision dashboard
│   ├── app.js                       # Dashboard JS (SSE, Leaflet map, charts)
│   ├── demo.html                    # API demo page
│   ├── visualization.html           # Agent pipeline visualization
│   ├── api.js                       # Frontend API client
│   └── styles.css                   # Global styles
├── alembic/
│   ├── env.py                       # Migration environment
│   └── versions/                    # Migration scripts
├── tests/
│   ├── conftest.py                  # Test fixtures (DB, drivers, routes)
│   └── [unit, integration, e2e]     # Test modules
├── cron/
│   └── daily_learning.py            # Nightly reward processing + model retraining
├── docs/
│   ├── ARCHITECTURE.md              # System architecture detail
│   ├── ARCHITECTURE_UPGRADE_CHANGELOG.md
│   ├── CHANGES.md                   # Version changelog
│   └── integration.md               # API integration guide
├── Dockerfile                       # python:3.12-slim, multi-purpose image
├── docker-compose.yml               # 5-service stack: postgres, redis, web, worker, beat, flower
├── Makefile                         # Test and CI commands
├── requirements.txt                 # All Python dependencies
├── pytest.ini                       # Pytest configuration
├── alembic.ini                      # Alembic migration config
├── .env.example                     # Environment variable template
└── README.md                        # Project quickstart documentation
```

---

## 11. Potential Improvements

The following improvements are suggested **without modifying any existing code**:

### Documentation
1. **API contract documentation**: Add OpenAPI tags and descriptions to all router endpoints; the auto-generated `/docs` page would become richer
2. **`AGENTS.md`**: A dedicated document explaining each agent's decision logic, inputs, outputs, and thresholds would help new contributors
3. **Architecture diagram**: A visual Mermaid or draw.io diagram embedded in `docs/ARCHITECTURE.md` showing the full LangGraph DAG and service boundaries
4. **Threshold rationale**: Document why default values (Gini ≤ 0.33, std_dev ≤ 25.0, etc.) were chosen and how to tune them

### Architecture
5. **CORS hardening**: Replace `allow_origins=["*"]` with an environment-controlled allowlist for production
6. **Authentication layer**: The API currently has no auth; adding JWT-based auth for driver and admin endpoints would be critical before production deployment
7. **Async Celery result polling**: The LangGraph endpoint currently runs synchronously inside the request; wrapping it in a Celery task and returning a run ID for polling would improve responsiveness for large inputs
8. **Model drift detection**: The XGBoost models are retrained daily, but no drift alert exists; adding an MSE threshold alarm in the nightly cron would prevent silent degradation
9. **Database connection pooling**: For high-concurrency prod scenarios, configuring `pool_size` and `max_overflow` in the SQLAlchemy engine would improve stability
10. **SQLite → PostgreSQL migration guide**: A clearly documented step to switch from the default SQLite dev mode to PostgreSQL would help onboarding

---

## Summary

The **Fair Dispatch System** is a production-grade, multi-agent AI platform that combines **classical optimization** (K-Means clustering, OR-Tools ILP assignment) with a **LangGraph-orchestrated agent pipeline** to fairly allocate delivery routes to drivers. The system goes beyond static algorithms by incorporating per-driver negotiation (comfort-band analysis), an EV-aware routing module, an explainability engine that generates plain-language assignment justifications, and a continuous learning loop using **Thompson Sampling** for fairness configuration selection and per-driver **XGBoost regression** for personalized effort prediction. The entire stack is containerized with Docker Compose, with Celery workers handling async allocation and Celery Beat running nightly model retraining — making it both ethically principled and operationally scalable.
