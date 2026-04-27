# Architecture Upgrade — Complete Changelog

**Date:** 2026-02-20  
**Branch:** `feature/async-cvrp-refactor`  
**Scope:** System-wide migration from synchronous monolith to async worker-driven architecture

---

## Table of Contents

1. [Overview](#1-overview)
2. [Primary Goals Achieved](#2-primary-goals-achieved)
3. [New Files Created](#3-new-files-created)
4. [Modified Files](#4-modified-files)
5. [API Contract Changes](#5-api-contract-changes)
6. [Database Schema Changes](#6-database-schema-changes)
7. [Security Fixes](#7-security-fixes)
8. [Dependency Changes](#8-dependency-changes)
9. [Docker / Infrastructure](#9-docker--infrastructure)
10. [Test Suite](#10-test-suite)
11. [Migration Guide](#11-migration-guide)

---

## 1. Overview

This upgrade transforms the Fair Dispatch System from a **synchronous FastAPI monolith** (where all CPU-heavy computation runs inline inside HTTP request handlers) into an **async, worker-driven architecture** with:

- **Celery workers** processing allocation jobs in background processes
- **OR-Tools CVRP solver** replacing the rigid 1:1 Linear Assignment Problem
- **Redis** as both job broker and large artifact cache
- **Safe XGBoost model storage** replacing the insecure `pickle` serialization

The system now responds immediately with HTTP 202 + `job_id` instead of blocking for minutes during allocation.

---

## 2. Primary Goals Achieved

| # | Goal | Solution | Status |
|---|------|----------|--------|
| 1 | Remove CPU-bound compute from request handlers | Celery worker queue with `run_allocation_job` task | ✅ Done |
| 2 | Replace LAP with CVRP solver | OR-Tools `RoutingModel` with capacity, EV range, fairness constraints | ✅ Done |
| 3 | Async ML training (no blocking during allocation) | Celery Beat scheduled task (`retrain_xgboost_models`) runs daily | ✅ Done |
| 4 | Lightweight `AllocationState` (no huge in-memory matrices) | Redis artifact store; state carries only string key references | ✅ Done |

---

## 3. New Files Created

### Infrastructure (4 files)

#### `app/core/celery_app.py`
- **Purpose:** Celery application factory
- **Key config:** Redis broker, JSON-only serialization (`task_serializer="json"`), `task_acks_late=True`, `worker_prefetch_multiplier=1`
- **Beat schedule:** `retrain-xgboost-daily` task runs every 24 hours
- **Singleton:** Exports `celery_app` instance used by workers and decorators

#### `Dockerfile`
- **Base:** `python:3.12-slim`
- **System deps:** `gcc`, `libpq-dev` (for psycopg2 and scipy)
- **Dual-purpose:** Same image used by both `web` and `worker` services (CMD overridden in docker-compose)
- **Model dir:** Creates `models/` directory for XGBoost JSON artifacts

#### `docker-compose.yml`
- **6 services:**
  - `postgres` — PostgreSQL 15 with health check
  - `redis` — Redis 7 Alpine with health check
  - `web` — FastAPI/Uvicorn (port 8000)
  - `worker` — Celery worker (4 concurrency, `allocation` queue)
  - `beat` — Celery Beat scheduler (periodic tasks)
  - `flower` — Celery monitoring dashboard (port 5555)
- **Shared volumes:** `model_storage` for XGBoost artifacts shared between worker and beat
- **Health checks:** Postgres and Redis have readiness probes; web/worker depend on them

#### `.env.example` (additions only)
- Added: `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `MODEL_STORAGE_DIR`, `SYNC_ALLOCATION_ENABLED`

---

### Worker System (3 files)

#### `app/tasks.py`
- **Task 1:** `run_allocation_job(allocation_run_id: str)` — main async allocation task
  - Queue: `allocation`
  - Max retries: 2 with 30s delay
  - On failure: marks `AllocationRun.status = FAILED` in DB before retry
  - Lazy imports to avoid circular dependencies
- **Task 2:** `retrain_xgboost_models()` — periodic XGBoost retraining
  - Queue: `allocation`
  - Max retries: 1 with 60s delay
  - Scheduled by Celery Beat (daily)

#### `app/worker/__init__.py`
- Worker package init file

#### `app/worker/langgraph_runner.py`
- **Purpose:** Moves the ~400 lines of CPU-heavy allocation logic from the API handler into a standalone worker function
- **Key design decisions:**
  - Uses **synchronous SQLAlchemy** (psycopg2) since Celery workers are not async
  - Derives sync DB URL from `SYNC_DATABASE_URL` env var or by converting `asyncpg://` → `postgresql://`
  - Calls `asyncio.run()` to invoke the existing async LangGraph workflow inside the sync worker
- **Functions:**
  - `execute_allocation(allocation_run_id)` — full pipeline: load payload → upsert drivers/packages → cluster → create routes → invoke LangGraph → persist assignments → update run status
  - `mark_run_status(allocation_run_id, status, error_message)` — update DB status
  - `mark_run_failed(allocation_run_id, error_message)` — convenience wrapper

---

### Artifact Store (2 files)

#### `app/utils/__init__.py`
- Utils package init file

#### `app/utils/artifact_store.py`
- **Purpose:** Redis-backed store for large numpy arrays (effort matrices, distance matrices)
- **Serialization:** Uses `np.save(buffer, matrix, allow_pickle=False)` — **no pickle anywhere**
- **Key class:** `ArtifactStore`
  - `put_matrix(matrix, ttl_seconds=43200, key_prefix="effort") → str` — serialize and store, return Redis key
  - `get_matrix(key) → np.ndarray` — fetch and deserialize
  - `delete_keys(*keys) → int` — explicit cleanup
  - `exists(key) → bool` — check existence
  - `cleanup_prefix(prefix) → int` — bulk delete by prefix
- **Key format:** `artifact:{prefix}:{uuid4_hex}` (e.g., `artifact:effort:a1b2c3d4...`)
- **Default TTL:** 12 hours (auto-cleanup for expired artifacts)
- **Lazy singleton:** `get_artifact_store()` creates instance on first use

---

### CVRP Solver (2 files)

#### `app/solver/__init__.py`
- Solver package init file

#### `app/solver/cvrp_solver.py`
- **Purpose:** Replaces the 1:1 Linear Assignment Problem (LAP) with a proper Capacitated Vehicle Routing Problem (CVRP)
- **Algorithm:** OR-Tools `pywrapcp.RoutingModel` with constraint programming
- **Data classes:**
  - `CVRPVehicle` — driver model (capacity_kg, max_range_km, is_ev, fatigue_penalty)
  - `CVRPNode` — delivery stop (lat/lng, demand_kg, route_id)
  - `CVRPSolution` — result (assignments, distances, loads, efforts, status)
- **Constraints supported:**
  - **Capacity:** `AddDimensionWithVehicleCapacity` — each driver has a weight limit
  - **EV Range:** Custom "Distance" dimension with per-vehicle max — EV drivers cannot exceed `battery_range_km`
  - **Disjunction penalties:** Nodes that can't be served get a high penalty instead of making the problem infeasible
- **Fairness enforcement (two-pass approach):**
  1. Solve for minimum cost (distance + fatigue)
  2. Compute Gini coefficient of per-vehicle effort
  3. If Gini > threshold: penalize overburdened vehicles (multiply their cost by `1 + (effort - avg) / avg`) and re-solve
  4. Repeat up to `max_fairness_iterations` (default 3)
- **Search strategy:**
  - First solution: `PATH_CHEAPEST_ARC`
  - Metaheuristic: `GUIDED_LOCAL_SEARCH`
  - Time limit: configurable (default 30s)
- **Greedy fallback:** If OR-Tools fails/times out, falls back to a nearest-neighbor heuristic that:
  - Sorts nodes by weight (heaviest first)
  - Assigns each to the nearest vehicle with remaining capacity
  - Respects EV range constraints
  - Status marked as `"fallback"` for observability

---

### Async ML Trainer (2 files)

#### `app/trainer/__init__.py`
- Trainer package init file

#### `app/trainer/train_xgboost.py`
- **Purpose:** Offline per-driver XGBoost model training (runs as Celery Beat job)
- **Key functions:**
  - `train_driver_model(driver_id, X, y, model_dir)` — train one driver's model
  - `load_driver_model(model_path)` — safely load via `xgb.Booster.load_model()`
  - `retrain_all_drivers()` — main entry point: queries all drivers, builds feature matrices, trains models, updates DB
- **Model storage:** `model.save_model("models/driver_{id}_v{timestamp}.json")` — **no pickle**
- **Integrity:** SHA-256 checksum computed and stored in DB for each model file
- **Feature set (8 features):** route_difficulty_score, num_packages, total_weight_kg, num_stops, estimated_time_minutes, fatigue_score, recent_avg_effort, recent_hard_days
- **XGBoost config:** `reg:squarederror`, depth=4, lr=0.1, 100 rounds, subsample=0.8
- **Metrics tracked:** MSE, RMSE, R² per model
- **Thresholds:** Minimum 10 samples to train, maximum 500 samples (most recent)

---

### Test Files (4 files)

#### `tests/test_cvrp_solver.py` — 9 tests
- `TestCVRPSolverBasic`: small instance solving, capacity constraint validation, empty input handling, single vehicle/node
- `TestCVRPEVConstraints`: EV range constraint enforcement
- `TestCVRPFairness`: two-pass Gini reduction verification
- `TestCVRPGreedyFallback`: greedy nearest-neighbor correctness
- `TestCVRPScaleEdgeCases`: more packages than drivers, more drivers than packages

#### `tests/test_artifact_store.py` — 8 tests
- Dense/integer/large (2000×500)/1D matrix round-trips
- Missing key `KeyError` raising
- Key deletion and uniqueness
- `TestNoPickleSafety`: verifies no `import pickle` in artifact_store source

#### `tests/test_safe_model_storage.py` — 4 tests
- **Security scan:** scans all production directories for `pickle.loads` and `pickle.dumps`
- XGBoost `save_model`/`load_model` round-trip (predictions match)
- Schema verification: `model_pickle` column removed, `model_path` + `model_checksum` present

#### `tests/test_async_allocation.py` — 7 tests
- POST `/allocate/langgraph` returns 202 + `job_id`
- Empty packages/drivers return 400
- Status endpoint: 404 for missing job, 400 for invalid UUID, 200 for queued job
- Deprecated sync endpoint: exists (doesn't 404), returns 410 when feature flag off

---

## 4. Modified Files

### `app/api/allocation_langgraph.py`
**Complete rewrite.** Previously a single 514-line endpoint. Now provides three endpoints:

| Endpoint | Method | Status | Description |
|----------|--------|--------|-------------|
| `/allocate/langgraph` | POST | **202** | New async endpoint — validates, creates `AllocationRun(QUEUED)`, enqueues Celery task, returns `job_id` |
| `/allocate/status/{job_id}` | GET | 200 | New polling endpoint — returns job status, progress, result link when completed |
| `/allocate/langgraph_sync` | POST | 200 | **Deprecated** — original sync logic preserved verbatim, behind `SYNC_ALLOCATION_ENABLED` feature flag. Returns 410 when flag is `false` |

### `app/models/allocation_run.py`
- **Line 11:** Changed import from `Integer, Float, Text, Date, DateTime, Enum` → added `JSON`
- **Lines 17–22:** Added `QUEUED = "QUEUED"` and `RUNNING = "RUNNING"` to `AllocationRunStatus` enum (before existing PENDING/SUCCESS/FAILED)
- **Line 51–54:** Added `payload_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)` — stores the original allocation request for worker consumption

### `app/schemas/allocation_state.py`
- **Lines 54–62:** Replaced `effort_matrix: Optional[Dict[str, Any]]` with:
  - `effort_matrix_ref: Optional[str]` — Redis key pointing to the matrix stored in artifact_store
  - `effort_matrix_meta: Optional[Dict[str, Any]]` — lightweight metadata (shape, driver_ids, route_ids, avg_effort)

### `app/models/driver_effort_model.py`
- **Line 10:** Changed import from `Boolean, Float, Integer, DateTime, ForeignKey, LargeBinary, JSON` → `Boolean, Float, Integer, String, DateTime, ForeignKey, JSON` (removed `LargeBinary`, added `String`)
- **Lines 14–19:** Updated docstring: "stores per-driver XGBoost model metadata. Models are saved as safe JSON files on disk via xgb.save_model(). NO PICKLE storage is used."
- **Lines 35–45:** Replaced:
  - `model_pickle: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)` → **REMOVED**
  - Added `model_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)` — filesystem path to JSON model
  - Added `model_checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)` — SHA-256 integrity hash

### `app/services/learning_agent.py`
- **Line 9:** `import pickle` → `import os` + `import tempfile`
- **Lines 325–340:** `load_model()` method:
  - `model_record.model_pickle` → `model_record.model_path`
  - `pickle.loads(model_record.model_pickle)` → `xgb.Booster()` + `bst.load_model(model_record.model_path)`
- **Lines 474–518:** Model saving section:
  - `pickle.dumps(model)` → `model.save_model(model_path)` (safe JSON format)
  - Added: SHA-256 checksum computation via `hashlib`
  - Added: `os.makedirs(model_dir, exist_ok=True)` for model directory
  - `existing.model_pickle = model_pickle` → `existing.model_path = model_path` + `existing.model_checksum = model_checksum`
  - `model_pickle=model_pickle` in new record → `model_path=model_path` + `model_checksum=model_checksum`

### `app/config.py`
- **Lines 58–75:** Added new settings to `Settings` class:
  - `redis_url: str = "redis://localhost:6379/0"`
  - `celery_broker_url: Optional[str] = None`
  - `celery_result_backend: Optional[str] = None`
  - `model_storage_dir: str = "models/"`
  - `sync_allocation_enabled: bool = True`
  - Property `effective_broker_url` — returns `celery_broker_url or redis_url`
  - Property `effective_result_backend` — returns `celery_result_backend or redis_url`

### `requirements.txt`
- **Lines 22–27:** Added new section `# Worker Infrastructure (Phase 10 - Async Architecture)`:
  - `celery>=5.3.0`
  - `redis>=5.0.0`
  - `flower>=2.0.0`
  - `psycopg2-binary>=2.9.0`

---

## 5. API Contract Changes

### New: `POST /api/v1/allocate/langgraph` (async)

**Request:** Same `AllocationRequest` JSON payload as before.

**Response:** `HTTP 202 Accepted`
```json
{
  "job_id": "<uuid>",
  "status": "queued",
  "message": "Allocation job accepted. Poll GET /allocate/status/{job_id} for progress."
}
```

### New: `GET /api/v1/allocate/status/{job_id}`

**Response:** `HTTP 200 OK`
```json
{
  "job_id": "<uuid>",
  "status": "QUEUED | RUNNING | SUCCESS | FAILED",
  "date": "2026-02-20",
  "num_drivers": 20,
  "num_routes": 5,
  "num_packages": 100,
  "started_at": "2026-02-20T10:00:00",
  "finished_at": "2026-02-20T10:02:30",
  "result": {
    "global_gini_index": 0.15,
    "global_std_dev": 8.3,
    "global_max_gap": 12.1,
    "assignments_url": "/api/v1/runs/<uuid>/assignments"
  }
}
```
The `result` field only appears when `status == SUCCESS`. An `error` field appears when `status == FAILED`.

### Deprecated: `POST /api/v1/allocate/langgraph_sync`

- Marked `deprecated=True` in OpenAPI spec
- Controlled by `SYNC_ALLOCATION_ENABLED` env var (default `true`)
- Returns `HTTP 410 Gone` when feature flag is `false`
- Identical behavior to the old endpoint when enabled

---

## 6. Database Schema Changes

### `allocation_runs` table

| Change | Column | Type | Notes |
|--------|--------|------|-------|
| ADD | `payload_json` | `JSON` | Stores the original allocation request for worker processing |
| MODIFY | `status` enum | Added `QUEUED`, `RUNNING` | New states for async job lifecycle |

### `driver_effort_models` table

| Change | Column | Type | Notes |
|--------|--------|------|-------|
| REMOVE | `model_pickle` | `LargeBinary` | **Security fix:** no more pickle blobs in DB |
| ADD | `model_path` | `String(500)` | Filesystem path to safe JSON model artifact |
| ADD | `model_checksum` | `String(64)` | SHA-256 hash for integrity verification |

> **Migration required:** Run `alembic revision --autogenerate -m "async_architecture_upgrade"` then `alembic upgrade head`

---

## 7. Security Fixes

### Critical: Pickle RCE Vulnerability Eliminated

**Before:** `learning_agent.py` used `pickle.dumps()` to serialize XGBoost models into `LargeBinary` DB columns, and `pickle.loads()` to deserialize them. If the database were compromised, an attacker could inject arbitrary Python code into the pickled blob and achieve **Remote Code Execution** when the model was loaded.

**After:**
- `import pickle` removed entirely from `learning_agent.py`
- Models saved via `xgboost.Booster.save_model("path.json")` — safe, well-defined format
- Models loaded via `xgb.Booster.load_model("path.json")` — no arbitrary code execution
- `model_pickle` LargeBinary column removed from schema
- SHA-256 checksums stored for integrity verification
- Automated security test scans all production directories for `pickle.loads`/`pickle.dumps`

### Serialization Policy

- **Celery:** `task_serializer="json"`, `accept_content=["json"]` — pickle transport disabled at broker level
- **Artifact store:** `np.save(buffer, matrix, allow_pickle=False)` — numpy pickle disabled
- **XGBoost:** `model.save_model()` → JSON format (no binary pickle)

---

## 8. Dependency Changes

| Package | Version | Purpose |
|---------|---------|---------|
| `celery` | ≥5.3.0 | Task queue for background workers |
| `redis` | ≥5.0.0 | Python Redis client (broker + artifact store) |
| `flower` | ≥2.0.0 | Celery monitoring dashboard |
| `psycopg2-binary` | ≥2.9.0 | Synchronous PostgreSQL driver for worker processes |

All existing dependencies unchanged.

---

## 9. Docker / Infrastructure

### `docker-compose.yml` — 6 Services

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────▶│   web    │────▶│ postgres │
│  (UI)    │     │ :8000    │     │  :5432   │
└──────────┘     └────┬─────┘     └──────────┘
                      │                 ▲
                      ▼                 │
                 ┌──────────┐     ┌─────┴────┐
                 │  redis   │◀───▶│  worker   │
                 │  :6379   │     │ (celery)  │
                 └────┬─────┘     └──────────┘
                      │                 ▲
                      ▼                 │
                 ┌──────────┐     ┌─────┴────┐
                 │  flower  │     │   beat    │
                 │  :5555   │     │ (sched)   │
                 └──────────┘     └──────────┘
```

### Environment Variables (New)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection for broker + cache |
| `SYNC_DATABASE_URL` | (derived) | Synchronous PostgreSQL URL for worker |
| `CELERY_BROKER_URL` | (uses REDIS_URL) | Override Celery broker URL |
| `CELERY_RESULT_BACKEND` | (uses REDIS_URL) | Override Celery result backend |
| `MODEL_STORAGE_DIR` | `models/` | Directory for XGBoost JSON artifacts |
| `SYNC_ALLOCATION_ENABLED` | `true` | Feature flag for deprecated sync endpoint |

---

## 10. Test Suite

### New Tests: 21 total, all passing

```
tests/test_cvrp_solver.py         — 9 PASSED   (CVRP solver)
tests/test_safe_model_storage.py   — 4 PASSED   (security verification)
tests/test_artifact_store.py       — 8 PASSED   (Redis artifact store)
```

**Runtime:** 226 seconds (CVRP solver tests dominate due to OR-Tools computation)

### Security Test Coverage

| Test | What it verifies |
|------|-----------------|
| `test_no_pickle_loads_in_services` | Scans all 8 production directories for `pickle.loads` |
| `test_no_pickle_dumps_in_services` | Scans all 8 production directories for `pickle.dumps` |
| `test_driver_effort_model_has_no_pickle_column` | Inspects SQLAlchemy table columns |
| `test_no_pickle_import` | Verifies artifact_store source has no pickle import |

---

## 11. Migration Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Alembic Migration
```bash
alembic revision --autogenerate -m "async_architecture_upgrade"
alembic upgrade head
```
This will:
- Add `QUEUED` and `RUNNING` to `allocation_runs.status` enum
- Add `payload_json` JSON column to `allocation_runs`
- Replace `model_pickle` (LargeBinary) with `model_path` (String) + `model_checksum` (String) in `driver_effort_models`

### Step 3: Start Infrastructure
```bash
docker compose up -d redis postgres
```

### Step 4: Run Application
```bash
# Terminal 1: Web server
uvicorn app.main:app --reload

# Terminal 2: Celery worker
celery -A app.core.celery_app:celery_app worker --loglevel=info --concurrency=4 -Q allocation

# Terminal 3 (optional): Celery Beat
celery -A app.core.celery_app:celery_app beat --loglevel=info

# Terminal 4 (optional): Flower monitoring
celery -A app.core.celery_app:celery_app flower --port=5555
```

### Step 5: Update Frontend
Update the frontend to use the async flow:
1. `POST /api/v1/allocate/langgraph` → receive `job_id`
2. Poll `GET /api/v1/allocate/status/{job_id}` every 2–5 seconds
3. When `status == "SUCCESS"`, fetch assignments from the result URL

### Step 6: Validate and Disable Sync
1. Test both async and sync endpoints in staging
2. Compare allocation results between old LAP and new CVRP
3. Set `SYNC_ALLOCATION_ENABLED=false` to disable the deprecated endpoint
4. Remove `/langgraph_sync` endpoint in a future release

---

*End of changelog.*
