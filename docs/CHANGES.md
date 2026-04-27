# Fair Dispatch System — Changes Documentation

> **Session date:** 2026-02-21  
> **Scope:** Architectural hardening, SQLite compatibility, visualization bug fixes  
> **Policy:** No existing business logic was modified. All changes are additive, observability-focused, or bug-fix-only.

---

## Table of Contents

1. [Hardening Feature 1 — Deterministic OR-Tools Seed](#1-hardening-feature-1--deterministic-or-tools-seed)
2. [Hardening Feature 2 — Model Version Tracking](#2-hardening-feature-2--model-version-tracking)
3. [Hardening Feature 3 — Explicit Redis Artifact Cleanup](#3-hardening-feature-3--explicit-redis-artifact-cleanup)
4. [Hardening Feature 4 — Idempotency Key Support](#4-hardening-feature-4--idempotency-key-support)
5. [Hardening Feature 5 — Fairness Iteration Logging](#5-hardening-feature-5--fairness-iteration-logging)
6. [New Files Created](#6-new-files-created)
7. [Bug Fix — SQLite Enum Compatibility](#7-bug-fix--sqlite-enum-compatibility)
8. [Bug Fix — AllocationState Missing Field](#8-bug-fix--allocationstate-missing-effort_matrix-field)
9. [Bug Fix — Visualization Demo Page](#9-bug-fix--visualization-demo-page)
10. [Summary Table](#10-summary-table)

---

## 1. Hardening Feature 1 — Deterministic OR-Tools Seed

**File:** `app/solver/cvrp_solver.py`

### What changed

- Added `import os` at the top of the file.
- Added a module-level constant:
  ```python
  CVRP_RANDOM_SEED = int(os.getenv("CVRP_RANDOM_SEED", "42"))
  ```
- In `CVRPSolver._solve_iteration()`, the OR-Tools search parameters are now configured with:
  ```python
  try:
      search_params.random_seed = CVRP_RANDOM_SEED
  except AttributeError:
      # OR-Tools version does not support random_seed — PATH_CHEAPEST_ARC is deterministic
      logger.debug("OR-Tools random_seed not supported in this version ...")
  try:
      search_params.log_search = False  # suppress noisy solver output
  except AttributeError:
      pass
  ```
  The `try/except AttributeError` guards are necessary because the installed OR-Tools version's `RoutingSearchParameters` protobuf may not expose `random_seed` as a field. The guard makes the code forward-compatible — if a newer version adds the field, it will be set automatically.

### Why

Reproducibility: the OR-Tools GLS metaheuristic is non-deterministic by default. A fixed seed means back-to-back runs on the same input produce the same route assignments.  
`PATH_CHEAPEST_ARC` (the first-solution strategy) is already deterministic regardless of seed.

### Configurable via

```bash
CVRP_RANDOM_SEED=123 uvicorn app.main:app ...
```

---

## 2. Hardening Feature 2 — Model Version Tracking

### Files changed

#### `app/models/allocation_run.py`

- Added `String` and `Index` to the SQLAlchemy imports.
- Added two new columns to the `AllocationRun` model:
  ```python
  model_version_used: Mapped[Optional[str]] = mapped_column(
      String(100), nullable=True
  )
  idempotency_key: Mapped[Optional[str]] = mapped_column(
      String(100), unique=True, nullable=True, index=True
  )
  ```
  - `model_version_used` — stores a string like `"v3"` identifying which `DriverEffortModel` version was active when the allocation ran.
  - `idempotency_key` — stores a client-supplied UUID (covered separately in Feature 4).

#### `app/worker/langgraph_runner.py`

- Before invoking the LangGraph workflow, the worker now queries the database for the most recently active `DriverEffortModel` and records its version:
  ```python
  active_model = session.query(DriverEffortModel).filter_by(is_active=True).first()
  if active_model:
      run.model_version_used = f"v{active_model.model_version}"
      session.commit()
  ```

#### `app/api/allocation_langgraph.py`

- The `GET /api/v1/allocate/status/{job_id}` response now includes `model_version_used`:
  ```python
  response = {
      ...
      "model_version_used": run.model_version_used,
  }
  ```

### Why

Traceability and reproducibility: knowing which exact model version was used for a past allocation allows the result to be reproduced or audited.

---

## 3. Hardening Feature 3 — Explicit Redis Artifact Cleanup

**File:** `app/worker/langgraph_runner.py`

### What changed

Inside `execute_allocation()`, a list `_artifact_keys: List[str] = []` is initialised before the `try` block. Artifact keys are appended to it as they are created. In the `finally` block:

```python
finally:
    session.close()
    if _artifact_keys:
        try:
            store.delete_keys(*_artifact_keys)
            logger.info("Cleaned up %d artifact keys from Redis", len(_artifact_keys))
        except Exception as cleanup_err:
            logger.warning("Artifact cleanup error: %s", cleanup_err)
```

### Why

Belt-and-suspenders safety: Redis TTLs already expire keys, but if a worker process crashes mid-flight, the TTL might not fire immediately. Explicit deletion in the `finally` block guarantees cleanup regardless of whether the allocation succeeded, failed, or raised an unhandled exception.

---

## 4. Hardening Feature 4 — Idempotency Key Support

**File:** `app/api/allocation_langgraph.py`

### What changed

The `POST /api/v1/allocate/langgraph` endpoint now accepts an optional HTTP header:

```
Idempotency-Key: <client-generated UUID>
```

Implementation:
```python
async def allocate_langgraph_async(
    request: AllocationRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(
        default=None,
        alias="Idempotency-Key",
        description="Optional client-generated idempotency key (UUID). "
                    "Prevents duplicate allocations on retry.",
    ),
) -> dict:
    if idempotency_key:
        existing_result = await db.execute(
            select(AllocationRun).where(
                AllocationRun.idempotency_key == idempotency_key
            )
        )
        existing_run = existing_result.scalar_one_or_none()
        if existing_run:
            return {
                "job_id": str(existing_run.id),
                "status": ...,
                "message": "Duplicate request — returning existing job.",
                "idempotency_key": idempotency_key,
            }
    # ... create new run and enqueue job
```

The `AllocationRun` row is also created with the key stored:
```python
run.idempotency_key = idempotency_key
```

### Why

Client-side retries (e.g. on network timeout) must not create duplicate allocation jobs. A unique idempotency key means the second identical request is a no-op that returns the original job ID.

---

## 5. Hardening Feature 5 — Fairness Iteration Logging

**File:** `app/solver/cvrp_solver.py`

### What changed

#### `CVRPSolution` dataclass — new fields

```python
@dataclass
class CVRPSolution:
    ...
    fairness_iterations: int = 0
    initial_gini: float = 0.0
    final_gini: float = 0.0
```

#### Structured logging in `CVRPSolver.solve()`

Fairness iteration progress is now logged using `%`-format (structured, no f-strings):

```python
logger.info(
    "Fairness iteration %d/%d | Gini=%.4f → %.4f | threshold=%.4f",
    iteration,
    max_fairness_iterations,
    current_gini,
    new_gini,
    gini_threshold,
)
```

The `CVRPSolution` returned by `solve()` is now populated:

```python
solution.fairness_iterations = iteration_count
solution.initial_gini = initial_gini
solution.final_gini = final_gini
```

#### API exposure

`GET /api/v1/allocate/status/{job_id}` — if the run succeeded and `result_json` contains `fairness_metadata`, it is returned:

```python
try:
    result_meta = getattr(run, "result_json", None) or {}
    if result_meta.get("fairness_metadata"):
        response["fairness_metadata"] = result_meta["fairness_metadata"]
except Exception:
    pass
```

### Why

Observability: operators can now see in logs whether the fairness optimiser converged quickly or iterated many times, and by how much the Gini coefficient was reduced. The data is also accessible programmatically via the status endpoint.

---

## 6. New Files Created

### `alembic/versions/a1b2c3d4e5f6_hardening_columns.py`

Alembic migration script that applies the database schema changes introduced by Features 2 and 4.

**Upgrade** (`upgrade()` function):
- Adds `model_version_used VARCHAR(100)` (nullable) to `allocation_runs`.
- Adds `idempotency_key VARCHAR(100)` (nullable, unique) to `allocation_runs`.
- Creates a unique index on `idempotency_key`.
- Adds `model_path VARCHAR(500)` (nullable) to `driver_effort_models` (safe model file path instead of pickle blob).
- Adds `model_checksum VARCHAR(64)` (nullable) to `driver_effort_models` (SHA-256 integrity hash).

**Downgrade** (`downgrade()` function):
- Drops all added columns and indexes, reverting the schema.

> **Note:** Run `alembic upgrade head` against a **PostgreSQL** database to apply. The SQLite dev database is re-created automatically from SQLAlchemy metadata on startup (via `init_db()`), so no Alembic migration is needed locally.

---

### `tests/test_hardening_features.py`

New test file with **18 tests** covering all 5 hardening improvements.

| Class | Tests | What is verified |
|---|---|---|
| `TestDeterministicSeed` | 3 | `CVRP_RANDOM_SEED` constant exists, is overridable via env var, and produces identical assignments across two runs |
| `TestFairnessMetadata` | 4 | `CVRPSolution` has `fairness_iterations`/`initial_gini`/`final_gini` fields; `solve()` populates them; logging uses `%` format; `max_fairness_iterations` is enforced |
| `TestArtifactCleanup` | 3 | `delete_keys` is called after allocation; it is still called when the allocation raises an exception; the `finally` block exists in worker source |
| `TestIdempotencyKey` | 4 | `AllocationRun` has `idempotency_key` column; it's nullable; it's unique; API source contains idempotency check logic |
| `TestModelVersionTracking` | 4 | `AllocationRun` has `model_version_used` column; it's nullable; worker source queries the active model; status endpoint returns the field |

All 18 new tests plus 21 pre-existing tests pass (39/39 total).

---

## 7. Bug Fix — SQLite Enum Compatibility

**Files:** `app/models/allocation_run.py`, `app/models/appeal.py`, `app/models/delivery_log.py`, `app/models/driver.py`, `app/models/package.py`, `app/models/route_swap.py`, `app/models/stop_issue.py`

### Problem

SQLAlchemy `Enum(SomeEnum)` without options defaults to `native_enum=True`, which attempts to create PostgreSQL `CREATE TYPE` enum objects. SQLite (used in local development) does not support native enum types and raises:

```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) ...
Background on this error at: https://sqlalche.me/e/20/e3q8
```

### Fix

All `Enum(...)` column definitions across 7 model files were updated to:
```python
Enum(SomeName, native_enum=False)
```

`native_enum=False` tells SQLAlchemy to store enums as `VARCHAR` columns, which works on both SQLite (local dev) and PostgreSQL (production). The valid values are still constrained at the application layer by the Python `enum.Enum` class.

### Affected columns

| File | Column | Enum type |
|---|---|---|
| `allocation_run.py` | `status` | `AllocationRunStatus` |
| `appeal.py` | `status` | `AppealStatus` |
| `delivery_log.py` | `status` | `DeliveryStatus` |
| `delivery_log.py` | `issue_type` | `DeliveryIssueType` |
| `driver.py` | `preferred_language` | `PreferredLanguage` |
| `driver.py` | `vehicle_type` | `VehicleType` |
| `driver.py` | `hardest_aspect` | `HardestAspect` |
| `package.py` | `priority` | `PackagePriority` |
| `route_swap.py` | `status` | `SwapRequestStatus` |
| `stop_issue.py` | `issue_type` | `StopIssueType` |

---

## 8. Bug Fix — `AllocationState` Missing `effort_matrix` Field

**File:** `app/schemas/allocation_state.py`

### Problem

`langgraph_nodes.py` → `ml_effort_node()` returns a dict update containing:
```python
{"effort_matrix": effort_dict, "decision_logs": ...}
```

However, `AllocationState` (a Pydantic `BaseModel`) only declared `effort_matrix_ref` and `effort_matrix_meta`. Pydantic models discard unknown fields by default, so `effort_matrix` was silently dropped from the LangGraph state. When `route_planner_node()` later accessed `state.effort_matrix["matrix"]`, it raised:

```
AttributeError: 'AllocationState' object has no attribute 'effort_matrix'
During task with name 'route_planner_1'
```

This caused a `500 Internal Server Error` on every call to the sync and async allocation endpoints.

### Fix

Added the missing field to `AllocationState`:

```python
effort_matrix: Optional[Dict[str, Any]] = Field(
    default=None,
    description="Inline effort matrix dict: matrix, driver_ids, route_ids, stats, breakdown, infeasible_pairs"
)
```

All downstream nodes (`route_planner_node`, `route_planner_reoptimize_node`, `driver_liaison_node`, `final_resolution_node`, `explainability_node`) that access `state.effort_matrix[...]` now function correctly.

---

## 9. Bug Fix — Visualization Demo Page

**File:** `frontend/visualization.html`

### Problems fixed

#### A. Outdated request JSON payload (422 Unprocessable Entity)

The demo textarea shipped with a JSON payload matching an older version of `AllocationRequest`. Updated to the current schema:

| Field (old) | Field (new) | Notes |
|---|---|---|
| `external_id` | `id` | Renamed in Pydantic schema |
| `vehicle_type` | *(removed)* | Not in current `DriverInput` |
| `lat`, `lng` | `latitude`, `longitude` | Renamed in `PackageInput` |
| *(missing)* | `address` | Required field |
| *(missing)* | `fragility_level` | Required field (1–5) |
| *(missing)* | `preferred_language` | `"en"` / `"ta"` etc. |
| *(missing)* | `warehouse` | Required `{lat, lng}` object |
| `"standard"`, `"express"`, `"bulk"` | `"HIGH"`, `"EXPRESS"` | Uppercase Python enum values |
| `"2026-02-05"` | `"2026-02-21"` | Updated to current date |

#### B. Async endpoint without a Celery worker (500 Internal Server Error)

The `runAllocation()` JavaScript function was calling `POST /api/v1/allocate/langgraph` (the async, 202-returning endpoint). This endpoint enqueues a job via Celery, which requires Redis + a running Celery worker. Without those, the job is never processed.

**Fix:** Switched the demo to the synchronous endpoint:
```js
// Before
const response = await fetch('/api/v1/allocate/langgraph', {...});

// After
const response = await fetch('/api/v1/allocate/langgraph_sync', {...});
```

The sync endpoint (`SYNC_ALLOCATION_ENABLED=true` in `.env`) executes the full LangGraph pipeline in-process and returns the complete result immediately — no external dependencies required.

#### C. Unhelpful error display for non-string `detail` (displays `[object Object]`)

```js
// Before
throw new Error(error.detail?.message || error.detail || 'Allocation failed');

// After
throw new Error(
    error.detail?.message ||
    (typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail)) ||
    'Allocation failed'
);
```

When Pydantic returns a 422 `detail` as an array of validation errors, `JSON.stringify` is now used so the full error is readable in the Response panel.

---

## 10. Summary Table

| # | Category | File(s) | Nature |
|---|---|---|---|
| 1 | Hardening | `app/solver/cvrp_solver.py` | Deterministic OR-Tools seed via env var |
| 2 | Hardening | `app/models/allocation_run.py`, `app/worker/langgraph_runner.py`, `app/api/allocation_langgraph.py` | Model version tracking per allocation run |
| 3 | Hardening | `app/worker/langgraph_runner.py` | Explicit Redis artifact cleanup in `finally` block |
| 4 | Hardening | `app/models/allocation_run.py`, `app/api/allocation_langgraph.py` | Idempotency key header on async allocation endpoint |
| 5 | Hardening | `app/solver/cvrp_solver.py`, `app/api/allocation_langgraph.py` | Fairness iteration structured logging + API exposure |
| 6 | New file | `alembic/versions/a1b2c3d4e5f6_hardening_columns.py` | Alembic migration for new columns |
| 7 | New file | `tests/test_hardening_features.py` | 18 tests for all 5 hardening improvements |
| 8 | Bug fix | 7 model files | SQLite compatibility: `native_enum=False` on all Enum columns |
| 9 | Bug fix | `app/schemas/allocation_state.py` | Added missing `effort_matrix` field to `AllocationState` |
| 10 | Bug fix | `frontend/visualization.html` | Fixed demo JSON payload, endpoint, and error display |

---

## Environment Variables Added

| Variable | Default | Description |
|---|---|---|
| `CVRP_RANDOM_SEED` | `42` | Integer seed for OR-Tools solver reproducibility |

---

## Test Results

```
39 passed, 2 warnings in 391.49s
 - 18 new tests in tests/test_hardening_features.py
 - 9  existing tests in tests/test_cvrp_solver.py
 - 4  existing tests in tests/test_safe_model_storage.py
 - 8  existing tests in tests/test_artifact_store.py
```

---

*Document generated: 2026-02-21*
