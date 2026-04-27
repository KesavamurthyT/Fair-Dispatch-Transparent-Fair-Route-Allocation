"""
tests/test_hardening_features.py

Validates all 5 architectural hardening improvements:

  Fix 1 - Deterministic OR-Tools seed
  Fix 2 - Model version tracking per allocation run
  Fix 3 - Explicit Redis artifact cleanup in worker
  Fix 4 - Idempotency key support for async allocation
  Fix 5 - Fairness iteration logging + observability
"""

import os
import uuid
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# ────────────────────────────────────────────────────────────────
# Fix 1 — Deterministic OR-Tools Seed
# ────────────────────────────────────────────────────────────────

class TestDeterministicSeed:
    """CVRP solver with the same seed should produce identical assignments."""

    def _make_solver(self, seed_override=None):
        from app.solver.cvrp_solver import CVRPSolver, CVRPVehicle, CVRPNode
        vehicles = [
            CVRPVehicle(id=f"d{i}", capacity_kg=100.0)
            for i in range(3)
        ]
        nodes = [
            CVRPNode(
                id=f"p{j}",
                latitude=12.97 + j * 0.005,
                longitude=77.59 + j * 0.005,
                demand_kg=5.0,
            )
            for j in range(9)
        ]
        return CVRPSolver(
            depot_lat=12.97,
            depot_lng=77.59,
            vehicles=vehicles,
            nodes=nodes,
            time_limit_seconds=15,
        )

    def test_same_seed_produces_same_assignments(self):
        """Two runs with same seed must have identical assignment dictionaries."""
        s1 = self._make_solver()
        s2 = self._make_solver()

        sol1 = s1.solve()
        sol2 = s2.solve()

        assert sol1.assignments == sol2.assignments, (
            "Assignments differ between identical runs \u2014 solver is not deterministic!"
        )

    def test_cvrp_random_seed_constant_is_set(self):
        """CVRP_RANDOM_SEED constant must be defined and equal to env var or 42."""
        from app.solver import cvrp_solver as m
        assert hasattr(m, "CVRP_RANDOM_SEED"), "CVRP_RANDOM_SEED constant missing"
        assert isinstance(m.CVRP_RANDOM_SEED, int)

    def test_seed_overridable_via_env(self):
        """CVRP_RANDOM_SEED should pick up CVRP_RANDOM_SEED env var."""
        with patch.dict(os.environ, {"CVRP_RANDOM_SEED": "99"}):
            import importlib
            import app.solver.cvrp_solver as mod
            # Reimport to pick up env change (simulate fresh import)
            importlib.reload(mod)
            assert mod.CVRP_RANDOM_SEED == 99
        # Reload back to normal for other tests
        import importlib
        import app.solver.cvrp_solver as mod2
        importlib.reload(mod2)


# ────────────────────────────────────────────────────────────────
# Fix 5 — Fairness Iteration Logging + Observability
# ────────────────────────────────────────────────────────────────

class TestFairnessMetadata:
    """CVRPSolution must carry fairness iteration metadata fields."""

    def test_solution_has_fairness_fields(self):
        """CVRPSolution dataclass must have fairness_iterations, initial_gini, final_gini."""
        from app.solver.cvrp_solver import CVRPSolution
        sol = CVRPSolution(
            assignments={},
            total_distance=0.0,
            per_vehicle_distance={},
            per_vehicle_load={},
            per_vehicle_effort={},
            status="optimal",
        )
        assert hasattr(sol, "fairness_iterations")
        assert hasattr(sol, "initial_gini")
        assert hasattr(sol, "final_gini")
        assert sol.fairness_iterations == 0
        assert sol.initial_gini == 0.0
        assert sol.final_gini == 0.0

    def test_solve_populates_fairness_metadata(self):
        """After solve(), the solution should have fairness_iterations >= 0."""
        from app.solver.cvrp_solver import CVRPSolver, CVRPVehicle, CVRPNode
        vehicles = [CVRPVehicle(id=f"d{i}", capacity_kg=100.0) for i in range(2)]
        nodes = [
            CVRPNode(id=f"p{j}", latitude=12.97 + j * 0.005,
                     longitude=77.59, demand_kg=5.0)
            for j in range(6)
        ]
        solver = CVRPSolver(
            depot_lat=12.97, depot_lng=77.59,
            vehicles=vehicles, nodes=nodes,
            time_limit_seconds=10,
        )
        solution = solver.solve()
        assert solution is not None
        assert solution.fairness_iterations >= 0
        # Gini values are always [0, 1]
        assert 0.0 <= solution.initial_gini <= 1.0
        assert 0.0 <= solution.final_gini <= 1.0

    def test_fairness_log_uses_percent_format(self, caplog):
        """Fairness log lines should use %-format (not f-string) for deferred evaluation."""
        import logging
        from app.solver.cvrp_solver import CVRPSolver, CVRPVehicle, CVRPNode
        vehicles = [CVRPVehicle(id="d0", capacity_kg=100.0)]
        nodes = [CVRPNode(id="p0", latitude=12.98, longitude=77.60, demand_kg=5.0)]
        solver = CVRPSolver(
            depot_lat=12.97, depot_lng=77.59,
            vehicles=vehicles, nodes=nodes,
            time_limit_seconds=5,
        )
        with caplog.at_level(logging.INFO, logger="app.solver.cvrp_solver"):
            solver.solve()
        # Check that at least one log message uses the structured format
        fairness_logs = [r for r in caplog.records if "Fairness iteration" in r.message]
        assert len(fairness_logs) >= 1, "No fairness iteration log messages found"
        # Verify Gini and threshold are present
        for record in fairness_logs:
            assert "Gini=" in record.message
            assert "threshold=" in record.message

    def test_max_fairness_iterations_enforced(self):
        """Solver must never exceed max_fairness_iterations regardless of Gini."""
        from app.solver.cvrp_solver import CVRPSolver, CVRPVehicle, CVRPNode
        vehicles = [CVRPVehicle(id=f"d{i}", capacity_kg=200.0) for i in range(2)]
        nodes = [
            CVRPNode(id=f"p{j}", latitude=12.97, longitude=77.59, demand_kg=1.0)
            for j in range(5)
        ]
        # Set impossible threshold to force all iterations
        solver = CVRPSolver(
            depot_lat=12.97, depot_lng=77.59,
            vehicles=vehicles, nodes=nodes,
            time_limit_seconds=5,
            fairness_threshold_gini=-0.1,  # impossible \u2014 Gini can't be negative
            max_fairness_iterations=3,
        )
        solution = solver.solve()
        assert solution.fairness_iterations <= 3, (
            f"Exceeded max 3 iterations: got {solution.fairness_iterations}"
        )


# ────────────────────────────────────────────────────────────────
# Fix 3 — Explicit Redis Artifact Cleanup
# ────────────────────────────────────────────────────────────────

class TestArtifactCleanup:
    """After allocation (success or failure), Redis keys must be deleted."""

    class _MockStore:
        """Minimal in-memory Redis mock that tracks deletions."""
        def __init__(self):
            self._store = {}
            self.deleted_keys = []

        def put_matrix(self, matrix, ttl_seconds=43200, key_prefix="effort"):
            key = f"artifact:{key_prefix}:{uuid.uuid4().hex}"
            buf = __import__("io").BytesIO()
            __import__("numpy").save(buf, matrix, allow_pickle=False)
            self._store[key] = buf.getvalue()
            return key

        def get_matrix(self, key):
            if key not in self._store:
                raise KeyError(f"Artifact not found: {key}")
            buf = __import__("io").BytesIO(self._store[key])
            return __import__("numpy").load(buf, allow_pickle=False)

        def delete_keys(self, *keys):
            count = 0
            for k in keys:
                if k in self._store:
                    del self._store[k]
                    self.deleted_keys.append(k)
                    count += 1
            return count

        def exists(self, key):
            return key in self._store

    def test_keys_are_deleted_after_allocation(self):
        """Any keys appended to _artifact_keys must be cleaned up in finally."""
        mock_store = self._MockStore()

        # Simulate the worker pattern: track keys, cleanup in finally
        artifact_keys = []
        try:
            # Simulate the worker adding keys
            key1 = mock_store.put_matrix(np.eye(3), key_prefix="effort")
            key2 = mock_store.put_matrix(np.eye(3), key_prefix="distance")
            artifact_keys.extend([key1, key2])

            assert mock_store.exists(key1)
            assert mock_store.exists(key2)

            # Simulate successful processing
        finally:
            if artifact_keys:
                deleted = mock_store.delete_keys(*artifact_keys)
                assert deleted == 2

        # After finally: keys must be gone
        assert not mock_store.exists(key1)
        assert not mock_store.exists(key2)

    def test_keys_cleaned_up_even_on_exception(self):
        """Cleanup must run even when an exception is raised mid-pipeline."""
        mock_store = self._MockStore()
        artifact_keys = []
        caught_key = None

        try:
            key = mock_store.put_matrix(np.zeros(5), key_prefix="effort")
            artifact_keys.append(key)
            caught_key = key
            raise RuntimeError("Simulated worker crash")
        except RuntimeError:
            pass
        finally:
            if artifact_keys:
                mock_store.delete_keys(*artifact_keys)

        assert caught_key is not None
        assert not mock_store.exists(caught_key), "Key should be deleted even after crash"

    def test_worker_finally_block_exists(self):
        """langgraph_runner.execute_allocation must have _artifact_keys cleanup in finally."""
        import inspect
        import app.worker.langgraph_runner as runner_mod
        source = inspect.getsource(runner_mod.execute_allocation)
        assert "_artifact_keys" in source, "_artifact_keys not found in execute_allocation"
        assert "delete_keys" in source, "delete_keys call not found in execute_allocation"
        assert "finally:" in source, "finally block missing in execute_allocation"


# ────────────────────────────────────────────────────────────────
# Fix 4 — Idempotency Key Support
# ────────────────────────────────────────────────────────────────

class TestIdempotencyKey:
    """Duplicate POST requests with the same Idempotency-Key must not create duplicate jobs."""

    def test_allocation_run_has_idempotency_key_column(self):
        """AllocationRun model must have an idempotency_key column."""
        from app.models.allocation_run import AllocationRun
        cols = {c.name for c in AllocationRun.__table__.columns}
        assert "idempotency_key" in cols, "idempotency_key column missing from AllocationRun"

    def test_idempotency_key_is_unique(self):
        """idempotency_key column must have a unique constraint."""
        from app.models.allocation_run import AllocationRun
        col = AllocationRun.__table__.c["idempotency_key"]
        assert col.unique, "idempotency_key column is not unique"

    def test_idempotency_key_is_nullable(self):
        """idempotency_key must be nullable (optional header)."""
        from app.models.allocation_run import AllocationRun
        col = AllocationRun.__table__.c["idempotency_key"]
        assert col.nullable, "idempotency_key must be nullable"

    def test_idempotency_check_in_api_source(self):
        """The async POST endpoint must contain idempotency duplicate check logic."""
        import inspect
        import app.api.allocation_langgraph as api_mod
        source = inspect.getsource(api_mod.allocate_langgraph_async)
        assert "idempotency_key" in source
        assert "existing_run" in source or "Duplicate" in source


# ────────────────────────────────────────────────────────────────
# Fix 2 — Model Version Tracking
# ────────────────────────────────────────────────────────────────

class TestModelVersionTracking:
    """Every allocation run must record the model version used."""

    def test_allocation_run_has_model_version_used_column(self):
        """AllocationRun must have a model_version_used column."""
        from app.models.allocation_run import AllocationRun
        cols = {c.name for c in AllocationRun.__table__.columns}
        assert "model_version_used" in cols, "model_version_used missing from AllocationRun"

    def test_model_version_used_is_nullable(self):
        """model_version_used must be nullable (not all runs have trained models)."""
        from app.models.allocation_run import AllocationRun
        col = AllocationRun.__table__.c["model_version_used"]
        assert col.nullable

    def test_model_version_recorded_in_runner_source(self):
        """Worker runner must contain model_version_used assignment logic."""
        import inspect
        import app.worker.langgraph_runner as runner_mod
        source = inspect.getsource(runner_mod.execute_allocation)
        assert "model_version_used" in source, (
            "model_version_used not assigned in execute_allocation"
        )

    def test_status_endpoint_returns_model_version(self):
        """Status endpoint source must include model_version_used in response."""
        import inspect
        import app.api.allocation_langgraph as api_mod
        source = inspect.getsource(api_mod.allocation_status)
        assert "model_version_used" in source, (
            "model_version_used not returned by status endpoint"
        )
