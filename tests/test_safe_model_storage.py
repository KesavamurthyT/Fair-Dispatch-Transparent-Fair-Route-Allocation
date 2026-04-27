"""
Tests to verify NO unsafe pickle usage exists in production code.

Also verifies that XGBoost models are saved/loaded using the safe
save_model/load_model API instead of pickle.
"""

import os
import pytest
import tempfile
import numpy as np


class TestNoPickleInProduction:
    """Verify that pickle is not used unsafely in any production module."""

    PRODUCTION_DIRS = [
        "app/services",
        "app/trainer",
        "app/worker",
        "app/solver",
        "app/utils",
        "app/api",
        "app/core",
        "app/models",
    ]

    def _get_python_files(self, directory: str) -> list:
        """Get all .py files in a directory."""
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), directory)
        files = []
        if os.path.exists(root):
            for fname in os.listdir(root):
                if fname.endswith(".py") and not fname.startswith("__"):
                    files.append(os.path.join(root, fname))
        return files

    def test_no_pickle_loads_in_services(self):
        """No pickle.loads in app/services (the RCE vulnerability)."""
        for directory in self.PRODUCTION_DIRS:
            for filepath in self._get_python_files(directory):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Allow 'import pickle' only if there's no .loads or .dumps
                if "pickle.loads" in content:
                    pytest.fail(
                        f"SECURITY: pickle.loads found in {filepath}. "
                        f"This is an RCE vulnerability!"
                    )

    def test_no_pickle_dumps_in_services(self):
        """No pickle.dumps in production code."""
        for directory in self.PRODUCTION_DIRS:
            for filepath in self._get_python_files(directory):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                if "pickle.dumps" in content:
                    pytest.fail(
                        f"SECURITY: pickle.dumps found in {filepath}. "
                        f"Use safe serialization instead!"
                    )


class TestXGBoostSafeStorage:
    """Test that XGBoost models can be safely saved/loaded without pickle."""

    def test_save_and_load_model_json(self):
        """Train a small model, save with save_model, load with load_model."""
        try:
            import xgboost as xgb
        except ImportError:
            pytest.skip("XGBoost not installed")

        # Train a tiny model
        X = np.random.rand(50, 4).astype(np.float32)
        y = np.random.rand(50).astype(np.float32)
        dtrain = xgb.DMatrix(X, label=y)

        params = {"objective": "reg:squarederror", "max_depth": 2}
        bst = xgb.train(params, dtrain, num_boost_round=10)

        # Save using safe JSON format
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            model_path = f.name

        try:
            bst.save_model(model_path)

            # Load from JSON
            bst2 = xgb.Booster()
            bst2.load_model(model_path)

            # Predictions should match
            dtest = xgb.DMatrix(X)
            pred1 = bst.predict(dtest)
            pred2 = bst2.predict(dtest)

            np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)
        finally:
            os.unlink(model_path)

    def test_driver_effort_model_has_no_pickle_column(self):
        """Verify the DriverEffortModel no longer has a model_pickle column."""
        from app.models.driver_effort_model import DriverEffortModel

        columns = DriverEffortModel.__table__.columns
        column_names = [c.name for c in columns]

        assert "model_pickle" not in column_names, (
            "model_pickle column still exists! It should be replaced with model_path."
        )
        assert "model_path" in column_names, (
            "model_path column missing from DriverEffortModel."
        )
        assert "model_checksum" in column_names, (
            "model_checksum column missing from DriverEffortModel."
        )
