"""
Tests for DriverEffortLearner XGBoost implementation.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/tests", 1)[0])

from app.services.learning_agent import DriverEffortLearner


class TestDriverEffortLearner:
    """Tests for DriverEffortLearner class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock async database session."""
        mock = AsyncMock()
        mock.execute = AsyncMock()
        mock.add = MagicMock()
        return mock
    
    @pytest.fixture
    def learner(self, mock_db):
        """Create DriverEffortLearner instance."""
        return DriverEffortLearner(mock_db)
    
    def test_feature_names_defined(self, learner):
        """Test that feature names are properly defined."""
        assert len(learner.FEATURE_NAMES) > 0
        assert "num_packages" in learner.FEATURE_NAMES
        assert "total_weight_kg" in learner.FEATURE_NAMES
        assert "num_stops" in learner.FEATURE_NAMES
    
    def test_min_training_samples(self, learner):
        """Test minimum training samples requirement."""
        assert learner.MIN_TRAINING_SAMPLES == 10
        assert learner.MAX_TRAINING_SAMPLES == 100
    
    @pytest.mark.asyncio
    async def test_load_model_no_record(self, learner, mock_db):
        """Test load_model returns None when no record exists."""
        mock_db.execute.return_value = MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        )
        
        driver_id = uuid4()
        model = await learner.load_model(driver_id)
        
        assert model is None
    
    @pytest.mark.asyncio
    async def test_get_model_version_no_record(self, learner, mock_db):
        """Test get_model_version returns None when no record exists."""
        mock_db.execute.return_value = MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        )
        
        driver_id = uuid4()
        version = await learner.get_model_version(driver_id)
        
        assert version is None
    
    @pytest.mark.asyncio
    async def test_predict_effort_no_model(self, learner, mock_db):
        """Test predict_effort returns None when no model exists."""
        mock_db.execute.return_value = MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        )
        
        driver_id = uuid4()
        route_features = {
            "num_packages": 20,
            "total_weight_kg": 50.0,
            "num_stops": 10,
        }
        
        prediction, version = await learner.predict_effort(driver_id, route_features)
        
        assert prediction is None
        assert version is None
    
    @pytest.mark.asyncio
    async def test_update_model_insufficient_data(self, learner, mock_db):
        """Test update_model skips when insufficient data."""
        # Return only 5 records (less than MIN_TRAINING_SAMPLES)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            MagicMock(actual_effort=50.0) for _ in range(5)
        ]
        mock_db.execute.return_value = mock_result
        
        driver_id = uuid4()
        result = await learner.update_model(driver_id)
        
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_data"
        assert result["samples"] == 5
    
    @pytest.mark.asyncio
    async def test_get_model_status_no_record(self, learner, mock_db):
        """Test get_model_status returns None when no record exists."""
        mock_db.execute.return_value = MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        )
        
        driver_id = uuid4()
        status = await learner.get_model_status(driver_id)
        
        assert status is None
    
    @pytest.mark.asyncio
    async def test_get_model_status_with_record(self, learner, mock_db):
        """Test get_model_status returns correct data."""
        driver_id = uuid4()
        mock_model = MagicMock(
            model_version=3,
            training_samples=50,
            current_mse=8.5,
            r2_score=0.85,
            mse_history={"values": [10.0, 9.0, 8.5]},
            active=True,
            last_trained_at=datetime.utcnow(),
        )
        mock_db.execute.return_value = MagicMock(
            scalar_one_or_none=MagicMock(return_value=mock_model)
        )
        
        status = await learner.get_model_status(driver_id)
        
        assert status is not None
        assert status["model_version"] == 3
        assert status["training_samples"] == 50
        assert status["current_mse"] == 8.5
        assert status["r2_score"] == 0.85
        assert status["active"] is True


class TestXGBoostIntegration:
    """Tests for XGBoost integration (requires xgboost installed)."""
    
    @pytest.fixture
    def mock_db(self):
        mock = AsyncMock()
        mock.execute = AsyncMock()
        mock.add = MagicMock()
        return mock
    
    def test_xgboost_availability_check(self, mock_db):
        """Test that XGBoost availability is properly detected."""
        learner = DriverEffortLearner(mock_db)
        
        # Should be True if xgboost is installed, False otherwise
        assert isinstance(learner._xgb_available, bool)
    
    @pytest.mark.skipif(
        not DriverEffortLearner(AsyncMock())._xgb_available,
        reason="XGBoost not installed"
    )
    def test_xgboost_can_train(self):
        """Test that XGBoost model can be trained manually."""
        import xgboost as xgb
        import pandas as pd
        
        # Create synthetic data
        X = pd.DataFrame({
            "num_packages": [10, 20, 30, 15, 25],
            "total_weight_kg": [30, 50, 80, 40, 60],
            "num_stops": [5, 10, 15, 7, 12],
        })
        y = np.array([40, 60, 85, 50, 70])
        
        model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == 5
        assert all(isinstance(p, (float, np.floating)) for p in predictions)


class TestModelSerialization:
    """Tests for model serialization."""
    
    @pytest.mark.skipif(
        not DriverEffortLearner(AsyncMock())._xgb_available,
        reason="XGBoost not installed"
    )
    def test_model_pickle_roundtrip(self):
        """Test that model can be pickled and unpickled."""
        import pickle
        import xgboost as xgb
        import pandas as pd
        
        # Train a small model
        X = pd.DataFrame({
            "num_packages": [10, 20, 30],
            "total_weight_kg": [30, 50, 80],
        })
        y = np.array([40, 60, 85])
        
        model = xgb.XGBRegressor(n_estimators=5)
        model.fit(X, y)
        
        # Pickle and unpickle
        pickled = pickle.dumps(model)
        loaded = pickle.loads(pickled)
        
        # Should produce same predictions
        original_preds = model.predict(X)
        loaded_preds = loaded.predict(X)
        
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)
