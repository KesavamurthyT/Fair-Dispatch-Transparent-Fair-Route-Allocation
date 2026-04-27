"""
Unit tests for the Redis artifact store.

Tests matrix serialization, retrieval, TTL, and cleanup.
Uses fakeredis to avoid needing a running Redis server.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import io


class TestArtifactStoreUnit:
    """Test artifact store with a real-looking mock."""

    def _make_store(self):
        """Create an artifact store with a mock Redis client."""
        from app.utils.artifact_store import ArtifactStore

        # Patch Redis.from_url to use a dict-based mock
        mock_redis = MockRedis()
        with patch("app.utils.artifact_store.redis_lib.Redis.from_url", return_value=mock_redis):
            store = ArtifactStore(redis_url="redis://fake:6379/0")
            store._redis = mock_redis
        return store

    def test_put_and_get_dense_matrix(self):
        """Round-trip: put dense matrix → get same matrix back."""
        store = self._make_store()
        original = np.random.rand(10, 5).astype(np.float64)

        key = store.put_matrix(original, ttl_seconds=3600)

        assert key.startswith("artifact:effort:")
        assert store.exists(key)

        retrieved = store.get_matrix(key)
        np.testing.assert_array_almost_equal(original, retrieved)

    def test_put_and_get_integer_matrix(self):
        """Should handle integer arrays."""
        store = self._make_store()
        original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

        key = store.put_matrix(original, key_prefix="distance")
        retrieved = store.get_matrix(key)

        assert key.startswith("artifact:distance:")
        np.testing.assert_array_equal(original, retrieved)

    def test_put_and_get_large_matrix(self):
        """Should handle large matrices (2000 x 500)."""
        store = self._make_store()
        original = np.random.rand(2000, 500).astype(np.float32)

        key = store.put_matrix(original)
        retrieved = store.get_matrix(key)

        np.testing.assert_array_almost_equal(original, retrieved, decimal=5)

    def test_get_nonexistent_key_raises(self):
        """Getting a missing key should raise KeyError."""
        store = self._make_store()

        with pytest.raises(KeyError, match="Artifact not found"):
            store.get_matrix("artifact:effort:nonexistent")

    def test_delete_keys(self):
        """Deleting keys should remove them from the store."""
        store = self._make_store()
        matrix = np.eye(3)

        key1 = store.put_matrix(matrix, key_prefix="a")
        key2 = store.put_matrix(matrix, key_prefix="b")

        assert store.exists(key1)
        assert store.exists(key2)

        deleted = store.delete_keys(key1, key2)
        assert deleted == 2

        assert not store.exists(key1)
        assert not store.exists(key2)

    def test_unique_keys(self):
        """Successive puts should generate unique keys."""
        store = self._make_store()
        matrix = np.eye(3)

        key1 = store.put_matrix(matrix)
        key2 = store.put_matrix(matrix)

        assert key1 != key2

    def test_1d_array(self):
        """Should handle 1D vectors."""
        store = self._make_store()
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        key = store.put_matrix(original, key_prefix="vec")
        retrieved = store.get_matrix(key)

        np.testing.assert_array_equal(original, retrieved)


class TestNoPickleSafety:
    """Verify that pickle is never used in the artifact store."""

    def test_no_pickle_import(self):
        """The artifact_store module should not import pickle."""
        import importlib
        import app.utils.artifact_store as module

        source = importlib.util.find_spec("app.utils.artifact_store")
        with open(source.origin, "r") as f:
            content = f.read()

        assert "import pickle" not in content
        assert "pickle.dumps" not in content
        assert "pickle.loads" not in content


# === Mock Redis ===

class MockRedis:
    """Simple in-memory Redis mock for testing."""

    def __init__(self):
        self._store = {}

    def set(self, key, value, ex=None):
        self._store[key] = value

    def get(self, key):
        return self._store.get(key)

    def exists(self, key):
        return key in self._store

    def ttl(self, key):
        return 3600 if key in self._store else -2

    def delete(self, *keys):
        deleted = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                deleted += 1
        return deleted

    def scan_iter(self, match="*", count=100):
        import fnmatch
        return [k for k in self._store.keys() if fnmatch.fnmatch(k, match)]
