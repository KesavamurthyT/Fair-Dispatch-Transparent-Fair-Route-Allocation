"""
Redis-backed artifact store for large intermediate data.

Stores large arrays (effort matrices, distance matrices) in Redis with TTL,
returning lightweight string keys for LangGraph state references.

Serialization uses numpy's safe binary format (np.save/np.load) —
NO pickle is used anywhere in this module.

Usage:
    from app.utils.artifact_store import artifact_store

    key = artifact_store.put_matrix(effort_matrix, ttl_seconds=3600)
    matrix = artifact_store.get_matrix(key)
    artifact_store.delete_keys(key)
"""

import io
import logging
from typing import Optional
from uuid import uuid4

import numpy as np
import redis as redis_lib

from app.config import get_settings

logger = logging.getLogger(__name__)


class ArtifactStore:
    """
    High-throughput Redis-backed store for large numpy arrays.

    Design:
    - Matrices are serialized using numpy's native binary format (np.save).
    - Keys are namespaced with "artifact:" prefix and UUID4 suffixes.
    - All stored artifacts have a TTL (default 12 hours) for automatic cleanup.
    - Manual cleanup via delete_keys() after job completion.

    Why not pickle:
    - numpy's binary format (.npy) is safe and well-defined.
    - pickle is a security risk (RCE via crafted blobs).

    Why not scipy.sparse:
    - Most effort matrices are dense in practice (every driver can serve every route).
    - For truly sparse cases, callers can convert to dense before storing,
      or use put_sparse / get_sparse (future extension).
    """

    DEFAULT_TTL = 12 * 3600  # 12 hours

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize with Redis connection."""
        url = redis_url or get_settings().redis_url
        self._redis = redis_lib.Redis.from_url(url, decode_responses=False)

    def put_matrix(
        self,
        matrix: np.ndarray,
        ttl_seconds: int = DEFAULT_TTL,
        key_prefix: str = "effort",
    ) -> str:
        """
        Store a numpy array in Redis.

        Args:
            matrix: The numpy array to store.
            ttl_seconds: Time-to-live in seconds (default 12 hours).
            key_prefix: Prefix for the key (e.g., "effort", "distance").

        Returns:
            Redis key string that can be used with get_matrix().
        """
        key = f"artifact:{key_prefix}:{uuid4().hex}"

        # Serialize to numpy's safe binary format
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        buffer.seek(0)
        data = buffer.read()

        self._redis.set(key, data, ex=ttl_seconds)

        logger.debug(
            f"Stored matrix {key}: shape={matrix.shape}, "
            f"size={len(data)} bytes, TTL={ttl_seconds}s"
        )
        return key

    def get_matrix(self, key: str) -> np.ndarray:
        """
        Retrieve a numpy array from Redis.

        Args:
            key: Redis key returned by put_matrix().

        Returns:
            The numpy array.

        Raises:
            KeyError: If the key does not exist (expired or deleted).
        """
        data = self._redis.get(key)
        if data is None:
            raise KeyError(f"Artifact not found: {key} (expired or deleted)")

        buffer = io.BytesIO(data)
        buffer.seek(0)
        matrix = np.load(buffer, allow_pickle=False)

        logger.debug(f"Retrieved matrix {key}: shape={matrix.shape}")
        return matrix

    def delete_keys(self, *keys: str) -> int:
        """
        Delete one or more artifact keys from Redis.

        Args:
            *keys: Redis keys to delete.

        Returns:
            Number of keys that were actually deleted.
        """
        if not keys:
            return 0

        deleted = self._redis.delete(*keys)
        logger.debug(f"Deleted {deleted} artifact keys")
        return deleted

    def exists(self, key: str) -> bool:
        """Check if an artifact key exists in Redis."""
        return bool(self._redis.exists(key))

    def ttl(self, key: str) -> int:
        """Get the remaining TTL of an artifact key in seconds."""
        return self._redis.ttl(key)

    def cleanup_prefix(self, prefix: str = "artifact:") -> int:
        """
        Delete all keys with a given prefix. Use with caution.
        Primarily for testing/development cleanup.
        """
        keys = list(self._redis.scan_iter(match=f"{prefix}*", count=100))
        if keys:
            return self._redis.delete(*keys)
        return 0


# Lazy singleton instance
_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Get or create the global artifact store singleton."""
    global _store
    if _store is None:
        _store = ArtifactStore()
    return _store


# Convenience alias
artifact_store = None


def _init_store():
    """Initialize the module-level store. Called lazily on first use."""
    global artifact_store
    if artifact_store is None:
        try:
            artifact_store = get_artifact_store()
        except Exception:
            logger.warning("Redis not available — artifact_store disabled")
            artifact_store = None
