"""Shared Redis connection.

Redis is already required for the job queue, so it also holds the two pieces of
state the old app had nowhere to put: document ingestion status (previously only
visible as ``print()`` output in the server log) and conversation history
(previously kept in a browser-side array that the backend never saw).
"""

from __future__ import annotations

from redis.asyncio import ConnectionPool, Redis

from rag.config import get_settings

NAMESPACE = "rag"

_pool: ConnectionPool | None = None


def get_redis() -> Redis:
    """A Redis client over a shared pool. Cheap to call per request."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool.from_url(
            get_settings().redis_url,
            decode_responses=True,
            max_connections=32,
        )
    return Redis(connection_pool=_pool)


async def close_redis() -> None:
    global _pool
    if _pool is not None:
        await _pool.disconnect()
        _pool = None


def key(*parts: str) -> str:
    return ":".join((NAMESPACE, *parts))
