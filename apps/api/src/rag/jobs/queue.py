"""Enqueue side of the ingestion queue.

The old upload endpoint used ``BackgroundTasks``, which runs the work inside the
web process after the response is sent. A restart, a crash, or a container
rescheduling during a long PDF parse lost the job with no record that it had
ever been started. Jobs now live in Redis and are executed by a separate worker
process, so the API can restart freely and the job survives.
"""

from __future__ import annotations

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from rag.config import get_settings
from rag.logging import get_logger

log = get_logger(__name__)

INGEST_TASK = "ingest_document"

_pool: ArqRedis | None = None


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


async def get_queue() -> ArqRedis:
    global _pool
    if _pool is None:
        _pool = await create_pool(redis_settings())
    return _pool


async def enqueue_ingest(doc_id: str) -> str | None:
    """Queue a document for ingestion. Returns the arq job id."""
    queue = await get_queue()
    job = await queue.enqueue_job(INGEST_TASK, doc_id, _job_id=f"ingest:{doc_id}")
    if job is None:
        # arq de-duplicates by job id: the same document is already queued.
        log.info("ingest_already_queued", doc_id=doc_id)
        return None
    log.info("ingest_enqueued", doc_id=doc_id, job_id=job.job_id)
    return job.job_id


async def queue_healthy() -> bool:
    try:
        queue = await get_queue()
        await queue.ping()
        return True
    except Exception as exc:
        log.warning("queue_unreachable", error=str(exc))
        return False


async def close_queue() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
