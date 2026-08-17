"""Ingestion worker.

Run with:  ``uv run arq rag.jobs.worker.WorkerSettings``
"""

from __future__ import annotations

from typing import Any, ClassVar

from arq.connections import RedisSettings

from rag import obs
from rag.config import get_settings
from rag.ingest.pipeline import get_pipeline
from rag.jobs.queue import redis_settings as _redis_settings
from rag.kv import close_redis
from rag.logging import configure_logging, get_logger
from rag.registry import get_registry
from rag.retrieval.embeddings import get_embedder
from rag.retrieval.store import reset_store

log = get_logger(__name__)


async def ingest_document(ctx: dict[str, Any], doc_id: str) -> str:
    """Run the full pipeline for one uploaded document."""
    document = await get_registry().get(doc_id)
    if document is None:
        log.warning("ingest_skipped_unknown_document", doc_id=doc_id)
        return "unknown"

    await get_pipeline().run(document)
    return document.stage.value


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    # Pay the model-loading cost once, at boot, instead of on the first upload.
    await get_embedder().warm_up()
    log.info("worker_ready", parser=settings.pdf_parser, collection=settings.qdrant_collection)


async def shutdown(ctx: dict[str, Any]) -> None:
    obs.flush()
    await reset_store()
    await close_redis()
    log.info("worker_stopped")


class WorkerSettings:
    functions: ClassVar[list[Any]] = [ingest_document]
    on_startup = startup
    on_shutdown = shutdown
    #: PDF parsing and embedding are CPU-bound, so a small number of
    #: simultaneous documents keeps latency predictable.
    max_jobs = 2
    job_timeout = 3600
    #: Ingestion is not idempotent-cheap; a failed document is reported to the
    #: user rather than silently retried in a loop.
    max_tries = 1
    keep_result = 3600
    # arq reads this as a value, not a callable.
    redis_settings: RedisSettings = _redis_settings()
