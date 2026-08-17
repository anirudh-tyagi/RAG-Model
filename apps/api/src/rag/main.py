"""FastAPI application factory.

Run with:  ``uv run uvicorn rag.main:app --reload``
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag import __version__, obs
from rag.api.routes import router
from rag.chat.llm import LlmError
from rag.config import get_settings
from rag.jobs.queue import close_queue
from rag.kv import close_redis
from rag.logging import configure_logging, get_logger
from rag.retrieval.embeddings import get_embedder
from rag.retrieval.rerank import get_reranker
from rag.retrieval.store import get_store, reset_store

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm models and check dependencies without ever aborting startup.

    The old ``load_models`` called ``exit()`` on any failure — a missing API key
    or an unreachable database killed the process before it could tell anyone
    why. Here every dependency failure is logged and surfaced through
    ``/api/health``, and the app still serves.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    embedder = get_embedder()
    warmups = [embedder.warm_up()]
    if settings.rerank_enabled:
        warmups.append(get_reranker().warm_up())

    results = await asyncio.gather(*warmups, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            log.error("model_warmup_failed", error=str(result))

    try:
        await get_store().ensure_collection(await embedder.dim())
    except Exception as exc:
        # Common in development when Qdrant starts a moment after the API.
        # Ingestion retries this, so it is not fatal.
        log.warning("collection_setup_deferred", error=str(exc))

    log.info(
        "api_ready",
        version=__version__,
        llm=settings.llm_model,
        llm_configured=settings.ollama_api_key is not None,
        dense_model=settings.dense_model,
        parser=settings.pdf_parser,
        rerank=settings.rerank_enabled,
        collection=settings.qdrant_collection,
    )
    try:
        yield
    finally:
        obs.flush()
        await close_queue()
        await reset_store()
        await close_redis()
        log.info("api_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Document RAG API",
        version=__version__,
        summary="Hybrid retrieval over your PDFs, with reranking and cited answers.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(LlmError)
    async def llm_error_handler(_: Request, exc: LlmError) -> JSONResponse:
        """Provider problems are the provider's fault, not a 500."""
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    app.include_router(router)
    obs.instrument_fastapi(app)
    return app


app = create_app()
