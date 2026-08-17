"""Health and readiness.

Each dependency is reported separately, so "the app is up but Qdrant isn't" is
distinguishable from a total outage. The old code called ``exit()`` at startup
on any dependency failure, leaving nothing running to ask.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from pydantic import BaseModel

from rag import __version__
from rag.api.deps import LlmDep, SettingsDep, StoreDep
from rag.jobs.queue import queue_healthy

router = APIRouter(tags=["health"])


class HealthOut(BaseModel):
    status: str
    version: str
    qdrant: bool
    queue: bool
    llm_configured: bool
    dense_model: str
    llm_model: str
    parser: str
    rerank_enabled: bool


@router.get("/health", response_model=HealthOut)
async def health(settings: SettingsDep, store: StoreDep, llm: LlmDep) -> HealthOut:
    qdrant_ok, queue_ok = await asyncio.gather(store.health(), queue_healthy())
    llm_configured = llm.configured

    return HealthOut(
        status="ok" if (qdrant_ok and queue_ok and llm_configured) else "degraded",
        version=__version__,
        qdrant=qdrant_ok,
        queue=queue_ok,
        llm_configured=llm_configured,
        dense_model=settings.dense_model,
        llm_model=settings.llm_model,
        parser=settings.pdf_parser,
        rerank_enabled=settings.rerank_enabled,
    )
