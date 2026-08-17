"""Raw retrieval endpoint.

Useful on its own, and essential for tuning: it shows exactly which passages the
retriever surfaces and what reranking did to their order, with no LLM in the
way. The eval harness drives the same code path.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from rag.api.deps import SearcherDep
from rag.schemas import SearchResponse

router = APIRouter(tags=["search"])


class SearchBody(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    doc_ids: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=50)


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchBody, searcher: SearcherDep) -> SearchResponse:
    return await searcher.search_response(
        body.query, doc_ids=body.doc_ids or None, top_k=body.top_k
    )


@router.get("/search", response_model=SearchResponse)
async def search_get(
    searcher: SearcherDep,
    q: str = Query(min_length=1, max_length=2000),
    top_k: int | None = Query(default=None, ge=1, le=50),
) -> SearchResponse:
    """Convenience form for poking at retrieval from a browser or curl."""
    return await searcher.search_response(q, top_k=top_k)
