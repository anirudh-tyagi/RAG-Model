"""The retrieval pipeline: encode → hybrid fuse → rerank → truncate."""

from __future__ import annotations

import time

from rag import obs
from rag.config import Settings, get_settings
from rag.logging import get_logger
from rag.retrieval.embeddings import Embedder, get_embedder
from rag.retrieval.rerank import Reranker, get_reranker
from rag.retrieval.store import VectorStore, get_store
from rag.schemas import SearchResponse, SearchResult, Source

log = get_logger(__name__)


class Searcher:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker

    @obs.observe("retrieval")
    async def search(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        candidates: int | None = None,
    ) -> list[SearchResult]:
        top_k = top_k or self._settings.retrieval_top_k
        candidates = candidates or self._settings.retrieval_candidates
        # Reranking is only useful if it has more to choose from than it returns.
        candidates = max(candidates, top_k)

        dense, sparse = await self._embedder.encode_query(query)
        fused = await self._store.hybrid_search(
            dense_query=dense,
            sparse_query=sparse,
            limit=candidates,
            doc_ids=doc_ids or None,
        )
        if not fused:
            return []

        if not self._settings.rerank_enabled:
            return fused[:top_k]

        scores = await self._reranker.score(query, [r.chunk.text for r in fused])
        for result, score in zip(fused, scores, strict=True):
            result.rerank_score = score
        fused.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)

        top = fused[:top_k]
        log.debug(
            "retrieval_complete",
            query_len=len(query),
            candidates=len(fused),
            returned=len(top),
            best=top[0].rerank_score if top else None,
        )
        return top

    async def search_response(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> SearchResponse:
        """Search with timing, for the debug/eval endpoint."""
        started = time.perf_counter()
        results = await self.search(query, doc_ids=doc_ids, top_k=top_k)
        return SearchResponse(
            query=query,
            results=results,
            took_ms=(time.perf_counter() - started) * 1000,
        )


def to_sources(results: list[SearchResult], excerpt_chars: int = 320) -> list[Source]:
    """Convert hits into user-facing citations, numbered from 1 to match the prompt."""
    sources: list[Source] = []
    for n, result in enumerate(results, start=1):
        text = result.chunk.text.strip()
        excerpt = text if len(text) <= excerpt_chars else text[:excerpt_chars].rstrip() + "…"
        sources.append(
            Source(
                n=n,
                chunk_id=result.chunk.id,
                doc_id=result.chunk.doc_id,
                doc_title=result.chunk.doc_title,
                page=result.chunk.page,
                heading=result.chunk.heading,
                excerpt=excerpt,
                score=result.rerank_score if result.rerank_score is not None else result.score,
            )
        )
    return sources


_searcher: Searcher | None = None


def get_searcher() -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher(get_settings(), get_store(), get_embedder(), get_reranker())
    return _searcher
