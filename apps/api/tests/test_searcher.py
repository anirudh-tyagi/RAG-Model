"""The retrieval pipeline: fusion order in, reranked order out."""

from __future__ import annotations

from typing import Any

from rag.config import Settings
from rag.retrieval.searcher import Searcher, to_sources
from rag.schemas import Chunk, SearchResult


def chunk(text: str, doc_id: str = "d1", **kwargs: Any) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        doc_title="Doc",
        filename="doc.pdf",
        chunk_index=0,
        text=text,
        **kwargs,
    )


def build(settings: Settings, store: Any, embedder: Any, reranker: Any) -> Searcher:
    return Searcher(settings, store, embedder, reranker)


async def test_reranker_promotes_the_best_match(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any
) -> None:
    # Deliberately ordered worst-first, as fusion might return them.
    fake_store.chunks = [
        chunk("completely unrelated prose about gardening"),
        chunk("partially about revenue only"),
        chunk("revenue rose sharply in the fourth quarter"),
    ]
    searcher = build(settings, fake_store, fake_embedder, fake_reranker)

    results = await searcher.search("revenue rose sharply", top_k=3)

    assert results[0].chunk.text.startswith("revenue rose sharply")
    assert results[0].rerank_score is not None
    # Scores are monotonically decreasing after the sort.
    scores = [r.rerank_score or 0.0 for r in results]
    assert scores == sorted(scores, reverse=True)


async def test_rerank_disabled_preserves_fusion_order(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any
) -> None:
    fake_store.chunks = [chunk("first"), chunk("second"), chunk("third")]
    no_rerank = settings.model_copy(update={"rerank_enabled": False})
    searcher = build(no_rerank, fake_store, fake_embedder, fake_reranker)

    results = await searcher.search("second", top_k=3)

    assert [r.chunk.text for r in results] == ["first", "second", "third"]
    assert all(r.rerank_score is None for r in results)
    assert fake_reranker.calls == []


async def test_top_k_truncates_after_reranking(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any
) -> None:
    fake_store.chunks = [chunk(f"passage {n} about revenue") for n in range(10)]
    searcher = build(settings, fake_store, fake_embedder, fake_reranker)

    results = await searcher.search("revenue", top_k=3)

    assert len(results) == 3
    # The reranker still saw the whole candidate set, not just the top 3.
    assert fake_reranker.calls[0][1] == 10


async def test_candidates_are_never_fewer_than_top_k(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any
) -> None:
    fake_store.chunks = [chunk(f"passage {n}") for n in range(20)]
    narrow = settings.model_copy(update={"retrieval_candidates": 2})
    searcher = build(narrow, fake_store, fake_embedder, fake_reranker)

    results = await searcher.search("passage", top_k=8)

    assert len(results) == 8


async def test_empty_index_returns_no_results(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any
) -> None:
    searcher = build(settings, fake_store, fake_embedder, fake_reranker)

    assert await searcher.search("anything") == []


async def test_doc_ids_scope_the_search(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any
) -> None:
    fake_store.chunks = [
        chunk("revenue in report A", doc_id="a"),
        chunk("revenue in report B", doc_id="b"),
    ]
    searcher = build(settings, fake_store, fake_embedder, fake_reranker)

    results = await searcher.search("revenue", doc_ids=["b"])

    assert [r.chunk.doc_id for r in results] == ["b"]


async def test_search_response_reports_timing(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any
) -> None:
    fake_store.chunks = [chunk("revenue")]
    searcher = build(settings, fake_store, fake_embedder, fake_reranker)

    response = await searcher.search_response("revenue")

    assert response.query == "revenue"
    assert response.took_ms >= 0
    assert len(response.results) == 1


# --- citation conversion ------------------------------------------------------


def test_sources_are_numbered_from_one() -> None:
    results = [
        SearchResult(chunk=chunk("first"), score=0.9),
        SearchResult(chunk=chunk("second"), score=0.8),
    ]

    sources = to_sources(results)

    assert [s.n for s in sources] == [1, 2]


def test_source_excerpt_is_truncated_with_an_ellipsis() -> None:
    results = [SearchResult(chunk=chunk("x" * 500), score=0.9)]

    sources = to_sources(results, excerpt_chars=50)

    assert len(sources[0].excerpt) <= 51
    assert sources[0].excerpt.endswith("…")


def test_short_excerpt_is_left_intact() -> None:
    results = [SearchResult(chunk=chunk("brief passage"), score=0.9)]

    assert to_sources(results)[0].excerpt == "brief passage"


def test_source_score_prefers_the_rerank_score() -> None:
    results = [SearchResult(chunk=chunk("text"), score=0.1, rerank_score=0.95)]

    assert to_sources(results)[0].score == 0.95


def test_source_score_falls_back_to_the_fusion_score() -> None:
    results = [SearchResult(chunk=chunk("text"), score=0.42)]

    assert to_sources(results)[0].score == 0.42


def test_source_carries_page_and_heading_for_the_citation_ui() -> None:
    results = [SearchResult(chunk=chunk("text", page=12, heading="Revenue"), score=0.5)]

    source = to_sources(results)[0]
    assert source.page == 12
    assert source.heading == "Revenue"
