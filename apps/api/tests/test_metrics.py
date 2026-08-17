"""Eval metric arithmetic."""

from __future__ import annotations

import math

from rag.eval.metrics import score_answer, score_retrieval
from rag.schemas import Chunk, SearchResult


def hit(page: int) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            doc_id="d1",
            doc_title="Doc",
            filename="d.pdf",
            chunk_index=0,
            text="text",
            page=page,
        ),
        score=1.0,
    )


def test_perfect_retrieval() -> None:
    metrics = score_retrieval([hit(4), hit(5)], relevant_pages=[4, 5])

    assert metrics.recall == 1.0
    assert metrics.precision == 1.0
    assert metrics.mrr == 1.0
    assert metrics.hit is True


def test_partial_recall() -> None:
    metrics = score_retrieval([hit(4), hit(9)], relevant_pages=[4, 5])

    assert metrics.recall == 0.5
    assert metrics.precision == 0.5


def test_mrr_uses_the_first_relevant_rank() -> None:
    metrics = score_retrieval([hit(9), hit(8), hit(4)], relevant_pages=[4])

    assert metrics.mrr == 1 / 3


def test_complete_miss_scores_zero() -> None:
    metrics = score_retrieval([hit(1), hit(2)], relevant_pages=[7])

    assert metrics.recall == 0.0
    assert metrics.mrr == 0.0
    assert metrics.ndcg == 0.0
    assert metrics.hit is False


def test_ndcg_rewards_putting_the_relevant_hit_first() -> None:
    early = score_retrieval([hit(4), hit(9), hit(8)], relevant_pages=[4])
    late = score_retrieval([hit(9), hit(8), hit(4)], relevant_pages=[4])

    assert early.ndcg > late.ndcg
    assert math.isclose(early.ndcg, 1.0)


def test_unlabelled_case_reports_volume_without_claiming_a_score() -> None:
    metrics = score_retrieval([hit(1), hit(2)], relevant_pages=[])

    assert metrics.retrieved == 2
    assert metrics.recall == 0.0
    assert metrics.hit is False


def test_empty_retrieval() -> None:
    metrics = score_retrieval([], relevant_pages=[3])

    assert metrics.retrieved == 0
    assert metrics.recall == 0.0


def test_keyword_coverage_is_case_insensitive() -> None:
    metrics = score_answer("Revenue rose to 5.1B", ["revenue", "5.1B"], source_count=2)

    assert metrics.keyword_coverage == 1.0


def test_missing_keyword_lowers_coverage() -> None:
    metrics = score_answer("Revenue rose", ["revenue", "5.1B"], source_count=2)

    assert metrics.keyword_coverage == 0.5


def test_citation_validity_flags_a_fabricated_reference() -> None:
    # Only two passages were supplied, so [7] cannot be real.
    metrics = score_answer("Supported [1] and invented [7].", [], source_count=2)

    assert metrics.citation_count == 2
    assert metrics.citation_validity == 0.5


def test_all_citations_valid() -> None:
    metrics = score_answer("Both [1] and [2] agree.", [], source_count=2)

    assert metrics.citation_validity == 1.0


def test_full_width_brackets_count_as_citations() -> None:
    """gpt-oss models emit 【1】 rather than [1]; scoring that as uncited is wrong."""
    metrics = score_answer("Revenue was $5.1B【1】 and margin 18.3%【2】.", [], source_count=2)

    assert metrics.citation_count == 2
    assert metrics.citation_validity == 1.0


def test_mixed_bracket_styles_are_both_counted() -> None:
    metrics = score_answer("First [1] and second 【2】.", [], source_count=2)

    assert metrics.citation_count == 2


def test_full_width_fabricated_citation_still_caught() -> None:
    metrics = score_answer("Real【1】 and invented【9】.", [], source_count=2)

    assert metrics.citation_validity == 0.5


def test_uncited_answer_reports_zero_citations() -> None:
    metrics = score_answer("No citations at all.", [], source_count=3)

    assert metrics.citation_count == 0
    assert metrics.citation_validity == 0.0


def test_abstention_is_detected() -> None:
    metrics = score_answer("I couldn't find that in the documents.", [], source_count=3)

    assert metrics.abstained is True


def test_normal_answer_is_not_an_abstention() -> None:
    metrics = score_answer("Revenue was 5.1B [1].", [], source_count=1)

    assert metrics.abstained is False
