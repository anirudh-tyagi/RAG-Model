"""Retrieval and answer metrics.

Retrieval is scored against gold *pages*; answers are scored on keyword
coverage and on whether their inline citations point at passages that were
actually retrieved. A citation like ``[7]`` when only six passages were supplied
is a fabricated reference, and catching that needs no LLM judge at all.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from rag.schemas import SearchResult

# Accepts full-width brackets as well as ASCII: gpt-oss models routinely emit
# 【1】 instead of [1], and scoring those as "no citation" would be wrong.
CITATION = re.compile(r"[\[【](\d{1,2})[\]】]")


@dataclass(slots=True)
class RetrievalMetrics:
    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit: bool
    retrieved: int

    @property
    def as_dict(self) -> dict[str, float | bool | int]:
        return {
            "recall": self.recall,
            "precision": self.precision,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "hit": self.hit,
            "retrieved": self.retrieved,
        }


@dataclass(slots=True)
class AnswerMetrics:
    keyword_coverage: float
    citation_count: int
    #: Fraction of inline citations that reference a passage that exists.
    citation_validity: float
    abstained: bool
    groundedness: float | None = None

    @property
    def as_dict(self) -> dict[str, float | bool | int | None]:
        return {
            "keyword_coverage": self.keyword_coverage,
            "citation_count": self.citation_count,
            "citation_validity": self.citation_validity,
            "abstained": self.abstained,
            "groundedness": self.groundedness,
        }


def score_retrieval(results: list[SearchResult], relevant_pages: list[int]) -> RetrievalMetrics:
    """Binary-relevance retrieval metrics against a set of gold pages."""
    retrieved_pages = [r.chunk.page for r in results]
    gold = set(relevant_pages)

    if not gold:
        # No gold labels: nothing to measure, but still report volume.
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, False, len(results))

    relevance = [1 if page in gold else 0 for page in retrieved_pages]
    found = {page for page in retrieved_pages if page in gold}

    recall = len(found) / len(gold)
    precision = (sum(relevance) / len(relevance)) if relevance else 0.0

    mrr = 0.0
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            mrr = 1.0 / rank
            break

    return RetrievalMetrics(
        recall=recall,
        precision=precision,
        mrr=mrr,
        ndcg=_ndcg(relevance, len(gold)),
        hit=bool(found),
        retrieved=len(results),
    )


def _ndcg(relevance: list[int], total_relevant: int) -> float:
    """nDCG with binary gains, normalised by the best achievable ordering."""
    if not relevance or total_relevant == 0:
        return 0.0
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance, start=1))
    ideal_hits = min(total_relevant, len(relevance))
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def score_answer(
    answer: str,
    expected_keywords: list[str],
    source_count: int,
    groundedness: float | None = None,
) -> AnswerMetrics:
    lowered = answer.lower()
    coverage = (
        sum(1 for kw in expected_keywords if kw.lower() in lowered) / len(expected_keywords)
        if expected_keywords
        else 0.0
    )

    cited = [int(n) for n in CITATION.findall(answer)]
    valid = [n for n in cited if 1 <= n <= source_count]
    validity = (len(valid) / len(cited)) if cited else 0.0

    return AnswerMetrics(
        keyword_coverage=coverage,
        citation_count=len(cited),
        citation_validity=validity,
        abstained=_looks_like_abstention(lowered),
        groundedness=groundedness,
    )


ABSTENTION_MARKERS = (
    "couldn't find",
    "could not find",
    "don't know",
    "do not know",
    "not in the",
    "no relevant",
    "does not contain",
)


def _looks_like_abstention(lowered_answer: str) -> bool:
    """Whether the model declined to answer.

    Tracked separately because an abstention is the *correct* behaviour when
    retrieval misses — averaging it in with wrong answers would hide that.
    """
    return any(marker in lowered_answer for marker in ABSTENTION_MARKERS)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
