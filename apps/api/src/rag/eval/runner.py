"""Eval runner.

Two modes:

* ``--retrieval-only`` scores retrieval alone. Fast, free, deterministic, and the
  right loop for tuning chunk size, the embedding model, candidate count or the
  reranker — none of which need an LLM in the loop.
* the full run also generates answers and scores them, optionally with an
  LLM-as-judge groundedness score.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from rag import obs
from rag.chat.llm import LlmClient, LlmError
from rag.chat.service import ChatService
from rag.config import Settings
from rag.eval.dataset import EvalCase
from rag.eval.metrics import (
    AnswerMetrics,
    RetrievalMetrics,
    mean,
    score_answer,
    score_retrieval,
)
from rag.logging import get_logger
from rag.retrieval.searcher import Searcher
from rag.schemas import SearchResult

log = get_logger(__name__)

JUDGE_SYSTEM = """\
You grade whether an answer is supported by the passages it was given.

Return only a JSON object: {"score": <0.0-1.0>, "reason": "<one short sentence>"}

score 1.0  every claim is directly supported by the passages
score 0.5  partly supported, or supported but with added unsupported detail
score 0.0  contradicted by the passages, or invented outright

An answer that correctly says the passages don't contain the information scores 1.0.\
"""


@dataclass(slots=True)
class CaseResult:
    case: EvalCase
    retrieval: RetrievalMetrics
    latency_ms: float
    answer: str | None = None
    answer_metrics: AnswerMetrics | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case.id,
            "question": self.case.question,
            "retrieval": self.retrieval.as_dict,
            "answer": self.answer,
            "answer_metrics": self.answer_metrics.as_dict if self.answer_metrics else None,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


@dataclass(slots=True)
class EvalReport:
    top_k: int
    results: list[CaseResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    # --- aggregates -------------------------------------------------------

    @property
    def recall(self) -> float:
        return mean([r.retrieval.recall for r in self._labelled])

    @property
    def mrr(self) -> float:
        return mean([r.retrieval.mrr for r in self._labelled])

    @property
    def ndcg(self) -> float:
        return mean([r.retrieval.ndcg for r in self._labelled])

    @property
    def hit_rate(self) -> float:
        labelled = self._labelled
        return (sum(1 for r in labelled if r.retrieval.hit) / len(labelled)) if labelled else 0.0

    @property
    def keyword_coverage(self) -> float:
        return mean([m.keyword_coverage for m in self._answer_metrics])

    @property
    def citation_validity(self) -> float:
        # Only answers that cited anything at all; averaging in uncited answers
        # as 0.0 would conflate "cited badly" with "didn't cite".
        cited = [m for m in self._answer_metrics if m.citation_count > 0]
        return mean([m.citation_validity for m in cited])

    @property
    def groundedness(self) -> float | None:
        scores = [m.groundedness for m in self._answer_metrics if m.groundedness is not None]
        return mean(scores) if scores else None

    @property
    def abstention_rate(self) -> float:
        metrics = self._answer_metrics
        if not metrics:
            return 0.0
        return sum(1 for m in metrics if m.abstained) / len(metrics)

    @property
    def errors(self) -> list[CaseResult]:
        return [r for r in self.results if r.error]

    @property
    def median_latency_ms(self) -> float:
        latencies = sorted(r.latency_ms for r in self.results)
        if not latencies:
            return 0.0
        return latencies[len(latencies) // 2]

    @property
    def _labelled(self) -> list[CaseResult]:
        """Only cases with gold pages contribute to retrieval averages."""
        return [r for r in self.results if r.case.relevant_pages]

    @property
    def _answer_metrics(self) -> list[AnswerMetrics]:
        """Metrics for cases that produced an answer, with the Nones filtered out."""
        return [r.answer_metrics for r in self.results if r.answer_metrics is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "config": self.config,
            "summary": {
                "cases": len(self.results),
                "labelled_cases": len(self._labelled),
                "recall": round(self.recall, 4),
                "mrr": round(self.mrr, 4),
                "ndcg": round(self.ndcg, 4),
                "hit_rate": round(self.hit_rate, 4),
                "keyword_coverage": round(self.keyword_coverage, 4),
                "citation_validity": round(self.citation_validity, 4),
                "groundedness": (
                    round(self.groundedness, 4) if self.groundedness is not None else None
                ),
                "abstention_rate": round(self.abstention_rate, 4),
                "median_latency_ms": round(self.median_latency_ms, 1),
                "errors": len(self.errors),
            },
            "cases": [r.to_dict() for r in self.results],
        }


async def run_eval(
    cases: list[EvalCase],
    searcher: Searcher,
    settings: Settings,
    chat_service: ChatService | None = None,
    llm: LlmClient | None = None,
    top_k: int | None = None,
    judge: bool = False,
    concurrency: int = 3,
) -> EvalReport:
    """Score every case. Pass ``chat_service=None`` for a retrieval-only run."""
    top_k = top_k or settings.retrieval_top_k
    semaphore = asyncio.Semaphore(concurrency)

    report = EvalReport(
        top_k=top_k,
        config={
            "dense_model": settings.dense_model,
            "sparse_model": settings.sparse_model,
            "reranker_model": settings.reranker_model if settings.rerank_enabled else None,
            "rerank_enabled": settings.rerank_enabled,
            "retrieval_candidates": settings.retrieval_candidates,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "llm_model": settings.llm_model if chat_service else None,
        },
    )

    async def run_case(case: EvalCase) -> CaseResult:
        async with semaphore:
            started = time.perf_counter()
            try:
                results = await searcher.search(
                    case.question, doc_ids=case.doc_ids or None, top_k=top_k
                )
            except Exception as exc:
                log.warning("eval_retrieval_failed", case=case.id, error=str(exc))
                return CaseResult(
                    case=case,
                    retrieval=score_retrieval([], case.relevant_pages),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=str(exc),
                )

            retrieval = score_retrieval(results, case.relevant_pages)
            if chat_service is None:
                return CaseResult(
                    case=case,
                    retrieval=retrieval,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            try:
                answer, answer_results = await chat_service.answer(
                    case.question, doc_ids=case.doc_ids or None
                )
            except LlmError as exc:
                return CaseResult(
                    case=case,
                    retrieval=retrieval,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=str(exc),
                )

            groundedness = None
            if judge and llm is not None:
                groundedness = await _judge(llm, case.question, answer, answer_results)

            return CaseResult(
                case=case,
                retrieval=retrieval,
                latency_ms=(time.perf_counter() - started) * 1000,
                answer=answer,
                answer_metrics=score_answer(
                    answer, case.expected_keywords, len(answer_results), groundedness
                ),
            )

    report.results = list(await asyncio.gather(*(run_case(case) for case in cases)))
    _publish_scores(report)
    return report


async def _judge(
    llm: LlmClient, question: str, answer: str, results: list[SearchResult]
) -> float | None:
    """LLM-as-judge groundedness, or ``None`` if the judge itself failed."""
    passages = "\n\n".join(
        f"[{n}] {r.chunk.text.strip()[:1200]}" for n, r in enumerate(results, start=1)
    )
    try:
        raw = await llm.complete(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"PASSAGES:\n{passages}\n\nQUESTION: {question}\n\nANSWER: {answer}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=200,
        )
    except LlmError as exc:
        log.warning("judge_failed", error=str(exc))
        return None

    return _parse_score(raw)


def _parse_score(raw: str) -> float | None:
    """Pull a 0-1 score out of the judge's reply, tolerating stray prose."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = float(json.loads(raw[start : end + 1])["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return min(1.0, max(0.0, value))


def _publish_scores(report: EvalReport) -> None:
    """Send aggregate scores to Langfuse when tracing is configured."""
    obs.score("eval_recall", report.recall)
    obs.score("eval_mrr", report.mrr)
    obs.score("eval_ndcg", report.ndcg)
    if report.groundedness is not None:
        obs.score("eval_groundedness", report.groundedness)
    obs.flush()
