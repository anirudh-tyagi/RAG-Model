"""Offline evaluation of retrieval and answer quality."""

from rag.eval.dataset import EvalCase, load_cases
from rag.eval.metrics import AnswerMetrics, RetrievalMetrics, score_answer, score_retrieval
from rag.eval.runner import CaseResult, EvalReport, run_eval

__all__ = [
    "AnswerMetrics",
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "RetrievalMetrics",
    "load_cases",
    "run_eval",
    "score_answer",
    "score_retrieval",
]
