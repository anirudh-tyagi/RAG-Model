"""Cross-encoder reranking.

Fusing dense and sparse results gets the right passage into the candidate set;
a cross-encoder is what gets it to the top. It scores each (query, passage)
pair jointly instead of comparing two independently-computed vectors, which is
slower but markedly more accurate — so we only run it over the ~40 fused
candidates, then keep the best handful.

This stage did not exist before: the old chain handed the LLM whatever the
top-5 cosine neighbours happened to be.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from rag.config import get_settings
from rag.logging import get_logger

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

log = get_logger(__name__)


class Reranker:
    """Lazily loaded ONNX cross-encoder."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: TextCrossEncoder | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                log.info("loading_reranker", model=self._model_name)
                self._model = TextCrossEncoder(model_name=self._model_name)

    async def warm_up(self) -> None:
        await asyncio.to_thread(self._load)
        log.info("reranker_ready", model=self._model_name)

    def _score(self, query: str, documents: list[str]) -> list[float]:
        self._load()
        assert self._model is not None
        return [float(s) for s in self._model.rerank(query, documents)]

    async def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per document. Higher is better; scale is model-specific."""
        if not documents:
            return []
        return await asyncio.to_thread(self._score, query, documents)


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker(get_settings().reranker_model)
    return _reranker
