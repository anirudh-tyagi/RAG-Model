"""Dense and sparse text embeddings.

Runs BGE and BM25 through fastembed's ONNX runtime rather than
sentence-transformers, which keeps torch out of the dependency tree entirely
(the old ``Emmbed.py`` pulled in ~2.5GB of torch to embed text on CPU) and
makes CPU inference noticeably faster.

BGE is an asymmetric model: passages are embedded bare, queries get the
"Represent this sentence…" instruction prefix. fastembed's ``passage_embed`` /
``query_embed`` apply the right convention per model, so we always go through
them instead of the generic ``embed``.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from qdrant_client import models as qm

from rag.config import get_settings
from rag.logging import get_logger

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding, TextEmbedding

log = get_logger(__name__)

_PROBE = "dimension probe"


class Embedder:
    """Lazily loaded dense + sparse encoders, safe to share across tasks."""

    def __init__(self, dense_model: str, sparse_model: str) -> None:
        self._dense_model_name = dense_model
        self._sparse_model_name = sparse_model
        self._dense: TextEmbedding | None = None
        self._sparse: SparseTextEmbedding | None = None
        self._dim: int | None = None
        # ONNX sessions are built on first use; guard so concurrent requests
        # during a cold start don't each build their own copy.
        self._lock = threading.Lock()

    # --- model loading ---------------------------------------------------

    def _load(self) -> None:
        if self._dense is not None and self._sparse is not None:
            return
        with self._lock:
            if self._dense is None:
                from fastembed import TextEmbedding

                log.info("loading_dense_model", model=self._dense_model_name)
                self._dense = TextEmbedding(model_name=self._dense_model_name)
            if self._sparse is None:
                from fastembed import SparseTextEmbedding

                log.info("loading_sparse_model", model=self._sparse_model_name)
                self._sparse = SparseTextEmbedding(model_name=self._sparse_model_name)

    async def warm_up(self) -> None:
        """Load models off the event loop so the first request isn't slow."""
        await asyncio.to_thread(self._load)
        await asyncio.to_thread(self._probe_dim)
        log.info("embedder_ready", dense_dim=self._dim)

    # --- dimensionality --------------------------------------------------

    def _probe_dim(self) -> int:
        """Determine the dense vector width by embedding one throwaway string.

        More robust than reading fastembed's model registry, whose shape has
        changed between releases.
        """
        if self._dim is None:
            self._load()
            assert self._dense is not None
            vector = next(iter(self._dense.passage_embed([_PROBE])))
            self._dim = len(vector)
        return self._dim

    async def dim(self) -> int:
        return await asyncio.to_thread(self._probe_dim)

    # --- encoding --------------------------------------------------------

    def _encode_passages(self, texts: list[str]) -> tuple[list[list[float]], list[qm.SparseVector]]:
        self._load()
        assert self._dense is not None and self._sparse is not None
        dense = [v.tolist() for v in self._dense.passage_embed(texts)]
        sparse = [
            qm.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
            for s in self._sparse.embed(texts)
        ]
        return dense, sparse

    async def encode_passages(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[qm.SparseVector]]:
        """Embed document chunks for indexing."""
        if not texts:
            return [], []
        return await asyncio.to_thread(self._encode_passages, texts)

    def _encode_query(self, text: str) -> tuple[list[float], qm.SparseVector]:
        self._load()
        assert self._dense is not None and self._sparse is not None
        dense = next(iter(self._dense.query_embed([text]))).tolist()
        raw = next(iter(self._sparse.query_embed([text])))
        sparse = qm.SparseVector(indices=raw.indices.tolist(), values=raw.values.tolist())
        return dense, sparse

    async def encode_query(self, text: str) -> tuple[list[float], qm.SparseVector]:
        """Embed a search query (with the model's query-side prefix applied)."""
        return await asyncio.to_thread(self._encode_query, text)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        _embedder = Embedder(settings.dense_model, settings.sparse_model)
    return _embedder
