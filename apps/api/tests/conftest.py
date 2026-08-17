"""Shared fixtures.

The whole suite runs offline: Redis is faked in-process, the vector store,
embedder, reranker and LLM are substituted, and no model weights are ever
downloaded. That is deliberate — tests that need Qdrant running and a 1.3GB
model download are tests nobody runs.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

import fakeredis
import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient
from qdrant_client import models as qm

from rag import kv
from rag.config import Settings, get_settings
from rag.main import create_app
from rag.schemas import Chunk, SearchResult

# Modules holding a lazily built singleton that must not leak between tests.
_SINGLETON_GLOBALS = [
    ("rag.retrieval.store", "_store"),
    ("rag.retrieval.embeddings", "_embedder"),
    ("rag.retrieval.rerank", "_reranker"),
    ("rag.retrieval.searcher", "_searcher"),
    ("rag.chat.llm", "_llm"),
    ("rag.chat.memory", "_store"),
    ("rag.chat.service", "_service"),
    ("rag.registry", "_registry"),
    ("rag.stt", "_transcriber"),
    ("rag.jobs.queue", "_pool"),
    ("rag.kv", "_pool"),
]


def _reset_singletons() -> None:
    import importlib

    for module_name, attribute in _SINGLETON_GLOBALS:
        setattr(importlib.import_module(module_name), attribute, None)


@pytest.fixture(autouse=True)
def settings(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Isolated settings, pointed at a temp data directory."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    get_settings.cache_clear()
    _reset_singletons()
    resolved = get_settings()
    resolved.ensure_dirs()
    yield resolved
    get_settings.cache_clear()
    _reset_singletons()


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> Callable[[], Any]:
    """Route every ``kv.get_redis()`` call at one in-process fake server."""
    server = fakeredis.FakeServer()

    def factory() -> Any:
        return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)

    monkeypatch.setattr(kv, "get_redis", factory)
    return factory


# --- test doubles -------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().replace("\n", " ").split() if t}


def _overlap(query: str, text: str) -> float:
    """Crude lexical similarity, enough to make ordering assertions meaningful."""
    q, t = _tokens(query), _tokens(text)
    return len(q & t) / len(q) if q else 0.0


class FakeEmbedder:
    """Deterministic hash embeddings. Same interface as the real Embedder."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim
        self.warmed = False

    async def warm_up(self) -> None:
        self.warmed = True

    async def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in _tokens(text):
            slot = int(hashlib.sha1(token.encode()).hexdigest(), 16) % self._dim
            vector[slot] += 1.0
        return vector

    async def encode_passages(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[qm.SparseVector]]:
        dense = [self._vector(t) for t in texts]
        sparse = [qm.SparseVector(indices=[0], values=[1.0]) for _ in texts]
        return dense, sparse

    async def encode_query(self, text: str) -> tuple[list[float], qm.SparseVector]:
        return self._vector(text), qm.SparseVector(indices=[0], values=[1.0])


class FakeStore:
    """In-memory stand-in for the Qdrant store."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.ensured_dim: int | None = None
        self.deleted: list[str] = []

    async def ensure_collection(self, dim: int) -> None:
        self.ensured_dim = dim

    async def health(self) -> bool:
        return True

    async def upsert(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[qm.SparseVector],
    ) -> None:
        self.chunks.extend(chunks)

    async def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]

    async def count(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return len(self.chunks)
        return sum(1 for c in self.chunks if c.doc_id == doc_id)

    async def hybrid_search(
        self,
        dense_query: list[float],
        sparse_query: qm.SparseVector,
        limit: int,
        doc_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        # Returns insertion order, standing in for "whatever fusion produced".
        # Tests assert on what the reranker then does to that order.
        pool = [c for c in self.chunks if not doc_ids or c.doc_id in doc_ids]
        return [SearchResult(chunk=c, score=1.0) for c in pool[:limit]]

    async def close(self) -> None:
        return None


class FakeReranker:
    """Scores by lexical overlap so ordering assertions are predictable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def warm_up(self) -> None:
        return None

    async def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, len(documents)))
        return [_overlap(query, doc) for doc in documents]


class FakeLlm:
    """Scriptable LLM. ``replies`` is consumed in order; the last one repeats."""

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self.replies = list(replies or ["A grounded answer [1]."])
        self.completions: list[list[dict[str, Any]]] = []
        self.streams: list[list[dict[str, Any]]] = []
        self.captions: list[str] = []
        self.configured = True
        self.error: Exception | None = None

    def _next(self) -> str:
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self.error:
            raise self.error
        self.completions.append(list(messages))
        return self._next()

    async def stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        if self.error:
            raise self.error
        self.streams.append(list(messages))
        for word in self._next().split(" "):
            yield word + " "

    async def describe_image(self, image_path: Any, prompt: str) -> str:
        if self.error:
            raise self.error
        self.captions.append(str(image_path))
        return f"Caption for {getattr(image_path, 'name', image_path)}"

    async def health(self) -> bool:
        return True


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def fake_llm() -> FakeLlm:
    return FakeLlm()


@pytest.fixture
def make_llm() -> Callable[..., FakeLlm]:
    """Build a FakeLlm with scripted replies, without importing across test modules."""
    return FakeLlm


@pytest.fixture
def app() -> Any:
    return create_app()


@pytest.fixture
def client(app: Any) -> Iterator[TestClient]:
    """A client that skips lifespan, so no models load and nothing connects."""
    with_lifespan = TestClient(app)
    yield with_lifespan
    app.dependency_overrides.clear()
