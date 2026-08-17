"""Chat and health endpoints, end to end through the HTTP layer."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.chat.llm import get_llm
from rag.chat.memory import ConversationStore
from rag.chat.service import ChatService, get_chat_service
from rag.config import Settings
from rag.retrieval.searcher import Searcher, get_searcher
from rag.retrieval.store import get_store
from rag.schemas import Chunk


def chunk(text: str, **kwargs: Any) -> Chunk:
    return Chunk(
        doc_id="d1",
        doc_title="Report",
        filename="report.pdf",
        chunk_index=0,
        text=text,
        **kwargs,
    )


@pytest.fixture
def wired(
    app: Any,
    settings: Settings,
    fake_store: Any,
    fake_embedder: Any,
    fake_reranker: Any,
    fake_llm: Any,
) -> Any:
    """Install a chat service, searcher, store and LLM backed entirely by fakes.

    Every dependency is overridden, including on routes that only ever return
    422 — FastAPI resolves dependencies alongside request validation, so a
    missing override would still build a real Qdrant client.
    """
    searcher = Searcher(settings, fake_store, fake_embedder, fake_reranker)
    service = ChatService(settings, searcher, fake_llm, ConversationStore())
    app.dependency_overrides[get_chat_service] = lambda: service
    app.dependency_overrides[get_searcher] = lambda: searcher
    app.dependency_overrides[get_store] = lambda: fake_store
    app.dependency_overrides[get_llm] = lambda: fake_llm
    return service


def read_events(body: str) -> list[tuple[str, str]]:
    """Parse an SSE body into (event_name, data) pairs, ignoring comments."""
    events: list[tuple[str, str]] = []
    name: str | None = None
    for line in body.splitlines():
        if line.startswith("event: "):
            name = line.removeprefix("event: ")
        elif line.startswith("data: ") and name:
            events.append((name, line.removeprefix("data: ")))
    return events


def test_chat_streams_sources_then_tokens_then_done(
    client: TestClient, wired: Any, fake_store: Any
) -> None:
    fake_store.chunks = [chunk("Revenue rose to 5.1B in 2024.", page=12)]

    response = client.post("/api/chat", json={"message": "What was revenue?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in read_events(response.text)]
    assert names[0] == "meta"
    assert "sources" in names
    assert names[-1] == "done"


def test_chat_answer_text_is_streamed_in_token_events(
    client: TestClient, wired: Any, fake_store: Any
) -> None:
    fake_store.chunks = [chunk("Revenue rose to 5.1B.")]

    response = client.post("/api/chat", json={"message": "revenue?"})

    import json

    text = "".join(
        json.loads(data)["text"] for name, data in read_events(response.text) if name == "token"
    )
    assert text.strip()


def test_chat_rejects_an_empty_message(client: TestClient, wired: Any) -> None:
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_a_missing_message(client: TestClient, wired: Any) -> None:
    assert client.post("/api/chat", json={}).status_code == 422


def test_conversation_is_retrievable_after_a_turn(
    client: TestClient, wired: Any, fake_store: Any
) -> None:
    import json

    fake_store.chunks = [chunk("Revenue rose.")]
    response = client.post("/api/chat", json={"message": "revenue?"})
    meta = next(data for name, data in read_events(response.text) if name == "meta")
    conversation_id = json.loads(meta)["conversation_id"]

    stored = client.get(f"/api/conversations/{conversation_id}")

    assert stored.status_code == 200
    assert len(stored.json()["messages"]) == 2


def test_unknown_conversation_is_404(client: TestClient, wired: Any) -> None:
    assert client.get("/api/conversations/missing").status_code == 404


def test_conversations_can_be_listed_and_deleted(
    client: TestClient, wired: Any, fake_store: Any
) -> None:
    import json

    fake_store.chunks = [chunk("Revenue rose.")]
    response = client.post("/api/chat", json={"message": "revenue?"})
    meta = next(data for name, data in read_events(response.text) if name == "meta")
    conversation_id = json.loads(meta)["conversation_id"]

    assert len(client.get("/api/conversations").json()) == 1
    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    assert client.get("/api/conversations").json() == []


# --- search -------------------------------------------------------------------


def test_search_returns_ranked_passages(client: TestClient, wired: Any, fake_store: Any) -> None:
    fake_store.chunks = [chunk("unrelated gardening prose"), chunk("revenue rose sharply")]

    response = client.post("/api/search", json={"query": "revenue rose sharply"})

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["chunk"]["text"] == "revenue rose sharply"
    assert body["took_ms"] >= 0


def test_search_rejects_an_empty_query(client: TestClient, wired: Any) -> None:
    assert client.post("/api/search", json={"query": ""}).status_code == 422


# --- health -------------------------------------------------------------------


def test_health_reports_ok_when_everything_is_reachable(
    client: TestClient, wired: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def healthy() -> bool:
        return True

    monkeypatch.setattr("rag.api.routes.health.queue_healthy", healthy)

    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["qdrant"] is True
    assert body["queue"] is True
    assert body["llm_configured"] is True


def test_health_reports_degraded_when_the_queue_is_down(
    client: TestClient, wired: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dependency outage degrades the report rather than killing the process."""

    async def unhealthy() -> bool:
        return False

    monkeypatch.setattr("rag.api.routes.health.queue_healthy", unhealthy)

    body = client.get("/api/health").json()

    assert body["status"] == "degraded"
    assert body["queue"] is False


def test_health_echoes_the_active_configuration(
    client: TestClient, wired: Any, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def healthy() -> bool:
        return True

    monkeypatch.setattr("rag.api.routes.health.queue_healthy", healthy)

    body = client.get("/api/health").json()

    assert body["dense_model"] == settings.dense_model
    assert body["llm_model"] == settings.llm_model
    assert body["parser"] == settings.pdf_parser
