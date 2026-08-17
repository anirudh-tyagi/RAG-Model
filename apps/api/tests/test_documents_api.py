"""Upload validation and document endpoints.

The upload path is where the old code's security bug lived, so it gets the most
attention here.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.config import Settings
from rag.retrieval.store import get_store

PDF_BYTES = b"%PDF-1.7\n" + b"x" * 512


@pytest.fixture(autouse=True)
def no_real_queue(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture enqueued document ids instead of talking to Redis via arq."""
    enqueued: list[str] = []

    async def fake_enqueue(doc_id: str) -> str:
        enqueued.append(doc_id)
        return f"ingest:{doc_id}"

    monkeypatch.setattr("rag.api.routes.documents.enqueue_ingest", fake_enqueue)
    return enqueued


def upload(client: TestClient, name: str, content: bytes) -> Any:
    return client.post(
        "/api/documents",
        files={"file": (name, content, "application/pdf")},
    )


# --- happy path ---------------------------------------------------------------


def test_valid_pdf_is_accepted_and_queued(client: TestClient, no_real_queue: list[str]) -> None:
    response = upload(client, "report.pdf", PDF_BYTES)

    assert response.status_code == 202
    body = response.json()
    assert body["document"]["filename"] == "report.pdf"
    assert body["document"]["stage"] == "queued"
    assert body["document"]["size_bytes"] == len(PDF_BYTES)
    assert no_real_queue == [body["document"]["id"]]


def test_upload_is_stored_under_a_uuid_not_the_supplied_name(
    client: TestClient, settings: Settings
) -> None:
    response = upload(client, "report.pdf", PDF_BYTES)
    doc_id = response.json()["document"]["id"]

    stored = list(settings.upload_dir.iterdir())
    assert [p.name for p in stored] == [f"{doc_id}.pdf"]


def test_title_is_derived_from_the_filename(client: TestClient) -> None:
    response = upload(client, "annual_report_2024.pdf", PDF_BYTES)

    assert response.json()["document"]["title"] == "annual report 2024"


# --- validation ---------------------------------------------------------------


def test_non_pdf_extension_is_rejected(client: TestClient) -> None:
    response = upload(client, "notes.txt", PDF_BYTES)

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_pdf_extension_with_wrong_magic_bytes_is_rejected(client: TestClient) -> None:
    response = upload(client, "fake.pdf", b"GIF89a not really a pdf")

    assert response.status_code == 400
    assert "header" in response.json()["detail"].lower()


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = upload(client, "empty.pdf", b"")

    assert response.status_code == 400


def test_oversized_file_is_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(type(settings), "max_upload_bytes", property(lambda self: 128))

    response = upload(client, "big.pdf", PDF_BYTES)

    assert response.status_code == 413


def test_rejected_upload_leaves_no_file_behind(client: TestClient, settings: Settings) -> None:
    upload(client, "fake.pdf", b"GIF89a not really a pdf")

    assert list(settings.upload_dir.iterdir()) == []


# --- path traversal -----------------------------------------------------------


@pytest.mark.parametrize(
    "hostile_name",
    [
        "../../../../etc/passwd.pdf",
        "..\\..\\windows\\system32\\evil.pdf",
        "/absolute/path/evil.pdf",
    ],
)
def test_traversal_filenames_cannot_escape_the_upload_directory(
    client: TestClient, settings: Settings, hostile_name: str
) -> None:
    response = upload(client, hostile_name, PDF_BYTES)

    assert response.status_code == 202
    doc_id = response.json()["document"]["id"]
    # Exactly one file, inside the upload dir, named after the uuid.
    stored = list(settings.upload_dir.iterdir())
    assert [p.name for p in stored] == [f"{doc_id}.pdf"]
    # The recorded filename is the basename only.
    assert "/" not in response.json()["document"]["filename"]
    assert "\\" not in response.json()["document"]["filename"]


# --- listing, fetching, deleting ---------------------------------------------


def test_list_returns_uploaded_documents(client: TestClient) -> None:
    upload(client, "one.pdf", PDF_BYTES)
    upload(client, "two.pdf", PDF_BYTES)

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert {d["filename"] for d in response.json()} == {"one.pdf", "two.pdf"}


def test_list_is_empty_initially(client: TestClient) -> None:
    assert client.get("/api/documents").json() == []


def test_get_document_by_id(client: TestClient) -> None:
    doc_id = upload(client, "one.pdf", PDF_BYTES).json()["document"]["id"]

    response = client.get(f"/api/documents/{doc_id}")

    assert response.status_code == 200
    assert response.json()["id"] == doc_id
    assert response.json()["progress"] == 0.0


def test_get_unknown_document_is_404(client: TestClient) -> None:
    assert client.get("/api/documents/missing").status_code == 404


def test_delete_removes_vectors_files_and_registry_entry(
    client: TestClient, app: Any, fake_store: Any, settings: Settings
) -> None:
    app.dependency_overrides[get_store] = lambda: fake_store
    doc_id = upload(client, "one.pdf", PDF_BYTES).json()["document"]["id"]

    response = client.delete(f"/api/documents/{doc_id}")

    assert response.status_code == 204
    assert fake_store.deleted == [doc_id]
    assert list(settings.upload_dir.iterdir()) == []
    assert client.get(f"/api/documents/{doc_id}").status_code == 404


def test_delete_unknown_document_is_404(client: TestClient, app: Any, fake_store: Any) -> None:
    app.dependency_overrides[get_store] = lambda: fake_store

    assert client.delete("/api/documents/missing").status_code == 404


# --- progress stream ----------------------------------------------------------


@pytest.fixture
def fast_keepalive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the idle keepalive so streaming tests don't wait 15 real seconds."""
    monkeypatch.setattr("rag.api.routes.documents.KEEPALIVE_SECONDS", 0.05)


def test_events_stream_sends_current_state_immediately(
    client: TestClient, fast_keepalive: None
) -> None:
    doc_id = upload(client, "one.pdf", PDF_BYTES).json()["document"]["id"]

    with client.stream("GET", f"/api/documents/{doc_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        chunks = []
        for line in response.iter_lines():
            chunks.append(line)
            # First frame is `event: progress`, then its data line.
            if len(chunks) >= 2:
                break

    assert chunks[0] == "event: progress"
    assert doc_id in chunks[1]


def test_events_stream_emits_keepalives_while_idle(
    client: TestClient, fast_keepalive: None
) -> None:
    doc_id = upload(client, "one.pdf", PDF_BYTES).json()["document"]["id"]

    with client.stream("GET", f"/api/documents/{doc_id}/events") as response:
        lines = []
        for line in response.iter_lines():
            lines.append(line)
            if any(line.startswith(":") for line in lines):
                break

    assert any(line.startswith(": keepalive") for line in lines)


def test_events_stream_closes_once_the_document_is_terminal(client: TestClient) -> None:
    """A ready document ends the stream instead of holding it open forever."""
    doc_id = upload(client, "one.pdf", PDF_BYTES).json()["document"]["id"]
    client.get(f"/api/documents/{doc_id}")

    import anyio

    from rag.registry import get_registry
    from rag.schemas import IngestStage

    async def mark_ready() -> None:
        registry = get_registry()
        document = await registry.get(doc_id)
        assert document is not None
        await registry.advance(document, IngestStage.READY, "Done", chunk_count=3)

    anyio.run(mark_ready)

    with client.stream("GET", f"/api/documents/{doc_id}/events") as response:
        body = "".join(response.iter_text())

    # One progress frame, then EOF — no keepalives, no hanging.
    assert body.count("event: progress") == 1
    assert '"stage":"ready"' in body


def test_events_stream_for_unknown_document_is_404(client: TestClient) -> None:
    assert client.get("/api/documents/missing/events").status_code == 404
