"""Document registry and its live progress feed."""

from __future__ import annotations

import asyncio

from rag.registry import DocumentRegistry
from rag.schemas import Document, IngestStage


def make_document(title: str = "Paper") -> Document:
    return Document(filename=f"{title}.pdf", title=title, size_bytes=100)


async def test_save_then_get_round_trips() -> None:
    registry = DocumentRegistry()
    document = await registry.save(make_document())

    loaded = await registry.get(document.id)

    assert loaded is not None
    assert loaded.id == document.id
    assert loaded.title == "Paper"


async def test_get_unknown_returns_none() -> None:
    assert await DocumentRegistry().get("nope") is None


async def test_list_all_is_newest_first() -> None:
    registry = DocumentRegistry()
    first = await registry.save(make_document("First"))
    second = make_document("Second")
    # created_at drives ordering, so make the second unambiguously later.
    second.created_at = first.created_at.replace(microsecond=0).replace(
        year=first.created_at.year + 1
    )
    await registry.save(second)

    titles = [d.title for d in await registry.list_all()]

    assert titles == ["Second", "First"]


async def test_advance_updates_stage_and_detail() -> None:
    registry = DocumentRegistry()
    document = await registry.save(make_document())

    await registry.advance(document, IngestStage.EMBEDDING, "Embedding 10 passages")

    loaded = await registry.get(document.id)
    assert loaded is not None
    assert loaded.stage is IngestStage.EMBEDDING
    assert loaded.detail == "Embedding 10 passages"


async def test_advance_sets_extra_fields() -> None:
    registry = DocumentRegistry()
    document = await registry.save(make_document())

    await registry.advance(document, IngestStage.READY, "Done", chunk_count=42)

    loaded = await registry.get(document.id)
    assert loaded is not None
    assert loaded.chunk_count == 42
    assert loaded.is_ready is True


async def test_fail_records_the_error() -> None:
    registry = DocumentRegistry()
    document = await registry.save(make_document())

    await registry.fail(document, "parser exploded")

    loaded = await registry.get(document.id)
    assert loaded is not None
    assert loaded.stage is IngestStage.FAILED
    assert loaded.error == "parser exploded"


async def test_progress_is_monotonic_across_stages() -> None:
    stages = [
        IngestStage.QUEUED,
        IngestStage.PARSING,
        IngestStage.CAPTIONING,
        IngestStage.CHUNKING,
        IngestStage.EMBEDDING,
        IngestStage.READY,
    ]
    document = make_document()
    progresses = []
    for stage in stages:
        document.stage = stage
        progresses.append(document.progress)

    assert progresses == sorted(progresses)
    assert progresses[0] == 0.0
    assert progresses[-1] == 1.0


async def test_delete_removes_from_index_and_store() -> None:
    registry = DocumentRegistry()
    document = await registry.save(make_document())

    await registry.delete(document.id)

    assert await registry.get(document.id) is None
    assert await registry.list_all() == []


async def test_watch_receives_published_updates() -> None:
    registry = DocumentRegistry()
    document = await registry.save(make_document())

    received: list[Document] = []
    updates = registry.watch(document.id)

    async def consume() -> None:
        async for update in updates:
            received.append(update)
            break

    task = asyncio.create_task(consume())
    # Give the subscriber a moment to actually subscribe before publishing.
    await asyncio.sleep(0.05)
    await registry.advance(document, IngestStage.PARSING, "Parsing")
    await asyncio.wait_for(task, timeout=2)
    await updates.aclose()

    assert len(received) == 1
    assert received[0].stage is IngestStage.PARSING


async def test_watch_filters_to_the_requested_document() -> None:
    registry = DocumentRegistry()
    watched = await registry.save(make_document("Watched"))
    other = await registry.save(make_document("Other"))

    received: list[Document] = []
    updates = registry.watch(watched.id)

    async def consume() -> None:
        async for update in updates:
            received.append(update)
            break

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    # The unrelated document's update must not wake the watcher.
    await registry.advance(other, IngestStage.PARSING, "Parsing other")
    await asyncio.sleep(0.05)
    assert received == []

    await registry.advance(watched, IngestStage.PARSING, "Parsing watched")
    await asyncio.wait_for(task, timeout=2)
    await updates.aclose()

    assert [d.title for d in received] == ["Watched"]
