"""Document registry: ingestion state plus a live progress feed.

Every stage transition is persisted *and* published, which is what lets the
upload page show real progress instead of the old ``upload.js`` behaviour —
a hardcoded 1.5 second wait and a redirect, whether or not the pipeline had
even started, let alone succeeded.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from rag import kv
from rag.kv import key
from rag.logging import get_logger
from rag.schemas import Document, IngestStage

log = get_logger(__name__)

EVENTS_CHANNEL = key("doc-events")
INDEX_KEY = key("docs")


def _doc_key(doc_id: str) -> str:
    return key("doc", doc_id)


class DocumentRegistry:
    async def save(self, document: Document, publish: bool = True) -> Document:
        document.updated_at = datetime.now(UTC)
        redis = kv.get_redis()
        payload = document.model_dump_json()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.set(_doc_key(document.id), payload)
            pipe.zadd(INDEX_KEY, {document.id: document.created_at.timestamp()})
            await pipe.execute()
        if publish:
            await redis.publish(EVENTS_CHANNEL, payload)
        return document

    async def get(self, doc_id: str) -> Document | None:
        raw = await kv.get_redis().get(_doc_key(doc_id))
        return Document.model_validate_json(raw) if raw else None

    async def list_all(self, limit: int = 200) -> list[Document]:
        """Newest first. Not named ``list``: that would shadow the builtin used
        in the return annotations of the methods below it."""
        redis = kv.get_redis()
        ids = await redis.zrevrange(INDEX_KEY, 0, limit - 1)
        if not ids:
            return []
        raws = await redis.mget([_doc_key(i) for i in ids])
        return [Document.model_validate_json(raw) for raw in raws if raw]

    async def list_ready(self) -> list[Document]:
        return [d for d in await self.list_all() if d.is_ready]

    async def delete(self, doc_id: str) -> None:
        redis = kv.get_redis()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.delete(_doc_key(doc_id))
            pipe.zrem(INDEX_KEY, doc_id)
            await pipe.execute()

    # --- stage transitions ------------------------------------------------

    async def advance(
        self,
        document: Document,
        stage: IngestStage,
        detail: str,
        **fields: object,
    ) -> Document:
        """Move a document to a new stage and broadcast it."""
        document.stage = stage
        document.detail = detail
        for name, value in fields.items():
            setattr(document, name, value)
        log.info("ingest_stage", doc_id=document.id, stage=stage.value, detail=detail)
        return await self.save(document)

    async def fail(self, document: Document, error: str) -> Document:
        return await self.advance(document, IngestStage.FAILED, "Ingestion failed", error=error)

    # --- live feed --------------------------------------------------------

    async def watch(self, doc_id: str | None = None) -> AsyncGenerator[Document, None]:
        """Yield documents as they change. Filters to one document when given an id.

        Typed as a generator, not merely an iterator, because callers need
        ``aclose()`` — the SSE endpoint closes it when the client disconnects,
        which is what releases the pubsub connection back to the pool.
        """
        redis = kv.get_redis()
        async with redis.pubsub() as pubsub:
            await pubsub.subscribe(EVENTS_CHANNEL)
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    document = Document.model_validate_json(message["data"])
                except Exception:  # pragma: no cover - malformed publish
                    continue
                if doc_id is None or document.id == doc_id:
                    yield document


_registry: DocumentRegistry | None = None


def get_registry() -> DocumentRegistry:
    global _registry
    if _registry is None:
        _registry = DocumentRegistry()
    return _registry
