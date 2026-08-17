"""Server-side conversation history.

The old app kept a ``chatHistory`` array in ``script.js`` and never sent it
anywhere — the backend saw one isolated question per request, so follow-ups like
"what about the second one?" had no chance of resolving. History now lives in
Redis, keyed by conversation, and feeds both the prompt and query condensing.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag import kv
from rag.kv import key
from rag.schemas import Conversation, Message

INDEX_KEY = key("conversations")
#: Conversations are a convenience, not a system of record.
TTL_SECONDS = 60 * 60 * 24 * 30

TITLE_MAX = 60


def _conv_key(conversation_id: str) -> str:
    return key("conv", conversation_id)


class ConversationStore:
    async def get(self, conversation_id: str) -> Conversation | None:
        raw = await kv.get_redis().get(_conv_key(conversation_id))
        return Conversation.model_validate_json(raw) if raw else None

    async def save(self, conversation: Conversation) -> Conversation:
        conversation.updated_at = datetime.now(UTC)
        redis = kv.get_redis()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.set(
                _conv_key(conversation.id),
                conversation.model_dump_json(),
                ex=TTL_SECONDS,
            )
            pipe.zadd(INDEX_KEY, {conversation.id: conversation.updated_at.timestamp()})
            await pipe.execute()
        return conversation

    async def get_or_create(self, conversation_id: str | None, doc_ids: list[str]) -> Conversation:
        if conversation_id:
            existing = await self.get(conversation_id)
            if existing is not None:
                # A later request may narrow or widen the document selection.
                if doc_ids and doc_ids != existing.doc_ids:
                    existing.doc_ids = doc_ids
                return existing
        return await self.save(Conversation(doc_ids=doc_ids))

    async def append(self, conversation: Conversation, message: Message) -> Conversation:
        conversation.messages.append(message)
        if len(conversation.messages) == 1:
            conversation.title = _derive_title(message.content)
        return await self.save(conversation)

    async def list_all(self, limit: int = 50) -> list[Conversation]:
        redis = kv.get_redis()
        ids = await redis.zrevrange(INDEX_KEY, 0, limit - 1)
        if not ids:
            return []
        raws = await redis.mget([_conv_key(i) for i in ids])
        return [Conversation.model_validate_json(raw) for raw in raws if raw]

    async def delete(self, conversation_id: str) -> None:
        redis = kv.get_redis()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.delete(_conv_key(conversation_id))
            pipe.zrem(INDEX_KEY, conversation_id)
            await pipe.execute()


def _derive_title(first_message: str) -> str:
    text = " ".join(first_message.split())
    if len(text) <= TITLE_MAX:
        return text or "New conversation"
    return text[:TITLE_MAX].rstrip() + "…"


_store: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store
