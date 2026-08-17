"""RAG orchestration: condense → retrieve → generate, streamed."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from pydantic import BaseModel

from rag import obs
from rag.chat.llm import LlmClient, LlmError, get_llm
from rag.chat.memory import ConversationStore, get_conversation_store
from rag.chat.prompts import (
    NO_CONTEXT_ANSWER,
    build_answer_messages,
    build_condense_messages,
)
from rag.config import Settings, get_settings
from rag.logging import get_logger
from rag.retrieval.searcher import Searcher, get_searcher, to_sources
from rag.schemas import (
    ChatRequest,
    Conversation,
    DoneEvent,
    ErrorEvent,
    Message,
    MetaEvent,
    Role,
    SearchResult,
    Source,
    SourcesEvent,
    TokenEvent,
)

log = get_logger(__name__)

#: A condensed query longer than this means the model ignored the instruction
#: and started answering instead of rewriting; fall back to the raw question.
CONDENSE_MAX_CHARS = 400


class ChatService:
    def __init__(
        self,
        settings: Settings,
        searcher: Searcher,
        llm: LlmClient,
        conversations: ConversationStore,
    ) -> None:
        self._settings = settings
        self._searcher = searcher
        self._llm = llm
        self._conversations = conversations

    # --- streaming --------------------------------------------------------

    async def stream(self, request: ChatRequest) -> AsyncIterator[tuple[str, BaseModel]]:
        """Yield ``(event_name, payload)`` pairs for the SSE endpoint.

        Emitted in order: ``meta``, ``sources``, many ``token``, then ``done``
        (or ``error``). Sources arrive before the first token so the UI can show
        what the answer is grounded in while it is still being written.
        """
        started = time.perf_counter()
        conversation = await self._conversations.get_or_create(
            request.conversation_id, request.doc_ids
        )
        yield "meta", MetaEvent(conversation_id=conversation.id)

        try:
            search_query = await self._condense(request.message, conversation)
            results = await self._searcher.search(search_query, doc_ids=conversation.doc_ids)

            if not results:
                yield "token", TokenEvent(text=NO_CONTEXT_ANSWER)
                await self._persist(conversation, request.message, NO_CONTEXT_ANSWER, [])
                yield "done", DoneEvent(took_ms=(time.perf_counter() - started) * 1000)
                return

            sources = to_sources(results)
            yield "sources", SourcesEvent(sources=sources)

            messages = build_answer_messages(
                request.message, results, conversation, self._settings.history_turns
            )
            parts: list[str] = []
            async for token in self._llm.stream(messages):
                parts.append(token)
                yield "token", TokenEvent(text=token)

            answer = "".join(parts).strip()
            await self._persist(conversation, request.message, answer, sources)
            yield "done", DoneEvent(took_ms=(time.perf_counter() - started) * 1000)

        except LlmError as exc:
            log.warning("chat_llm_error", error=str(exc))
            yield "error", ErrorEvent(message=str(exc))
        except Exception as exc:
            log.exception("chat_failed")
            yield "error", ErrorEvent(message=f"Unexpected error: {exc}")

    # --- non-streaming (used by the eval harness) -------------------------

    @obs.observe("answer")
    async def answer(
        self, question: str, doc_ids: list[str] | None = None
    ) -> tuple[str, list[SearchResult]]:
        results = await self._searcher.search(question, doc_ids=doc_ids)
        if not results:
            return NO_CONTEXT_ANSWER, []
        # No history: the eval harness asks each question independently.
        messages = build_answer_messages(question, results, Conversation(), 0)
        return await self._llm.complete(messages), results

    # --- internals --------------------------------------------------------

    @obs.observe("condense_query")
    async def _condense(self, question: str, conversation: Conversation) -> str:
        """Rewrite a follow-up into a standalone query for retrieval.

        Without this, "and the second one?" is embedded literally and retrieves
        nothing useful. On any failure the original question is used, so a
        condensing hiccup degrades quality rather than breaking the request.
        """
        if not conversation.messages:
            return question
        try:
            rewritten = (
                await self._llm.complete(
                    build_condense_messages(question, conversation),
                    temperature=0.0,
                    max_tokens=120,
                )
            ).strip()
        except LlmError as exc:
            log.debug("condense_failed", error=str(exc))
            return question

        if not rewritten or len(rewritten) > CONDENSE_MAX_CHARS:
            return question
        log.debug("condensed_query", original=question, rewritten=rewritten)
        return rewritten

    async def _persist(
        self,
        conversation: Conversation,
        question: str,
        answer: str,
        sources: list[Source],
    ) -> None:
        await self._conversations.append(conversation, Message(role=Role.USER, content=question))
        await self._conversations.append(
            conversation, Message(role=Role.ASSISTANT, content=answer, sources=sources)
        )


_service: ChatService | None = None


def get_chat_service() -> ChatService:
    global _service
    if _service is None:
        _service = ChatService(get_settings(), get_searcher(), get_llm(), get_conversation_store())
    return _service
