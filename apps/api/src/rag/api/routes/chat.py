"""Streaming chat and conversation history."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from rag.api.deps import ChatServiceDep, ConversationsDep
from rag.api.sse import SSE_HEADERS, event
from rag.chat.service import ChatService
from rag.logging import get_logger
from rag.schemas import ChatRequest, Conversation, ErrorEvent

router = APIRouter(tags=["chat"])
log = get_logger(__name__)


@router.post("/chat")
async def chat(request: ChatRequest, service: ChatServiceDep) -> StreamingResponse:
    """Answer a question over the selected documents, streamed as SSE.

    Event order is ``meta`` → ``sources`` → ``token``* → ``done``, or a single
    ``error``. The old endpoint blocked until generation finished and returned
    one JSON blob, so the user watched a typing indicator for the whole answer.
    """
    return StreamingResponse(
        _answer_stream(request, service),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _answer_stream(request: ChatRequest, service: ChatService) -> AsyncIterator[str]:
    try:
        async for name, payload in service.stream(request):
            yield event(name, payload)
    except Exception as exc:  # pragma: no cover - the service already guards
        log.exception("chat_stream_failed")
        yield event("error", ErrorEvent(message=str(exc)))


@router.get("/conversations", response_model=list[Conversation])
async def list_conversations(conversations: ConversationsDep) -> list[Conversation]:
    return await conversations.list_all()


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str, conversations: ConversationsDep) -> Conversation:
    conversation = await conversations.get(conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such conversation.")
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str, conversations: ConversationsDep) -> None:
    await conversations.delete(conversation_id)
