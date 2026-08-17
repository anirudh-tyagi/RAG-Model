"""Dependency aliases for the routes.

Going through ``Depends`` rather than calling the ``get_*`` singletons inline
gives the tests a supported seam (``app.dependency_overrides``) so they can run
with a fake store, a fake embedder and a mocked LLM instead of downloading
models and standing up Qdrant.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from rag.chat.llm import LlmClient, get_llm
from rag.chat.memory import ConversationStore, get_conversation_store
from rag.chat.service import ChatService, get_chat_service
from rag.config import Settings, get_settings
from rag.registry import DocumentRegistry, get_registry
from rag.retrieval.searcher import Searcher, get_searcher
from rag.retrieval.store import VectorStore, get_store
from rag.stt import Transcriber, get_transcriber

SettingsDep = Annotated[Settings, Depends(get_settings)]
RegistryDep = Annotated[DocumentRegistry, Depends(get_registry)]
StoreDep = Annotated[VectorStore, Depends(get_store)]
SearcherDep = Annotated[Searcher, Depends(get_searcher)]
LlmDep = Annotated[LlmClient, Depends(get_llm)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
ConversationsDep = Annotated[ConversationStore, Depends(get_conversation_store)]
TranscriberDep = Annotated[Transcriber, Depends(get_transcriber)]

__all__ = [
    "ChatServiceDep",
    "ConversationsDep",
    "LlmDep",
    "RegistryDep",
    "SearcherDep",
    "SettingsDep",
    "StoreDep",
    "TranscriberDep",
]
