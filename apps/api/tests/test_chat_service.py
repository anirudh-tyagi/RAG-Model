"""Chat orchestration: event sequence, memory, condensing, failure handling."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from rag.chat.llm import LlmError
from rag.chat.memory import ConversationStore
from rag.chat.prompts import NO_CONTEXT_ANSWER
from rag.chat.service import ChatService
from rag.config import Settings
from rag.retrieval.searcher import Searcher
from rag.schemas import ChatRequest, Chunk, Role


def chunk(text: str, doc_id: str = "d1", **kwargs: Any) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        doc_title="Doc",
        filename="doc.pdf",
        chunk_index=0,
        text=text,
        **kwargs,
    )


def build_service(
    settings: Settings,
    store: Any,
    embedder: Any,
    reranker: Any,
    llm: Any,
) -> tuple[ChatService, ConversationStore]:
    conversations = ConversationStore()
    searcher = Searcher(settings, store, embedder, reranker)
    return ChatService(settings, searcher, llm, conversations), conversations


async def collect(service: ChatService, request: ChatRequest) -> list[tuple[str, BaseModel]]:
    return [event async for event in service.stream(request)]


# --- happy path ---------------------------------------------------------------


async def test_event_sequence_is_meta_sources_tokens_done(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    fake_store.chunks = [chunk("Revenue rose to 5.1B in 2024.", page=12)]
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, fake_llm)

    events = await collect(service, ChatRequest(message="What was revenue?"))
    names = [name for name, _ in events]

    assert names[0] == "meta"
    assert names[1] == "sources"
    assert names[-1] == "done"
    assert "token" in names
    # Sources arrive before any token, so the UI can render them while streaming.
    assert names.index("sources") < names.index("token")


async def test_tokens_reassemble_into_the_answer(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, make_llm: Any
) -> None:
    llm = make_llm(["Revenue was 5.1B [1]."])
    fake_store.chunks = [chunk("Revenue rose to 5.1B.", page=12)]
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, llm)

    events = await collect(service, ChatRequest(message="What was revenue?"))
    text = "".join(p.text for name, p in events if name == "token")  # type: ignore[attr-defined]

    assert "5.1B" in text
    assert "[1]" in text


async def test_sources_carry_page_numbers(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    fake_store.chunks = [chunk("Revenue rose.", page=12, heading="Financials")]
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, fake_llm)

    events = await collect(service, ChatRequest(message="revenue"))
    sources_event = next(p for name, p in events if name == "sources")

    source = sources_event.sources[0]  # type: ignore[attr-defined]
    assert source.n == 1
    assert source.page == 12
    assert source.heading == "Financials"


async def test_meta_returns_a_conversation_id_that_persists(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    fake_store.chunks = [chunk("Revenue rose.")]
    service, conversations = build_service(
        settings, fake_store, fake_embedder, fake_reranker, fake_llm
    )

    events = await collect(service, ChatRequest(message="revenue"))
    conversation_id = next(p for name, p in events if name == "meta").conversation_id  # type: ignore[attr-defined]

    stored = await conversations.get(conversation_id)
    assert stored is not None
    assert [m.role for m in stored.messages] == [Role.USER, Role.ASSISTANT]
    assert stored.messages[1].sources


async def test_second_turn_reuses_the_same_conversation(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    fake_store.chunks = [chunk("Revenue rose.")]
    service, conversations = build_service(
        settings, fake_store, fake_embedder, fake_reranker, fake_llm
    )

    first = await collect(service, ChatRequest(message="revenue?"))
    conversation_id = next(p for name, p in first if name == "meta").conversation_id  # type: ignore[attr-defined]

    await collect(service, ChatRequest(message="and profit?", conversation_id=conversation_id))

    stored = await conversations.get(conversation_id)
    assert stored is not None
    assert len(stored.messages) == 4


async def test_conversation_title_comes_from_the_first_question(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    fake_store.chunks = [chunk("Revenue rose.")]
    service, conversations = build_service(
        settings, fake_store, fake_embedder, fake_reranker, fake_llm
    )

    events = await collect(service, ChatRequest(message="What was 2024 revenue?"))
    conversation_id = next(p for name, p in events if name == "meta").conversation_id  # type: ignore[attr-defined]

    stored = await conversations.get(conversation_id)
    assert stored is not None
    assert stored.title == "What was 2024 revenue?"


# --- condensing ---------------------------------------------------------------


async def test_first_question_is_not_condensed(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    fake_store.chunks = [chunk("Revenue rose.")]
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, fake_llm)

    await collect(service, ChatRequest(message="revenue?"))

    # No history, so no condensing round trip — only the streamed answer.
    assert fake_llm.completions == []


async def test_follow_up_is_condensed_before_retrieval(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, make_llm: Any
) -> None:
    llm = make_llm(["2024 segment revenue", "It was 5.1B [1]."])
    fake_store.chunks = [chunk("Revenue rose.")]
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, llm)

    first = await collect(service, ChatRequest(message="What was revenue?"))
    conversation_id = next(p for name, p in first if name == "meta").conversation_id  # type: ignore[attr-defined]

    await collect(
        service, ChatRequest(message="and the second one?", conversation_id=conversation_id)
    )

    # The condense call happened, and saw the earlier turn.
    assert len(llm.completions) == 1
    assert "and the second one?" in llm.completions[0][-1]["content"]


async def test_condense_failure_falls_back_to_the_raw_question(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any
) -> None:
    """A condensing hiccup should degrade quality, not break the request."""

    class FlakyCondenseLlm:
        configured = True

        def __init__(self) -> None:
            self.streamed = False

        async def complete(self, messages: Any, **kwargs: Any) -> str:
            raise LlmError("condense endpoint down")

        async def stream(self, messages: Any, **kwargs: Any) -> Any:
            self.streamed = True
            yield "Answer anyway [1]."

    llm = FlakyCondenseLlm()
    fake_store.chunks = [chunk("Revenue rose.")]
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, llm)

    first = await collect(service, ChatRequest(message="What was revenue?"))
    conversation_id = next(p for name, p in first if name == "meta").conversation_id  # type: ignore[attr-defined]
    events = await collect(
        service, ChatRequest(message="and profit?", conversation_id=conversation_id)
    )

    assert [name for name, _ in events][-1] == "done"
    assert llm.streamed is True


async def test_overlong_condense_output_is_discarded(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, make_llm: Any
) -> None:
    # The model ignored the instruction and answered instead of rewriting.
    llm = make_llm(["x" * 900, "Answer [1]."])
    fake_store.chunks = [chunk("Revenue rose.")]
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, llm)

    first = await collect(service, ChatRequest(message="What was revenue?"))
    conversation_id = next(p for name, p in first if name == "meta").conversation_id  # type: ignore[attr-defined]
    events = await collect(
        service, ChatRequest(message="and profit?", conversation_id=conversation_id)
    )

    assert [name for name, _ in events][-1] == "done"


# --- degraded paths -----------------------------------------------------------


async def test_no_results_abstains_without_emitting_sources(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, fake_llm)

    events = await collect(service, ChatRequest(message="anything"))
    names = [name for name, _ in events]

    assert "sources" not in names
    assert names[-1] == "done"
    text = "".join(p.text for name, p in events if name == "token")  # type: ignore[attr-defined]
    assert text == NO_CONTEXT_ANSWER
    # The model was never called: there was nothing to ground an answer in.
    assert fake_llm.streams == []


async def test_llm_failure_yields_an_error_event(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    fake_store.chunks = [chunk("Revenue rose.")]
    fake_llm.error = LlmError("provider is down")
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, fake_llm)

    events = await collect(service, ChatRequest(message="revenue?"))
    names = [name for name, _ in events]

    assert names[-1] == "error"
    assert "provider is down" in events[-1][1].message  # type: ignore[attr-defined]


async def test_doc_ids_scope_the_conversation(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    fake_store.chunks = [
        chunk("Revenue in A", doc_id="a"),
        chunk("Revenue in B", doc_id="b"),
    ]
    service, conversations = build_service(
        settings, fake_store, fake_embedder, fake_reranker, fake_llm
    )

    events = await collect(service, ChatRequest(message="revenue?", doc_ids=["b"]))
    sources = next(p for name, p in events if name == "sources").sources  # type: ignore[attr-defined]

    assert {s.doc_id for s in sources} == {"b"}
    conversation_id = next(p for name, p in events if name == "meta").conversation_id  # type: ignore[attr-defined]
    stored = await conversations.get(conversation_id)
    assert stored is not None
    assert stored.doc_ids == ["b"]


# --- non-streaming path used by the eval harness ------------------------------


async def test_answer_returns_text_and_results(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    fake_store.chunks = [chunk("Revenue rose to 5.1B.")]
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, fake_llm)

    answer, results = await service.answer("What was revenue?")

    assert answer
    assert len(results) == 1


async def test_answer_abstains_with_no_results(
    settings: Settings, fake_store: Any, fake_embedder: Any, fake_reranker: Any, fake_llm: Any
) -> None:
    service, _ = build_service(settings, fake_store, fake_embedder, fake_reranker, fake_llm)

    answer, results = await service.answer("anything")

    assert answer == NO_CONTEXT_ANSWER
    assert results == []
