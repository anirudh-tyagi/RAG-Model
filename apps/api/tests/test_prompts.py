"""Prompt assembly — numbering and provenance are what make citations possible."""

from __future__ import annotations

from rag.chat.prompts import build_answer_messages, build_condense_messages, format_context
from rag.schemas import Chunk, Conversation, Message, Role, SearchResult


def result(text: str, **kwargs: object) -> SearchResult:
    defaults: dict[str, object] = {
        "doc_id": "d1",
        "doc_title": "Annual Report",
        "filename": "report.pdf",
        "chunk_index": 0,
        "text": text,
    }
    defaults.update(kwargs)
    return SearchResult(chunk=Chunk(**defaults), score=1.0)  # type: ignore[arg-type]


def test_context_is_numbered_from_one() -> None:
    context = format_context([result("first"), result("second")])

    assert context.startswith("[1]")
    assert "[2]" in context
    assert "[0]" not in context


def test_context_labels_document_page_and_section() -> None:
    context = format_context([result("body", page=12, heading="Revenue")])

    assert "Annual Report" in context
    assert "page 12" in context
    assert "Revenue" in context


def test_context_marks_figure_chunks() -> None:
    context = format_context([result("a bar chart", page=4, is_figure=True)])

    assert "figure" in context


def test_answer_messages_start_with_the_system_prompt() -> None:
    messages = build_answer_messages(
        "What is revenue?", [result("Revenue was 5.1B")], Conversation(), 0
    )

    assert messages[0]["role"] == "system"
    assert "cite" in messages[0]["content"].lower()


def test_answer_messages_end_with_context_and_question() -> None:
    messages = build_answer_messages(
        "What is revenue?", [result("Revenue was 5.1B")], Conversation(), 0
    )

    last = messages[-1]
    assert last["role"] == "user"
    assert "CONTEXT:" in last["content"]
    assert "Revenue was 5.1B" in last["content"]
    assert "What is revenue?" in last["content"]


def test_answer_messages_include_recent_history() -> None:
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Who wrote it?"),
            Message(role=Role.ASSISTANT, content="Acme Corp."),
        ]
    )
    messages = build_answer_messages(
        "And revenue?", [result("5.1B")], conversation, history_turns=6
    )

    contents = [m["content"] for m in messages]
    assert "Who wrote it?" in contents
    assert "Acme Corp." in contents


def test_history_turns_zero_excludes_history() -> None:
    conversation = Conversation(messages=[Message(role=Role.USER, content="Earlier question")])
    messages = build_answer_messages("Now?", [result("x")], conversation, history_turns=0)

    assert not any("Earlier question" in m["content"] for m in messages)


def test_history_is_truncated_to_the_configured_window() -> None:
    conversation = Conversation(
        messages=[Message(role=Role.USER, content=f"turn {n}") for n in range(10)]
    )
    messages = build_answer_messages("Now?", [result("x")], conversation, history_turns=3)

    contents = " ".join(m["content"] for m in messages)
    assert "turn 9" in contents
    assert "turn 0" not in contents


def test_condense_messages_include_the_transcript_and_latest_turn() -> None:
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Compare the two models."),
            Message(role=Role.ASSISTANT, content="Model A and Model B differ in size."),
        ]
    )
    messages = build_condense_messages("what about the second one?", conversation)

    assert messages[0]["role"] == "system"
    body = messages[-1]["content"]
    assert "Compare the two models." in body
    assert "what about the second one?" in body
