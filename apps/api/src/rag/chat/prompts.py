"""Prompt construction.

The old template dropped raw concatenated chunks into the prompt with no
provenance, so the model could not cite and the user could not verify. Passages
are now numbered and labelled with their document, page and section, and the
model is told to cite those numbers inline.
"""

from __future__ import annotations

from typing import Any

from rag.schemas import Conversation, Message, SearchResult

ANSWER_SYSTEM = """\
You answer questions about documents the user has uploaded.

Rules:
- Answer only from the numbered passages in CONTEXT. Do not use outside knowledge.
- Cite the passages you used inline using ASCII square brackets, like [1] or [2][4]. \
Cite the specific passage that supports each claim, not every passage you were given.
- If the context does not contain the answer, say so plainly and name what is \
missing. Never invent a citation.
- Quote exact figures, names and units as they appear. Do not round or rephrase numbers.
- Be concise and direct. Use markdown, and short lists or tables where they help.\
"""

CONDENSE_SYSTEM = """\
Rewrite the user's latest message as a standalone search query, resolving any \
pronouns or references to earlier turns. Preserve all specific terms, names and \
numbers. Output only the rewritten query, with no preamble or quotation marks.\
"""

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant in the selected documents. Try rephrasing "
    "the question, or check that the document you mean has finished processing."
)


def format_context(results: list[SearchResult]) -> str:
    """Number and label passages so the model can cite them."""
    blocks: list[str] = []
    for n, result in enumerate(results, start=1):
        chunk = result.chunk
        location = [chunk.doc_title]
        if chunk.page is not None:
            location.append(f"page {chunk.page}")
        if chunk.heading:
            location.append(f"section “{chunk.heading}”")
        if chunk.is_figure:
            location.append("figure")
        blocks.append(f"[{n}] ({', '.join(location)})\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_answer_messages(
    question: str,
    results: list[SearchResult],
    conversation: Conversation,
    history_turns: int,
) -> list[dict[str, Any]]:
    """System prompt, recent history, then the question with its context."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": ANSWER_SYSTEM}]

    for message in _recent(conversation.messages, history_turns):
        messages.append({"role": message.role.value, "content": message.content})

    messages.append(
        {
            "role": "user",
            "content": f"CONTEXT:\n{format_context(results)}\n\nQUESTION: {question}",
        }
    )
    return messages


def build_condense_messages(question: str, conversation: Conversation) -> list[dict[str, Any]]:
    transcript = "\n".join(
        f"{message.role.value}: {message.content}" for message in _recent(conversation.messages, 4)
    )
    return [
        {"role": "system", "content": CONDENSE_SYSTEM},
        {
            "role": "user",
            "content": f"Conversation so far:\n{transcript}\n\nLatest message: {question}",
        },
    ]


def _recent(messages: list[Message], turns: int) -> list[Message]:
    """The last ``turns`` messages, oldest first. Assistant sources are dropped."""
    if turns <= 0:
        return []
    return messages[-turns:]
