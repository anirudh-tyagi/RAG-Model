"""Heading-aware markdown chunking.

The old pipeline ran ``UnstructuredMarkdownLoader`` — which flattens markdown to
plain text, discarding every heading — through a 512-character recursive
splitter. Chunks therefore arrived at the embedder with no idea what section
they came from, and 512 characters is small enough to routinely cut a table or
a paragraph mid-thought.

Here chunks:

* respect paragraph and heading boundaries,
* carry the section heading as a prefix, so the embedding sees the topic and
  the citation can show it,
* keep page attribution for citations,
* overlap by a configurable tail to avoid losing facts that straddle a break.
"""

from __future__ import annotations

import re

from rag.ingest.caption import Caption
from rag.ingest.parse import ParsedPage
from rag.schemas import Chunk, Document

HEADING = re.compile(r"^(#{1,6})\s+(?P<text>.+?)\s*#*$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_pages(
    document: Document,
    pages: list[ParsedPage],
    captions: list[Caption],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Turn parsed pages and figure captions into indexable chunks."""
    chunks: list[Chunk] = []

    for page in pages:
        for text, heading in _pack_page(page.markdown, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    doc_id=document.id,
                    doc_title=document.title,
                    filename=document.filename,
                    chunk_index=len(chunks),
                    text=text,
                    page=page.page,
                    heading=heading,
                )
            )

    for caption in captions:
        location = f"page {caption.page}" if caption.page else "the document"
        chunks.append(
            Chunk(
                doc_id=document.id,
                doc_title=document.title,
                filename=document.filename,
                chunk_index=len(chunks),
                text=f"Figure from {location} of {document.title}:\n\n{caption.text}",
                page=caption.page,
                heading="Figures",
                is_figure=True,
            )
        )

    return chunks


def _pack_page(markdown: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, str | None]]:
    """Greedily pack blocks up to ``chunk_size``, returning (text, heading) pairs."""
    out: list[tuple[str, str | None]] = []
    buffer: list[str] = []
    buffer_len = 0
    buffer_heading: str | None = None
    heading: str | None = None

    def flush() -> None:
        nonlocal buffer, buffer_len
        body = "\n\n".join(buffer).strip()
        buffer, buffer_len = [], 0
        if not body:
            return
        text = f"{buffer_heading}\n\n{body}" if buffer_heading else body
        out.append((text, buffer_heading))

    for kind, value in _iter_blocks(markdown):
        if kind == "heading":
            # Start a fresh chunk at a section break, but only once the current
            # one has real content — otherwise a run of nested headings would
            # emit a chunk per line.
            if buffer_len >= chunk_size // 2:
                flush()
            heading = value
            if not buffer:
                buffer_heading = heading
            continue

        for piece in _split_oversized(value, chunk_size):
            if buffer and buffer_len + len(piece) + 2 > chunk_size:
                carry = _overlap_tail(buffer, chunk_overlap)
                flush()
                buffer_heading = heading
                if carry:
                    buffer.append(carry)
                    buffer_len += len(carry) + 2
            if not buffer:
                buffer_heading = heading
            buffer.append(piece)
            buffer_len += len(piece) + 2

    flush()
    return out


def _iter_blocks(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into ``("heading", text)`` and ``("block", text)`` items."""
    blocks: list[tuple[str, str]] = []
    current: list[str] = []

    def close() -> None:
        if current:
            text = "\n".join(current).strip()
            if text:
                blocks.append(("block", text))
            current.clear()

    for line in markdown.splitlines():
        match = HEADING.match(line.strip())
        if match:
            close()
            blocks.append(("heading", match.group("text").strip()))
            continue
        if not line.strip():
            close()
            continue
        current.append(line)

    close()
    return blocks


def _split_oversized(text: str, chunk_size: int) -> list[str]:
    """Break a single over-long block on sentence, then whitespace, boundaries."""
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(text):
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > chunk_size:
            pieces.append(current.strip())
            current = sentence
        elif len(sentence) > chunk_size:
            # A single sentence longer than a chunk (dense tables, no punctuation).
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(_hard_wrap(sentence, chunk_size))
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        pieces.append(current.strip())
    return [p for p in pieces if p]


def _hard_wrap(text: str, chunk_size: int) -> list[str]:
    pieces: list[str] = []
    remaining = text
    while len(remaining) > chunk_size:
        cut = remaining.rfind(" ", 0, chunk_size)
        if cut <= 0:
            cut = chunk_size
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _overlap_tail(buffer: list[str], chunk_overlap: int) -> str:
    """The last ``chunk_overlap`` characters of the buffer, cut at a word boundary."""
    if chunk_overlap <= 0:
        return ""
    joined = "\n\n".join(buffer)
    if len(joined) <= chunk_overlap:
        return joined
    tail = joined[-chunk_overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail
