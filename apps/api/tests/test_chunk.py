"""Chunking behaviour — the part of the old pipeline that lost the most information."""

from __future__ import annotations

from rag.ingest.caption import Caption
from rag.ingest.chunk import _split_oversized, chunk_pages
from rag.ingest.parse import ParsedPage
from rag.schemas import Document


def make_document() -> Document:
    return Document(filename="paper.pdf", title="Paper", size_bytes=1024)


def test_heading_is_recorded_and_prefixed() -> None:
    pages = [
        ParsedPage(
            page=3,
            markdown="# Results\n\nAccuracy reached 91.4 percent on the held-out split.",
        )
    ]
    chunks = chunk_pages(make_document(), pages, [], chunk_size=400, chunk_overlap=0)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.heading == "Results"
    # The heading is prefixed into the text so the embedding sees the topic,
    # not just the sentence.
    assert chunk.text.startswith("Results")
    assert "91.4" in chunk.text
    assert chunk.page == 3


def test_page_number_is_preserved_per_chunk() -> None:
    pages = [
        ParsedPage(page=1, markdown="Intro paragraph one."),
        ParsedPage(page=7, markdown="Later paragraph two."),
    ]
    chunks = chunk_pages(make_document(), pages, [], chunk_size=400, chunk_overlap=0)

    assert [c.page for c in chunks] == [1, 7]


def test_chunk_indices_are_sequential_across_pages() -> None:
    pages = [
        ParsedPage(page=1, markdown="alpha " * 100),
        ParsedPage(page=2, markdown="beta " * 100),
    ]
    chunks = chunk_pages(make_document(), pages, [], chunk_size=200, chunk_overlap=0)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_long_page_splits_into_multiple_chunks_with_overlap() -> None:
    body = " ".join(f"sentence{n} about the topic." for n in range(60))
    pages = [ParsedPage(page=1, markdown=f"## Method\n\n{body}")]

    chunks = chunk_pages(make_document(), pages, [], chunk_size=300, chunk_overlap=60)

    assert len(chunks) > 1
    assert all(c.heading == "Method" for c in chunks)
    # Consecutive chunks should share some text, so a fact spanning the boundary
    # survives in at least one of them.
    first_tail_words = set(chunks[0].text.split()[-8:])
    assert first_tail_words & set(chunks[1].text.split())


def test_no_overlap_when_configured_to_zero() -> None:
    body = " ".join(f"sentence{n} about the topic." for n in range(60))
    pages = [ParsedPage(page=1, markdown=body)]

    chunks = chunk_pages(make_document(), pages, [], chunk_size=300, chunk_overlap=0)

    assert len(chunks) > 1
    joined = sum(len(c.text) for c in chunks)
    # Without overlap the pieces should not materially exceed the source length.
    assert joined <= len(body) + 4 * len(chunks)


def test_consecutive_headings_do_not_emit_empty_chunks() -> None:
    pages = [
        ParsedPage(
            page=1,
            markdown="# Title\n\n## Subtitle\n\n### Deeper\n\nOnly this line has content.",
        )
    ]
    chunks = chunk_pages(make_document(), pages, [], chunk_size=500, chunk_overlap=0)

    assert len(chunks) == 1
    assert "Only this line has content." in chunks[0].text
    # The innermost heading in effect is the one recorded.
    assert chunks[0].heading == "Deeper"


def test_figure_captions_become_their_own_chunks() -> None:
    captions = [Caption(target="fig1.png", text="Bar chart: revenue 4.2B in 2023.", page=12)]
    chunks = chunk_pages(
        make_document(),
        [ParsedPage(page=1, markdown="Body text.")],
        captions,
        chunk_size=500,
        chunk_overlap=0,
    )

    figures = [c for c in chunks if c.is_figure]
    assert len(figures) == 1
    assert figures[0].page == 12
    assert "4.2B" in figures[0].text
    assert figures[0].heading == "Figures"


def test_empty_page_produces_no_chunks() -> None:
    pages = [ParsedPage(page=1, markdown="   \n\n  \n")]
    assert chunk_pages(make_document(), pages, [], chunk_size=400, chunk_overlap=0) == []


def test_split_oversized_hard_wraps_unpunctuated_text() -> None:
    # A dense table row with no sentence boundaries at all.
    text = "col " * 500
    pieces = _split_oversized(text, 200)

    assert len(pieces) > 1
    assert all(len(piece) <= 200 for piece in pieces)


def test_split_oversized_leaves_short_text_alone() -> None:
    assert _split_oversized("short enough", 200) == ["short enough"]
