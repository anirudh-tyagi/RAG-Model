"""SSE framing.

Framing bugs are quiet: the browser just never fires the handler. Worth pinning.
"""

from __future__ import annotations

import json

from rag.api.sse import KEEPALIVE, SSE_HEADERS, event
from rag.schemas import DoneEvent, ErrorEvent, MetaEvent, Source, SourcesEvent, TokenEvent


def test_frame_has_named_event_and_data_lines() -> None:
    frame = event("token", TokenEvent(text="hello"))

    assert frame.startswith("event: token\ndata: ")
    assert frame.endswith("\n\n")


def test_data_payload_is_valid_json() -> None:
    frame = event("meta", MetaEvent(conversation_id="abc"))
    data_line = frame.splitlines()[1].removeprefix("data: ")

    assert json.loads(data_line)["conversation_id"] == "abc"


def test_multiline_text_stays_on_one_data_line() -> None:
    """A raw newline in the payload would terminate the frame early."""
    frame = event("token", TokenEvent(text="line one\nline two"))

    # Exactly two lines of content plus the blank terminator.
    assert len([line for line in frame.split("\n") if line]) == 2
    assert "\\n" in frame


def test_sources_frame_carries_citation_numbers() -> None:
    sources = [
        Source(
            n=1,
            chunk_id="c1",
            doc_id="d1",
            doc_title="Report",
            page=12,
            heading="Revenue",
            excerpt="Revenue rose",
            score=0.9,
        )
    ]
    frame = event("sources", SourcesEvent(sources=sources))
    payload = json.loads(frame.splitlines()[1].removeprefix("data: "))

    assert payload["sources"][0]["n"] == 1
    assert payload["sources"][0]["page"] == 12


def test_done_and_error_frames_are_distinct_event_names() -> None:
    assert event("done", DoneEvent(took_ms=12.5)).startswith("event: done")
    assert event("error", ErrorEvent(message="nope")).startswith("event: error")


def test_keepalive_is_a_comment_frame() -> None:
    # Comment frames are ignored by EventSource but keep proxies from timing out.
    assert KEEPALIVE.startswith(":")
    assert KEEPALIVE.endswith("\n\n")


def test_headers_disable_caching_and_proxy_buffering() -> None:
    assert "no-cache" in SSE_HEADERS["Cache-Control"]
    assert SSE_HEADERS["X-Accel-Buffering"] == "no"
