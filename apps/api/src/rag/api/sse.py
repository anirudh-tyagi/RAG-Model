"""Server-sent event framing.

SSE rather than WebSockets: the traffic is one-directional, it survives ordinary
HTTP proxies, and the browser's ``EventSource`` semantics (named events,
automatic reconnect) are exactly what a token stream and a progress feed want.
"""

from __future__ import annotations

from pydantic import BaseModel

#: Comment frame. Keeps intermediaries from timing out an idle stream, and is
#: ignored by EventSource and by our own reader.
KEEPALIVE = ": keepalive\n\n"


def event(name: str, payload: BaseModel) -> str:
    """Frame one named event. ``data`` is always a single-line JSON object."""
    return f"event: {name}\ndata: {payload.model_dump_json()}\n\n"


SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Nginx buffers streamed responses by default, which defeats the point.
    "X-Accel-Buffering": "no",
}
