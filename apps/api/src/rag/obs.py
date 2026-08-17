"""Optional tracing.

Langfuse gives per-query visibility into retrieval and generation; OTel covers
HTTP spans. Both are optional: with no keys configured, ``observe`` is an
identity decorator and ``span`` is a no-op context manager, so nothing in the
codebase needs to branch on whether tracing is enabled.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from rag.config import get_settings
from rag.logging import get_logger

log = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_client: Any | None = None
_initialised = False


def _get_client() -> Any | None:
    """Lazily build the Langfuse client, tolerating an uninstalled SDK."""
    global _client, _initialised
    if _initialised:
        return _client
    _initialised = True

    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning(
            "langfuse_keys_set_but_sdk_missing",
            hint="install with: uv sync --extra observability",
        )
        return None

    try:
        _client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),  # type: ignore[union-attr]
            secret_key=settings.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
            host=settings.langfuse_host,
        )
        log.info("langfuse_enabled", host=settings.langfuse_host)
    except Exception as exc:  # pragma: no cover - depends on remote service
        log.warning("langfuse_init_failed", error=str(exc))
        _client = None
    return _client


def observe(name: str, **static_meta: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Trace a function as a Langfuse span, if tracing is on.

    Works for sync functions, coroutines and async generators alike — the
    wrapper simply delegates to Langfuse's own decorator, which handles all
    three, and falls back to the bare function when tracing is off.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        client = _get_client()
        if client is None:
            return fn
        try:
            from langfuse import observe as langfuse_observe
        except ImportError:  # pragma: no cover - guarded by _get_client
            return fn
        wrapped = langfuse_observe(name=name, **static_meta)(fn)
        return functools.wraps(fn)(wrapped)

    return decorator


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Trace a block of work. Yields the span, or ``None`` when tracing is off."""
    client = _get_client()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_span(name=name, input=attributes or None) as current:
            yield current
    except Exception as exc:  # pragma: no cover - never break the request path
        log.debug("trace_span_failed", name=name, error=str(exc))
        yield None


def score(name: str, value: float, comment: str | None = None) -> None:
    """Attach a numeric score to the current trace (used by the eval harness)."""
    client = _get_client()
    if client is None:
        return
    try:
        client.score_current_trace(name=name, value=value, comment=comment)
    except Exception as exc:  # pragma: no cover
        log.debug("trace_score_failed", name=name, error=str(exc))


def flush() -> None:
    """Force-send buffered traces. Call before a short-lived process exits."""
    client = _get_client()
    if client is not None:
        with contextlib.suppress(Exception):  # pragma: no cover
            client.flush()


def instrument_fastapi(app: Any) -> None:
    """Add OTel HTTP spans when an OTLP endpoint is configured."""
    if not get_settings().otel_exporter_otlp_endpoint:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        log.warning("otel_endpoint_set_but_sdk_missing")
        return
    FastAPIInstrumentor.instrument_app(app)
    log.info("otel_instrumentation_enabled")
