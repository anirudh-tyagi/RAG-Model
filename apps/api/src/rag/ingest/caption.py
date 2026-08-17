"""Figure captioning with a vision model.

The old ``Image-Testo.py`` captioned images strictly one at a time inside a
``re.sub`` callback, so a paper with 40 figures meant 40 serial round trips to a
cloud VLM. Here the calls run concurrently under a semaphore, are capped per
document, and a failure on one figure no longer poisons the whole ingest.

Captions become their own chunks (tagged ``is_figure``) so "what does figure 3
show?" can hit the caption directly, while the body markdown keeps a short
inline marker so the surrounding prose still reads continuously.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from rag.chat.llm import LlmClient, LlmError
from rag.config import Settings
from rag.ingest.parse import IMAGE_LINK, ParsedDocument
from rag.logging import get_logger

log = get_logger(__name__)

CAPTION_PROMPT = (
    "Describe this figure from a document concisely and precisely. "
    "Report any numerical values, axis labels, units, and trends exactly as shown. "
    "If it is a table, summarise its columns and the notable values. "
    "Do not speculate about anything not visible."
)

MARKER_CHARS = 90


@dataclass(slots=True)
class Caption:
    target: str
    text: str
    page: int | None = None


async def caption_images(
    parsed: ParsedDocument,
    image_dir: Path,
    settings: Settings,
    llm: LlmClient,
    on_progress: object = None,
) -> list[Caption]:
    """Caption every resolvable figure, rewriting ``parsed`` markdown in place.

    Returns the captions produced; images that could not be resolved or
    captioned are simply dropped from the markdown.
    """
    targets = _unique_targets(parsed, image_dir)
    if not settings.caption_images or not targets:
        _strip_image_links(parsed)
        return []

    if len(targets) > settings.max_captions_per_doc:
        log.info(
            "capping_captions",
            found=len(targets),
            cap=settings.max_captions_per_doc,
        )
        targets = targets[: settings.max_captions_per_doc]

    semaphore = asyncio.Semaphore(settings.caption_concurrency)
    completed = 0
    total = len(targets)
    lock = asyncio.Lock()

    async def describe(target: str, path: Path, page: int | None) -> Caption | None:
        nonlocal completed
        async with semaphore:
            try:
                text = await llm.describe_image(path, CAPTION_PROMPT)
            except LlmError as exc:
                log.warning("caption_failed", target=target, error=str(exc))
                text = ""
            async with lock:
                completed += 1
                if callable(on_progress):
                    on_progress(completed, total)
            if not text:
                return None
            return Caption(target=target, text=text, page=page)

    results = await asyncio.gather(
        *(describe(target, path, page) for target, path, page in targets)
    )
    captions = [c for c in results if c is not None]
    log.info("captioned_images", requested=total, succeeded=len(captions))

    _inline_markers(parsed, {c.target: c for c in captions})
    return captions


def _unique_targets(parsed: ParsedDocument, image_dir: Path) -> list[tuple[str, Path, int | None]]:
    """Every distinct image link in the markdown that resolves to a real file."""
    pages_by_target: dict[str, int] = {}
    for page in parsed.pages:
        for match in IMAGE_LINK.finditer(page.markdown):
            pages_by_target.setdefault(match.group("target"), page.page)

    resolved: list[tuple[str, Path, int | None]] = []
    for target, page_no in pages_by_target.items():
        path = parsed.resolve_image(target, image_dir)
        if path is None:
            log.debug("image_unresolved", target=target)
            continue
        resolved.append((target, path, page_no))
    return resolved


def _inline_markers(parsed: ParsedDocument, captions: dict[str, Caption]) -> None:
    """Replace each image link with a one-line marker (or remove it)."""

    def replace(match: object) -> str:
        target = match.group("target")  # type: ignore[attr-defined]
        caption = captions.get(target)
        if caption is None:
            return ""
        summary = " ".join(caption.text.split())
        if len(summary) > MARKER_CHARS:
            summary = summary[:MARKER_CHARS].rstrip() + "…"
        return f"[Figure: {summary}]"

    for page in parsed.pages:
        page.markdown = IMAGE_LINK.sub(replace, page.markdown)


def _strip_image_links(parsed: ParsedDocument) -> None:
    for page in parsed.pages:
        page.markdown = IMAGE_LINK.sub("", page.markdown)
