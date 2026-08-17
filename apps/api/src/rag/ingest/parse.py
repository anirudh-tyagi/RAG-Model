"""PDF → per-page markdown, plus extracted figure images.

Two backends behind one protocol:

* ``pymupdf`` (default) — PyMuPDF4LLM. No torch, starts instantly, handles the
  large majority of PDFs well.
* ``docling`` — IBM's converter, much better on complex layouts and tables, at
  the cost of a torch dependency and a slow first run while models download.

Both emit markdown **per page**, which the old pipeline did not: `Base.py`
produced one flat markdown blob, so retrieved passages could never tell the
user which page an answer came from.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rag.config import Settings, get_settings
from rag.logging import get_logger

log = get_logger(__name__)

IMAGE_LINK = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")


@dataclass(slots=True)
class ParsedPage:
    page: int
    markdown: str


@dataclass(slots=True)
class ImageRef:
    """An extracted figure. ``target`` is how the markdown refers to it."""

    target: str
    path: Path
    page: int | None = None


@dataclass(slots=True)
class ParsedDocument:
    pages: list[ParsedPage] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def resolve_image(self, target: str, image_dir: Path) -> Path | None:
        """Find the file a markdown image link points at.

        Backends disagree about whether the link is absolute, relative to the
        artifact directory, or a bare filename, so try each in turn rather than
        assuming one shape the way ``Image-Testo.py`` did.
        """
        for known in self.images:
            if known.target == target:
                return known.path if known.path.exists() else None

        candidates = (
            Path(target),
            image_dir / target,
            image_dir / Path(target).name,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None


class PdfParser(Protocol):
    name: str

    def parse(self, pdf_path: Path, image_dir: Path) -> ParsedDocument: ...


def _supported_kwargs(fn: Any, **kwargs: Any) -> dict[str, Any]:
    """Drop kwargs the installed version of ``fn`` doesn't accept.

    pymupdf4llm's signature has shifted across releases; this keeps a version
    bump from turning into a hard crash mid-ingest.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


class PyMuPDFParser:
    name = "pymupdf"

    def parse(self, pdf_path: Path, image_dir: Path) -> ParsedDocument:
        import pymupdf4llm

        image_dir.mkdir(parents=True, exist_ok=True)
        kwargs = _supported_kwargs(
            pymupdf4llm.to_markdown,
            page_chunks=True,
            write_images=True,
            image_path=str(image_dir),
            image_format="png",
            dpi=150,
            show_progress=False,
        )
        raw = pymupdf4llm.to_markdown(str(pdf_path), **kwargs)

        parsed = ParsedDocument()
        # page_chunks=True yields one dict per page; without it we get a single
        # string and lose page attribution.
        if isinstance(raw, str):
            log.warning("page_chunks_unsupported", parser=self.name)
            parsed.pages.append(ParsedPage(page=1, markdown=raw))
        else:
            for index, chunk in enumerate(raw, start=1):
                text = (chunk.get("text") or "").strip()
                page_no = int(chunk.get("metadata", {}).get("page", index) or index)
                if text:
                    parsed.pages.append(ParsedPage(page=page_no, markdown=text))

        parsed.images.extend(self._collect_images(parsed, image_dir))
        return parsed

    @staticmethod
    def _collect_images(parsed: ParsedDocument, image_dir: Path) -> list[ImageRef]:
        """Register every image link found in the markdown that exists on disk."""
        seen: set[str] = set()
        images: list[ImageRef] = []
        for page in parsed.pages:
            for match in IMAGE_LINK.finditer(page.markdown):
                target = match.group("target")
                if target in seen:
                    continue
                seen.add(target)
                for candidate in (Path(target), image_dir / target, image_dir / Path(target).name):
                    if candidate.is_file():
                        images.append(ImageRef(target=target, path=candidate, page=page.page))
                        break
        return images


class DoclingParser:
    name = "docling"

    def parse(self, pdf_path: Path, image_dir: Path) -> ParsedDocument:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        image_dir.mkdir(parents=True, exist_ok=True)

        options = PdfPipelineOptions()
        options.generate_picture_images = True
        options.images_scale = 2.0
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        document = converter.convert(str(pdf_path)).document

        parsed = ParsedDocument()
        page_numbers = sorted(document.pages) if document.pages else [1]
        for page_no in page_numbers:
            markdown = document.export_to_markdown(page_no=page_no).strip()
            if markdown:
                parsed.pages.append(ParsedPage(page=int(page_no), markdown=markdown))

        parsed.images.extend(self._export_pictures(document, image_dir))
        return parsed

    @staticmethod
    def _export_pictures(document: Any, image_dir: Path) -> list[ImageRef]:
        """Write Docling's in-memory picture objects out as PNGs we can caption."""
        images: list[ImageRef] = []
        for index, picture in enumerate(getattr(document, "pictures", []) or []):
            try:
                pil_image = picture.get_image(document)
            except Exception as exc:  # pragma: no cover - depends on document
                log.debug("docling_picture_unavailable", index=index, error=str(exc))
                continue
            if pil_image is None:
                continue
            filename = f"figure-{index:03d}.png"
            path = image_dir / filename
            pil_image.save(path, format="PNG")
            page = None
            provenance = getattr(picture, "prov", None) or []
            if provenance:
                page = getattr(provenance[0], "page_no", None)
            images.append(ImageRef(target=filename, path=path, page=page))
        return images


def get_parser(settings: Settings | None = None) -> PdfParser:
    settings = settings or get_settings()
    if settings.pdf_parser == "docling":
        try:
            import docling  # noqa: F401
        except ImportError:
            log.warning(
                "docling_not_installed_falling_back",
                hint="install with: uv sync --extra docling",
            )
            return PyMuPDFParser()
        return DoclingParser()
    return PyMuPDFParser()
