"""The ingestion pipeline.

This replaces the old ``run_processing_pipeline``, which shelled out to three
standalone scripts via ``subprocess.run(..., capture_output=True)``. That design
had a few problems this one fixes:

* the three scripts passed data by convention — ``Base.py`` wrote a directory
  named after the PDF stem *relative to the process CWD*, and the next script
  had to guess the same path — so running the server from a different directory
  silently broke ingestion;
* every model was reloaded from scratch in each subprocess;
* ``capture_output=True`` meant progress and errors went nowhere the user could
  see, and a failure surfaced only as a chat reply saying "still processing".

Now it is one in-process async pipeline with shared models and a stage reported
to Redis (and onward to the browser) at each step.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

from rag import obs
from rag.chat.llm import LlmClient, get_llm
from rag.config import Settings, get_settings
from rag.ingest.caption import caption_images
from rag.ingest.chunk import chunk_pages
from rag.ingest.parse import get_parser
from rag.logging import get_logger
from rag.registry import DocumentRegistry, get_registry
from rag.retrieval.embeddings import Embedder, get_embedder
from rag.retrieval.store import VectorStore, get_store
from rag.schemas import Document, IngestStage

log = get_logger(__name__)

ProgressReporter = Callable[[IngestStage, str], Awaitable[None]]

EMBED_BATCH = 32


class IngestError(RuntimeError):
    pass


class IngestPipeline:
    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        store: VectorStore,
        embedder: Embedder,
        llm: LlmClient,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._store = store
        self._embedder = embedder
        self._llm = llm

    @obs.observe("ingest")
    async def run(self, document: Document) -> Document:
        pdf_path = self._settings.upload_dir / f"{document.id}.pdf"
        if not pdf_path.is_file():
            raise IngestError(f"uploaded file is missing: {pdf_path}")

        artifact_dir = self._settings.artifact_dir / document.id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        image_dir = artifact_dir / "images"

        try:
            # --- 1. parse ------------------------------------------------
            await self._registry.advance(
                document, IngestStage.PARSING, "Extracting text and figures from the PDF"
            )
            parser = get_parser(self._settings)
            # Both backends are synchronous and CPU-heavy; keep them off the loop.
            parsed = await asyncio.to_thread(parser.parse, pdf_path, image_dir)
            if not parsed.pages:
                raise IngestError(
                    "No extractable text found. If this is a scanned PDF, it needs OCR "
                    "(try PDF_PARSER=docling)."
                )
            document.pages = parsed.page_count
            log.info(
                "parsed_pdf",
                doc_id=document.id,
                parser=parser.name,
                pages=parsed.page_count,
                images=len(parsed.images),
            )

            # --- 2. caption figures --------------------------------------
            captions = []
            if self._settings.caption_images and parsed.images:
                await self._registry.advance(
                    document,
                    IngestStage.CAPTIONING,
                    f"Describing {len(parsed.images)} figures",
                )
                captions = await caption_images(
                    parsed,
                    image_dir,
                    self._settings,
                    self._llm,
                    on_progress=self._caption_progress(document, len(parsed.images)),
                )
                document.captioned_images = len(captions)
            else:
                await caption_images(parsed, image_dir, self._settings, self._llm)

            self._write_markdown(artifact_dir, parsed)

            # --- 3. chunk -------------------------------------------------
            await self._registry.advance(
                document, IngestStage.CHUNKING, "Splitting the document into passages"
            )
            chunks = chunk_pages(
                document,
                parsed.pages,
                captions,
                self._settings.chunk_size,
                self._settings.chunk_overlap,
            )
            if not chunks:
                raise IngestError("document produced no indexable passages")

            # --- 4. embed and index ---------------------------------------
            await self._registry.advance(
                document, IngestStage.EMBEDDING, f"Embedding {len(chunks)} passages"
            )
            await self._store.ensure_collection(await self._embedder.dim())
            # Re-ingesting the same document replaces its vectors rather than
            # accumulating duplicates.
            await self._store.delete_document(document.id)

            for start in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[start : start + EMBED_BATCH]
                dense, sparse = await self._embedder.encode_passages([c.text for c in batch])
                await self._store.upsert(batch, dense, sparse)
                done = min(start + EMBED_BATCH, len(chunks))
                await self._registry.advance(
                    document,
                    IngestStage.EMBEDDING,
                    f"Embedded {done} of {len(chunks)} passages",
                )

            # --- 5. done ---------------------------------------------------
            await self._registry.advance(
                document,
                IngestStage.READY,
                "Ready to answer questions",
                chunk_count=len(chunks),
                error=None,
            )
            log.info("ingest_complete", doc_id=document.id, chunks=len(chunks))
            return document

        except Exception as exc:
            log.exception("ingest_failed", doc_id=document.id)
            await self._registry.fail(document, str(exc))
            raise

    def _caption_progress(self, document: Document, total: int) -> Callable[[int, int], None]:
        """Bridge the captioner's sync callback onto the event loop."""
        loop = asyncio.get_running_loop()

        def report(done: int, _total: int) -> None:
            loop.create_task(  # noqa: RUF006 - fire and forget status update
                self._registry.advance(
                    document,
                    IngestStage.CAPTIONING,
                    f"Described {done} of {total} figures",
                )
            )

        return report

    @staticmethod
    def _write_markdown(artifact_dir: Path, parsed: object) -> None:
        """Persist the parsed markdown for debugging and manual inspection."""
        pages = getattr(parsed, "pages", [])
        body = "\n\n".join(f"<!-- page {p.page} -->\n\n{p.markdown}" for p in pages)
        (artifact_dir / "document.md").write_text(body, encoding="utf-8")


def get_pipeline() -> IngestPipeline:
    return IngestPipeline(get_settings(), get_registry(), get_store(), get_embedder(), get_llm())


async def remove_document_files(doc_id: str, settings: Settings | None = None) -> None:
    """Delete the upload and every derived artifact for a document."""
    settings = settings or get_settings()
    pdf_path = settings.upload_dir / f"{doc_id}.pdf"
    pdf_path.unlink(missing_ok=True)
    shutil.rmtree(settings.artifact_dir / doc_id, ignore_errors=True)
