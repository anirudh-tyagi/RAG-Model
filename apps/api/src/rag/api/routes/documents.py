"""Document upload, listing, live progress and deletion."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from rag.api.deps import RegistryDep, SettingsDep, StoreDep
from rag.api.sse import KEEPALIVE, SSE_HEADERS, event
from rag.ingest.pipeline import remove_document_files
from rag.jobs.queue import enqueue_ingest
from rag.logging import get_logger
from rag.registry import DocumentRegistry
from rag.schemas import Document, DocumentOut, IngestStage, ProgressEvent, UploadAccepted

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger(__name__)

READ_CHUNK = 1024 * 1024
PDF_MAGIC = b"%PDF"
#: How long to wait between keepalives on an idle progress stream.
KEEPALIVE_SECONDS = 15.0
TERMINAL_STAGES = {IngestStage.READY, IngestStage.FAILED}

PdfUpload = Annotated[UploadFile, File(description="The PDF to ingest")]

#: Starlette renamed the 413 constant; the integer works across versions.
HTTP_413 = 413


def _safe_basename(raw: str | None) -> str:
    """Reduce a client-supplied filename to a bare basename, for display only.

    ``Path(x).name`` strips only the *host* OS's separator, so a Windows client
    sending ``..\\..\\evil.pdf`` would keep its backslashes on a POSIX server.
    Normalising both separators first means the stored name is always clean,
    whatever the client's platform.
    """
    candidate = (raw or "document.pdf").replace("\\", "/")
    return Path(candidate).name or "document.pdf"


@router.post("", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: PdfUpload,
    settings: SettingsDep,
    registry: RegistryDep,
) -> UploadAccepted:
    """Accept a PDF and queue it for ingestion.

    The upload is stored as ``<uuid>.pdf`` and the original name kept only as
    metadata. Nothing user-supplied ever reaches a filesystem path, so the
    traversal hole in the old ``PDF_FOLDER / file.filename`` is closed by
    construction rather than by trying to sanitise the name.
    """
    settings.ensure_dirs()

    original_name = _safe_basename(file.filename)
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF files are accepted.")

    document = Document(
        filename=original_name,
        title=Path(original_name).stem.replace("_", " ").strip() or original_name,
        size_bytes=0,
    )
    target = settings.upload_dir / f"{document.id}.pdf"

    try:
        size = await _store_upload(file, target, settings.max_upload_bytes, settings.max_upload_mb)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        log.exception("upload_failed", filename=original_name)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Could not store upload: {exc}"
        ) from exc

    document.size_bytes = size
    await registry.save(document)

    try:
        await enqueue_ingest(document.id)
    except Exception as exc:
        log.exception("enqueue_failed", doc_id=document.id)
        await registry.fail(document, f"Could not queue ingestion: {exc}")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Ingestion queue is unavailable. Is Redis running?",
        ) from exc

    return UploadAccepted(document=DocumentOut.of(document))


async def _store_upload(file: UploadFile, target: Path, max_bytes: int, max_mb: int) -> int:
    """Stream the upload to disk, validating the header and enforcing the size cap.

    Streamed in chunks so an oversized file is rejected partway through rather
    than after the whole thing is buffered in memory.
    """
    size = 0
    with target.open("wb") as sink:
        first = True
        while chunk := await file.read(READ_CHUNK):
            if first:
                if not chunk.startswith(PDF_MAGIC):
                    raise HTTPException(
                        status.HTTP_400_BAD_REQUEST,
                        "That file is not a valid PDF (bad header).",
                    )
                first = False
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(HTTP_413, f"File exceeds the {max_mb}MB limit.")
            sink.write(chunk)

    if size == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    return size


@router.get("", response_model=list[DocumentOut])
async def list_documents(registry: RegistryDep) -> list[DocumentOut]:
    return [DocumentOut.of(doc) for doc in await registry.list_all()]


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(doc_id: str, registry: RegistryDep) -> DocumentOut:
    document = await registry.get(doc_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.")
    return DocumentOut.of(document)


@router.get("/{doc_id}/events")
async def document_events(
    doc_id: str, request: Request, registry: RegistryDep
) -> StreamingResponse:
    """Live ingestion progress for one document, as SSE.

    This is what the upload page waits on, replacing the old front end's
    hardcoded 1.5 second delay and unconditional redirect. Current state is sent
    immediately so a client that connects late still renders correctly.
    """
    document = await registry.get(doc_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.")

    return StreamingResponse(
        _progress_stream(document, request, registry),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _progress_stream(
    document: Document, request: Request, registry: DocumentRegistry
) -> AsyncIterator[str]:
    yield event("progress", ProgressEvent(document=DocumentOut.of(document)))
    if document.stage in TERMINAL_STAGES:
        return

    updates = registry.watch(document.id)
    try:
        while not await request.is_disconnected():
            try:
                update = await asyncio.wait_for(updates.__anext__(), timeout=KEEPALIVE_SECONDS)
            except TimeoutError:
                # Nothing changed; keep intermediaries from closing the stream.
                yield KEEPALIVE
                continue
            except StopAsyncIteration:  # pragma: no cover - pubsub closed
                return

            yield event("progress", ProgressEvent(document=DocumentOut.of(update)))
            if update.stage in TERMINAL_STAGES:
                return
    finally:
        await updates.aclose()


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    registry: RegistryDep,
    store: StoreDep,
    settings: SettingsDep,
) -> None:
    """Remove a document's vectors, files and registry entry."""
    if await registry.get(doc_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.")

    await store.delete_document(doc_id)
    await remove_document_files(doc_id, settings)
    await registry.delete(doc_id)
    log.info("document_deleted", doc_id=doc_id)
