"""Wire contracts shared by the HTTP API, the worker and the eval harness."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    # Dashed UUID form: Qdrant only accepts UUIDs or unsigned ints as point IDs.
    return str(uuid4())


class IngestStage(StrEnum):
    """Ordered pipeline stages. Progress is derived from position in this list."""

    QUEUED = "queued"
    PARSING = "parsing"
    CAPTIONING = "captioning"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


#: Fraction complete when a stage *starts*. Used for the progress bar.
STAGE_PROGRESS: dict[IngestStage, float] = {
    IngestStage.QUEUED: 0.0,
    IngestStage.PARSING: 0.1,
    IngestStage.CAPTIONING: 0.35,
    IngestStage.CHUNKING: 0.7,
    IngestStage.EMBEDDING: 0.8,
    IngestStage.READY: 1.0,
    IngestStage.FAILED: 1.0,
}


class Document(BaseModel):
    """A user-uploaded source document and its ingestion state."""

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(default_factory=new_id)
    filename: str
    title: str
    size_bytes: int
    stage: IngestStage = IngestStage.QUEUED
    detail: str = "Waiting for a worker"
    pages: int | None = None
    chunk_count: int | None = None
    captioned_images: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @property
    def progress(self) -> float:
        return STAGE_PROGRESS[self.stage]

    @property
    def is_ready(self) -> bool:
        return self.stage is IngestStage.READY


class DocumentOut(BaseModel):
    """Document as returned to the browser (progress flattened in)."""

    id: str
    filename: str
    title: str
    size_bytes: int
    stage: IngestStage
    detail: str
    progress: float
    pages: int | None
    chunk_count: int | None
    captioned_images: int
    error: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, doc: Document) -> DocumentOut:
        return cls(**doc.model_dump(), progress=doc.progress)


class UploadAccepted(BaseModel):
    document: DocumentOut
    message: str = "Upload accepted. Ingestion is running in the background."


class Chunk(BaseModel):
    """A retrievable passage. One Qdrant point per chunk."""

    id: str = Field(default_factory=new_id)
    doc_id: str
    doc_title: str
    filename: str
    chunk_index: int
    text: str
    page: int | None = None
    heading: str | None = None
    #: Chunk was synthesised from a figure caption rather than body text.
    is_figure: bool = False

    def to_payload(self) -> dict[str, object]:
        return self.model_dump(exclude={"id"})


class Source(BaseModel):
    """A cited chunk, as surfaced to the user."""

    n: int
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int | None
    heading: str | None
    excerpt: str
    score: float


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str
    sources: list[Source] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None
    #: Restrict retrieval to these documents. Empty/omitted searches all of them.
    doc_ids: list[str] = Field(default_factory=list)


class Conversation(BaseModel):
    id: str = Field(default_factory=new_id)
    title: str = "New conversation"
    doc_ids: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class SearchResult(BaseModel):
    """A retrieval hit, before it becomes a citation."""

    chunk: Chunk
    #: Reciprocal-rank-fusion score from Qdrant's hybrid query.
    score: float
    #: Cross-encoder score, when reranking ran. Results are ordered by this.
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    took_ms: float


class TranscriptionOut(BaseModel):
    text: str
    language: str | None = None
    duration_s: float | None = None


# --- Server-sent event payloads -----------------------------------------------
# Each is emitted as `event: <type>` with the model JSON as `data:`.


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class SourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    sources: list[Source]


class MetaEvent(BaseModel):
    type: Literal["meta"] = "meta"
    conversation_id: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    took_ms: float


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


class ProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    document: DocumentOut
