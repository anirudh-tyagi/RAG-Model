"""Typed application settings, loaded once from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Everything configurable, in one place.

    Replaces the module-level constants that used to be scattered across
    ``rag_components.py``, ``Emmbed.py`` and ``Image-Testo.py`` — where the
    embedding model name was duplicated in two files and could drift.
    """

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM -------------------------------------------------------------
    ollama_api_key: SecretStr | None = None
    ollama_base_url: str = "https://ollama.com"
    llm_model: str = "gpt-oss:120b"
    vision_model: str = "qwen3-vl:235b-cloud"
    llm_temperature: float = 0.2
    llm_timeout_s: float = 120.0

    # --- Vector store ----------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    # --- Queue / shared state --------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- Embeddings / reranking ------------------------------------------
    dense_model: str = "BAAI/bge-large-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_enabled: bool = True

    # --- Retrieval -------------------------------------------------------
    retrieval_candidates: int = Field(default=40, ge=1, le=500)
    retrieval_top_k: int = Field(default=6, ge=1, le=50)
    chunk_size: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    history_turns: int = Field(default=6, ge=0, le=50)

    # --- Ingestion -------------------------------------------------------
    pdf_parser: Literal["pymupdf", "docling"] = "pymupdf"
    caption_images: bool = True
    max_captions_per_doc: int = 60
    caption_concurrency: int = Field(default=4, ge=1, le=32)

    # --- Speech to text --------------------------------------------------
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_task: Literal["transcribe", "translate"] = "translate"

    # --- Observability ---------------------------------------------------
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    otel_exporter_otlp_endpoint: str | None = None
    log_level: str = "INFO"

    # --- API -------------------------------------------------------------
    data_dir: Path = Path("./data")
    max_upload_mb: int = Field(default=50, ge=1, le=1000)
    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def artifact_dir(self) -> Path:
        """Where parsed markdown and extracted images land, per document."""
        return self.data_dir / "artifacts"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
