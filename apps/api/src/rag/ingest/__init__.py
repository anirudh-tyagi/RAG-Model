"""Ingestion: PDF → markdown pages → figure captions → chunks → vectors."""

from rag.ingest.chunk import chunk_pages
from rag.ingest.parse import ImageRef, ParsedDocument, ParsedPage, get_parser
from rag.ingest.pipeline import IngestPipeline, ProgressReporter

__all__ = [
    "ImageRef",
    "IngestPipeline",
    "ParsedDocument",
    "ParsedPage",
    "ProgressReporter",
    "chunk_pages",
    "get_parser",
]
