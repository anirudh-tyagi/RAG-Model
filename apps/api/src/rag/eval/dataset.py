"""Golden question sets, as JSONL.

Relevance is judged at *page* level rather than chunk level. Chunk ids change
every time chunking parameters are touched, which would invalidate the whole
dataset on any tuning change; page numbers are stable properties of the source
document, so the same golden set stays valid while you tune chunk size, the
embedding model or the reranker — which is the entire point of having it.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One question plus what a good answer and good retrieval look like."""

    id: str
    question: str
    #: Strings the answer must contain (case-insensitive). Use exact figures.
    expected_keywords: list[str] = Field(default_factory=list)
    #: Pages that genuinely contain the answer. Drives recall/MRR/nDCG.
    relevant_pages: list[int] = Field(default_factory=list)
    #: Optional: restrict retrieval to these documents.
    doc_ids: list[str] = Field(default_factory=list)
    #: Optional free-text note for whoever reads the report.
    note: str | None = None


def load_cases(path: Path) -> list[EvalCase]:
    """Read a JSONL golden set, skipping blank lines and ``#`` comments."""
    if not path.is_file():
        raise FileNotFoundError(f"no eval dataset at {path}")

    cases: list[EvalCase] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(EvalCase.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc

    if not cases:
        raise ValueError(f"{path} contains no cases")
    return cases


def write_cases(path: Path, cases: list[EvalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(case.model_dump_json(exclude_none=True) for case in cases)
    path.write_text(body + "\n", encoding="utf-8")
