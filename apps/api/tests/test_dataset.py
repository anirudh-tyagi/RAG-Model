"""Golden-set loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.eval.dataset import EvalCase, load_cases, write_cases

VALID = (
    '{"id": "q1", "question": "What is revenue?", '
    '"expected_keywords": ["5.1B"], "relevant_pages": [12]}'
)


def test_loads_a_single_case(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(VALID + "\n", encoding="utf-8")

    cases = load_cases(path)

    assert len(cases) == 1
    assert cases[0].id == "q1"
    assert cases[0].relevant_pages == [12]


def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(f"# a comment\n\n{VALID}\n\n  \n", encoding="utf-8")

    assert len(load_cases(path)) == 1


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "nope.jsonl")


def test_file_with_only_comments_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text("# nothing but comments\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no cases"):
        load_cases(path)


def test_malformed_line_reports_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(f"{VALID}\nnot json at all\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r":2:"):
        load_cases(path)


def test_missing_required_field_reports_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"question": "no id here"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r":1:"):
        load_cases(path)


def test_optional_fields_default_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "q1", "question": "Anything?"}\n', encoding="utf-8")

    case = load_cases(path)[0]

    assert case.expected_keywords == []
    assert case.relevant_pages == []
    assert case.doc_ids == []


def test_write_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "golden.jsonl"
    original = [
        EvalCase(id="q1", question="First?", relevant_pages=[1]),
        EvalCase(id="q2", question="Second?", expected_keywords=["yes"]),
    ]

    write_cases(path, original)
    loaded = load_cases(path)

    assert [c.id for c in loaded] == ["q1", "q2"]
    assert loaded[0].relevant_pages == [1]
    assert loaded[1].expected_keywords == ["yes"]


def test_shipped_example_dataset_is_valid() -> None:
    """The example that ships with the repo must actually parse."""
    example = Path(__file__).resolve().parents[3] / "evals" / "golden.example.jsonl"

    cases = load_cases(example)

    assert len(cases) >= 3
    assert all(case.id for case in cases)
