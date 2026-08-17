"""``rag-eval`` command line entry point."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag.chat.llm import get_llm
from rag.chat.service import get_chat_service
from rag.config import get_settings
from rag.eval.dataset import load_cases
from rag.eval.runner import EvalReport, run_eval
from rag.logging import configure_logging
from rag.registry import get_registry
from rag.retrieval.searcher import get_searcher

app = typer.Typer(help="Evaluate retrieval and answer quality.", no_args_is_help=True)
console = Console()

DEFAULT_DATASET = Path("evals/golden.jsonl")


@app.command()
def run(
    dataset: Path = typer.Option(DEFAULT_DATASET, "--dataset", "-d", help="JSONL golden set"),
    top_k: int | None = typer.Option(None, "--top-k", "-k", help="Passages per question"),
    retrieval_only: bool = typer.Option(
        False, "--retrieval-only", help="Skip generation: fast, free, deterministic"
    ),
    judge: bool = typer.Option(False, "--judge", help="Add an LLM-as-judge groundedness score"),
    concurrency: int = typer.Option(3, "--concurrency", "-c", min=1, max=16),
    json_out: Path | None = typer.Option(None, "--json", help="Write the full report here"),
    fail_under_recall: float | None = typer.Option(
        None,
        "--fail-under-recall",
        help="Exit non-zero if mean recall falls below this. For CI gating.",
    ),
) -> None:
    """Score a golden set against the live index."""
    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        cases = load_cases(dataset)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    console.print(
        f"Scoring [bold]{len(cases)}[/bold] cases · "
        f"{'retrieval only' if retrieval_only else 'retrieval + generation'}"
    )

    report = asyncio.run(
        run_eval(
            cases=cases,
            searcher=get_searcher(),
            settings=settings,
            chat_service=None if retrieval_only else get_chat_service(),
            llm=get_llm() if judge else None,
            top_k=top_k,
            judge=judge,
            concurrency=concurrency,
        )
    )

    _print_report(report, retrieval_only)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        console.print(f"\nFull report written to [bold]{json_out}[/bold]")

    if fail_under_recall is not None and report.recall < fail_under_recall:
        console.print(
            f"\n[red]Recall {report.recall:.3f} is below the "
            f"{fail_under_recall:.3f} threshold.[/red]"
        )
        raise typer.Exit(1)


@app.command("list-docs")
def list_docs() -> None:
    """List ingested documents, to help write a golden set."""
    configure_logging(get_settings().log_level)
    documents = asyncio.run(get_registry().list_all())
    if not documents:
        console.print("[yellow]No documents ingested yet.[/yellow]")
        return

    table = Table(title="Ingested documents")
    for column in ("id", "title", "stage", "pages", "chunks"):
        table.add_column(column)
    for document in documents:
        table.add_row(
            document.id,
            document.title,
            document.stage.value,
            str(document.pages or "-"),
            str(document.chunk_count or "-"),
        )
    console.print(table)


def _print_report(report: EvalReport, retrieval_only: bool) -> None:
    summary = Table(title=f"Summary (top_k={report.top_k})", show_header=False)
    summary.add_column("metric", style="bold")
    summary.add_column("value", justify="right")

    summary.add_row("cases", str(len(report.results)))
    summary.add_row("recall", f"{report.recall:.3f}")
    summary.add_row("hit rate", f"{report.hit_rate:.3f}")
    summary.add_row("MRR", f"{report.mrr:.3f}")
    summary.add_row("nDCG", f"{report.ndcg:.3f}")
    if not retrieval_only:
        summary.add_row("keyword coverage", f"{report.keyword_coverage:.3f}")
        summary.add_row("citation validity", f"{report.citation_validity:.3f}")
        if report.groundedness is not None:
            summary.add_row("groundedness", f"{report.groundedness:.3f}")
        summary.add_row("abstention rate", f"{report.abstention_rate:.3f}")
    summary.add_row("median latency", f"{report.median_latency_ms:.0f} ms")
    if report.errors:
        summary.add_row("errors", f"[red]{len(report.errors)}[/red]")
    console.print(summary)

    detail = Table(title="Per case")
    detail.add_column("id")
    detail.add_column("recall", justify="right")
    detail.add_column("MRR", justify="right")
    if not retrieval_only:
        detail.add_column("keywords", justify="right")
        detail.add_column("cites", justify="right")
    detail.add_column("ms", justify="right")

    for result in report.results:
        row = [
            result.case.id,
            f"{result.retrieval.recall:.2f}" if result.case.relevant_pages else "-",
            f"{result.retrieval.mrr:.2f}" if result.case.relevant_pages else "-",
        ]
        if not retrieval_only:
            metrics = result.answer_metrics
            row.append(f"{metrics.keyword_coverage:.2f}" if metrics else "-")
            row.append(str(metrics.citation_count) if metrics else "-")
        row.append(f"{result.latency_ms:.0f}")
        detail.add_row(*row)
    console.print(detail)

    for result in report.errors:
        console.print(f"[red]{result.case.id}: {result.error}[/red]")


if __name__ == "__main__":
    app()
