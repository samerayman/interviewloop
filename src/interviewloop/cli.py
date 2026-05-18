"""Terminal driver for InterviewLoop.

Usage:
    python -m interviewloop.cli --cv examples/sample_cv.txt --jd examples/sample_jd.txt

The CLI drives the LangGraph step by step:
    1. Start a thread.
    2. Run until the next interrupt → print the interviewer's question.
    3. Read the user's answer from stdin.
    4. Resume with Command(resume=answer). Repeat until the graph ends.
    5. Pretty-print the final report.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import typer
from dotenv import load_dotenv
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from .graph import build_graph

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _print_question(round_name: str, question: str) -> None:
    console.print(Rule(f"[bold cyan]Round: {round_name}[/]"))
    console.print(Panel(question, title="Interviewer", border_style="cyan"))


def _print_report(report: dict) -> None:
    console.print(Rule("[bold green]Final Report[/]"))
    md = [
        f"## Overall: **{report.get('overall_score', 0):.2f} / 5.00**",
        "",
        f"**Verdict:** {report.get('fit_summary', '')}",
        "",
        "### Top strengths",
        *[f"- {s}" for s in report.get("top_strengths", [])],
        "",
        "### Top weaknesses",
        *[f"- {w}" for w in report.get("top_weaknesses", [])],
        "",
        "### Rehearse before the real interview",
        *[f"- {a}" for a in report.get("answers_to_rehearse", [])],
        "",
        "### Likely follow-ups they'll ask",
        *[f"- {q}" for q in report.get("likely_followups", [])],
    ]
    console.print(Markdown("\n".join(md)))


@app.command()
def run(
    cv: Path = typer.Option(..., help="Path to a plain-text CV."),
    jd: Path = typer.Option(..., help="Path to a plain-text job description."),
    thread_id: str = typer.Option(None, help="Resume an existing interview by thread id."),
    save_report: Path = typer.Option(None, help="Write the final report JSON to this path."),
) -> None:
    """Run an interactive mock-interview session in the terminal."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY not set. Add it to .env or your shell.[/]")
        raise typer.Exit(code=1)

    graph = build_graph()
    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    console.print(f"[dim]Thread id: {tid}[/]")
    initial: dict = {"cv_text": _read(cv), "jd_text": _read(jd)} if not thread_id else None

    payload: object = initial
    while True:
        # Stream/invoke until the next interrupt or END
        if isinstance(payload, dict) or payload is None:
            result = graph.invoke(payload, config=config)
        else:
            result = graph.invoke(payload, config=config)

        # When the graph pauses at an interrupt, LangGraph surfaces a `__interrupt__` key.
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if interrupts:
            iv = interrupts[0]
            data = iv.value if hasattr(iv, "value") else iv
            _print_question(data["round"], data["question"])
            console.print()
            try:
                answer = console.input("[bold]Your answer ▸ [/]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Interrupted. State checkpointed; resume with --thread-id.[/]")
                raise typer.Exit(code=130)
            payload = Command(resume=answer)
            continue

        # Graph finished
        report = result.get("final_report") if isinstance(result, dict) else None
        if report:
            _print_report(report)
            if save_report:
                save_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
                console.print(f"[dim]Report saved to {save_report}[/]")
        else:
            console.print("[yellow]Graph ended without a final_report.[/]")
        break


if __name__ == "__main__":
    app()
