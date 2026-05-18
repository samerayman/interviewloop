"""LangSmith-instrumented eval harness for InterviewLoop.

For each fixture, we drive the graph with the reference answers, capture the
FinalReport, and assert that overall_score falls inside the expected range.

Run:
    LANGSMITH_API_KEY=ls-...   # optional; enables tracing
    LANGSMITH_PROJECT=interviewloop-evals
    python evals/run_evals.py
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

# Make `src/` importable without an install
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from interviewloop.graph import build_graph  # noqa: E402

FIXTURES = ROOT / "evals" / "fixtures.json"

# Round order must match the graph. Bar-raiser may loop; we always supply the
# same fallback answer if it asks for more (rare with our fixtures).
ROUND_ORDER = ["recruiter", "hiring_manager", "tech", "bar_raiser"]


def run_fixture(fixture: dict) -> dict:
    graph = build_graph(checkpoint_path=None)
    config = {"configurable": {"thread_id": f"eval-{fixture['id']}-{uuid.uuid4().hex[:8]}"}}

    result = graph.invoke(
        {"cv_text": fixture["cv"], "jd_text": fixture["jd"]},
        config=config,
    )

    answers_by_round = fixture["reference_answers"]
    fallback = "I'd rather be honest — I don't have a strong answer for that."

    while True:
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if not interrupts:
            break
        iv = interrupts[0]
        data = iv.value if hasattr(iv, "value") else iv
        round_name = data["round"]
        answer = answers_by_round.get(round_name, fallback)
        result = graph.invoke(Command(resume=answer), config=config)

    report = result.get("final_report", {}) if isinstance(result, dict) else {}
    return {
        "id": fixture["id"],
        "label": fixture["label"],
        "report": report,
        "round_scores": result.get("round_scores", {}) if isinstance(result, dict) else {},
    }


def main() -> int:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — aborting.", file=sys.stderr)
        return 2

    data = json.loads(FIXTURES.read_text(encoding="utf-8"))
    failures = 0
    results = []

    for fx in data["fixtures"]:
        print(f"→ Running fixture: {fx['id']}")
        out = run_fixture(fx)
        results.append(out)

        overall = float(out["report"].get("overall_score", 0.0))
        lo = float(fx["expected_scores"]["overall_min"])
        hi = float(fx["expected_scores"]["overall_max"])
        ok = lo <= overall <= hi
        marker = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"   {marker} — overall {overall:.2f} (expected {lo:.2f} ≤ x ≤ {hi:.2f})")

    out_path = ROOT / "evals" / "last_run.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull run written to {out_path}")
    print(f"Summary: {len(results) - failures}/{len(results)} fixtures within expected range.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
