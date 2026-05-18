"""Offline smoke tests — no OpenAI calls required."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from interviewloop.state import (  # noqa: E402
    CandidateProfile,
    FinalReport,
    InterviewState,
    QA,
    RoleProfile,
    RoundScore,
)


def test_state_models_roundtrip():
    cp = CandidateProfile(name="Sam", years_experience=5.0, top_skills=["python", "n8n"])
    rp = RoleProfile(title="AI Engineer", must_have_skills=["python", "langchain"])
    rs = RoundScore(
        round_name="recruiter",
        scores={"clarity": 4, "motivation": 5},
        strengths=["clear"],
        weaknesses=["vague on metrics"],
        follow_up_questions=["What specifically?"],
        notes="STOP",
    )
    report = FinalReport(
        overall_score=4.2,
        fit_summary="Strong candidate",
        top_strengths=["communicates well"],
        top_weaknesses=["thin on production ML"],
        answers_to_rehearse=["Quantify the recommender impact"],
        likely_followups=["Walk me through the eval pipeline"],
    )

    for m in (cp, rp, rs, report):
        d = m.model_dump()
        assert isinstance(d, dict)


def test_graph_compiles_without_checkpointer():
    from interviewloop.graph import build_graph

    g = build_graph(checkpoint_path=None)
    assert g is not None
    # The compiled graph exposes a `nodes` view
    nodes = set(g.get_graph().nodes.keys())
    for expected in {"init", "recruiter", "hiring_manager", "tech", "bar_raiser", "synthesizer"}:
        assert expected in nodes, f"missing node: {expected}"


def test_qa_typed_dict():
    qa: QA = {"round": "recruiter", "question": "Why us?", "answer": "Because..."}
    assert qa["round"] == "recruiter"


def test_after_bar_raiser_routing():
    from interviewloop.graph import _after_bar_raiser

    assert _after_bar_raiser({"bar_raiser_done": True}) == "synthesizer"
    assert _after_bar_raiser({"bar_raiser_done": False}) == "bar_raiser"
    assert _after_bar_raiser({}) == "bar_raiser"
