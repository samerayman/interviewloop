"""LangGraph StateGraph for the InterviewLoop multi-agent panel.

Flow:
    init -> recruiter -> hiring_manager -> tech -> bar_raiser
                                                       |
                                              (loops up to 2x)
                                                       |
                                                 synthesizer -> END

Each interviewer node is one Python function that:
    1. Generates a question with an LLM (persona system prompt).
    2. Calls `interrupt(...)` to surface the question to the human and wait for the answer.
    3. Scores the answer with a second LLM call using `with_structured_output(RoundScore)`.
    4. Returns a state update with the QA pair and the score.

The bar_raiser node sets `bar_raiser_done` based on a CONTINUE_PROBING / STOP token
emitted in the rubric's `notes`, with a hard cap of 2 follow-ups. The conditional
edge after bar_raiser routes either back to itself or onward to the synthesizer.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .parsing import parse_cv, parse_jd
from .prompts import (
    BAR_RAISER_QUESTION_SYS,
    BAR_RAISER_SCORE_SYS,
    HM_QUESTION_SYS,
    HM_SCORE_SYS,
    RECRUITER_QUESTION_SYS,
    RECRUITER_SCORE_SYS,
    SYNTHESIZER_SYS,
    TECH_QUESTION_SYS,
    TECH_SCORE_SYS,
)
from .state import FinalReport, InterviewState, RoundScore

BAR_RAISER_MAX_DEPTH = 2


def _llm(temperature: float = 0.4) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("INTERVIEWLOOP_MODEL", "gpt-4o-mini"),
        temperature=temperature,
    )


# --- Helpers ---

def _format_context(state: InterviewState, include_transcript: bool = True) -> str:
    parts = [
        f"CANDIDATE PROFILE:\n{state.get('candidate_profile', {})}",
        f"\nROLE PROFILE:\n{state.get('role_profile', {})}",
    ]
    if include_transcript and state.get("transcript"):
        parts.append("\nTRANSCRIPT SO FAR:")
        for qa in state["transcript"]:
            parts.append(f"[{qa['round']}] Q: {qa['question']}\n  A: {qa['answer']}")
    return "\n".join(parts)


def _run_round(
    state: InterviewState,
    round_name: str,
    question_sys: str,
    score_sys: str,
) -> dict[str, Any]:
    """Shared scaffolding for one interview round: ask -> interrupt -> score."""
    creative = _llm(temperature=0.6)
    analytic = _llm(temperature=0.1)

    # 1. Generate the question
    q_resp = creative.invoke(
        [
            SystemMessage(content=question_sys),
            HumanMessage(content=_format_context(state)),
        ]
    )
    question = q_resp.content.strip()

    # 2. Pause for the human's answer
    answer = interrupt({"round": round_name, "question": question})

    # 3. Score with structured output
    scorer = analytic.with_structured_output(RoundScore)
    score: RoundScore = scorer.invoke(
        [
            SystemMessage(content=score_sys),
            HumanMessage(
                content=(
                    f"ROUND: {round_name}\n"
                    f"QUESTION: {question}\n"
                    f"ANSWER: {answer}\n\n"
                    f"For context, the candidate and role profiles are:\n"
                    f"{_format_context(state, include_transcript=False)}"
                )
            ),
        ]
    )
    score.round_name = round_name

    return {
        "transcript": [{"round": round_name, "question": question, "answer": str(answer)}],
        "round_scores": {**state.get("round_scores", {}), round_name: score.model_dump()},
    }


# --- Nodes ---

def init_node(state: InterviewState) -> dict[str, Any]:
    parser_llm = _llm(temperature=0.0)
    return {
        "candidate_profile": parse_cv(state["cv_text"], parser_llm),
        "role_profile": parse_jd(state["jd_text"], parser_llm),
        "transcript": [],
        "round_scores": {},
        "bar_raiser_depth": 0,
        "bar_raiser_done": False,
    }


def recruiter_node(state: InterviewState) -> dict[str, Any]:
    return _run_round(state, "recruiter", RECRUITER_QUESTION_SYS, RECRUITER_SCORE_SYS)


def hiring_manager_node(state: InterviewState) -> dict[str, Any]:
    return _run_round(state, "hiring_manager", HM_QUESTION_SYS, HM_SCORE_SYS)


def tech_node(state: InterviewState) -> dict[str, Any]:
    return _run_round(state, "tech", TECH_QUESTION_SYS, TECH_SCORE_SYS)


def bar_raiser_node(state: InterviewState) -> dict[str, Any]:
    update = _run_round(state, "bar_raiser", BAR_RAISER_QUESTION_SYS, BAR_RAISER_SCORE_SYS)
    depth = state.get("bar_raiser_depth", 0) + 1

    # Read the CONTINUE_PROBING / STOP token from the round score's notes.
    latest_score = update["round_scores"].get("bar_raiser", {})
    notes = (latest_score.get("notes") or "").upper()
    wants_more = "CONTINUE_PROBING" in notes
    done = (not wants_more) or (depth >= BAR_RAISER_MAX_DEPTH)

    update["bar_raiser_depth"] = depth
    update["bar_raiser_done"] = done
    return update


def synthesizer_node(state: InterviewState) -> dict[str, Any]:
    analytic = _llm(temperature=0.1)
    structured = analytic.with_structured_output(FinalReport)
    report: FinalReport = structured.invoke(
        [
            SystemMessage(content=SYNTHESIZER_SYS),
            HumanMessage(
                content=(
                    f"{_format_context(state)}\n\n"
                    f"ROUND SCORES:\n{state.get('round_scores', {})}"
                )
            ),
        ]
    )
    return {"final_report": report.model_dump()}


# --- Conditional edge ---

def _after_bar_raiser(state: InterviewState) -> str:
    return "synthesizer" if state.get("bar_raiser_done") else "bar_raiser"


# --- Public builder ---

def build_graph(checkpoint_path: str | None = "interviewloop.sqlite"):
    """Compile and return the StateGraph with a SqliteSaver checkpointer."""
    g = StateGraph(InterviewState)

    g.add_node("init", init_node)
    g.add_node("recruiter", recruiter_node)
    g.add_node("hiring_manager", hiring_manager_node)
    g.add_node("tech", tech_node)
    g.add_node("bar_raiser", bar_raiser_node)
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "init")
    g.add_edge("init", "recruiter")
    g.add_edge("recruiter", "hiring_manager")
    g.add_edge("hiring_manager", "tech")
    g.add_edge("tech", "bar_raiser")
    g.add_conditional_edges(
        "bar_raiser",
        _after_bar_raiser,
        {"bar_raiser": "bar_raiser", "synthesizer": "synthesizer"},
    )
    g.add_edge("synthesizer", END)

    if checkpoint_path:
        checkpointer = SqliteSaver.from_conn_string(checkpoint_path)
        # SqliteSaver.from_conn_string returns a contextmanager in some versions;
        # handle both cases for compatibility.
        if hasattr(checkpointer, "__enter__"):
            checkpointer = checkpointer.__enter__()
        return g.compile(checkpointer=checkpointer)

    return g.compile()
