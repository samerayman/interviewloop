"""Graph state and structured-output models for InterviewLoop."""
from __future__ import annotations

from operator import add
from typing import Annotated, Optional, TypedDict

from pydantic import BaseModel, Field


# --- Pydantic models used for structured LLM output ---

class CandidateProfile(BaseModel):
    """Distilled facts extracted from the candidate's CV."""
    name: Optional[str] = None
    years_experience: Optional[float] = None
    current_role: Optional[str] = None
    top_skills: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list, description="Industries / problem domains.")
    notable_projects: list[str] = Field(default_factory=list)


class RoleProfile(BaseModel):
    """Distilled facts extracted from the job description."""
    title: Optional[str] = None
    company: Optional[str] = None
    seniority: Optional[str] = None
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)


class RoundScore(BaseModel):
    """Rubric scoring for a single interview round."""
    round_name: str
    scores: dict[str, int] = Field(
        description="Criterion → 0–5 score. Use a consistent rubric for the round."
    )
    strengths: list[str] = Field(default_factory=list, description="Up to 3 short strengths.")
    weaknesses: list[str] = Field(default_factory=list, description="Up to 3 short weaknesses.")
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Questions a real interviewer would press on next.",
    )
    notes: str = ""


class FinalReport(BaseModel):
    """The end-of-interview report shown to the candidate."""
    overall_score: float = Field(description="0.0–5.0 weighted average across rounds.")
    fit_summary: str = Field(description="2–3 sentence verdict.")
    top_strengths: list[str] = Field(default_factory=list)
    top_weaknesses: list[str] = Field(default_factory=list)
    answers_to_rehearse: list[str] = Field(
        default_factory=list,
        description="Specific talking points the candidate should drill before the real interview.",
    )
    likely_followups: list[str] = Field(
        default_factory=list,
        description="Questions the real interviewer is most likely to ask given the weak spots.",
    )


# --- Graph state ---

class QA(TypedDict):
    round: str
    question: str
    answer: str


class InterviewState(TypedDict, total=False):
    # Inputs
    cv_text: str
    jd_text: str

    # Parsed
    candidate_profile: dict
    role_profile: dict

    # Per-round accumulating transcript (reducer = list-concat)
    transcript: Annotated[list[QA], add]

    # Round outputs (round_name -> RoundScore dict)
    round_scores: dict[str, dict]

    # Bar-raiser bookkeeping
    bar_raiser_depth: int
    bar_raiser_done: bool

    # Final
    final_report: dict
