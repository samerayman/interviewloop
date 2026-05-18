"""Extract structured profiles from raw CV and JD text."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import PARSE_CV_SYS, PARSE_JD_SYS
from .state import CandidateProfile, RoleProfile


def parse_cv(cv_text: str, llm) -> dict:
    """Return a CandidateProfile as a dict (so it round-trips through graph state)."""
    structured = llm.with_structured_output(CandidateProfile)
    profile: CandidateProfile = structured.invoke(
        [
            SystemMessage(content=PARSE_CV_SYS),
            HumanMessage(content=cv_text[:8000]),
        ]
    )
    return profile.model_dump()


def parse_jd(jd_text: str, llm) -> dict:
    """Return a RoleProfile as a dict."""
    structured = llm.with_structured_output(RoleProfile)
    profile: RoleProfile = structured.invoke(
        [
            SystemMessage(content=PARSE_JD_SYS),
            HumanMessage(content=jd_text[:6000]),
        ]
    )
    return profile.model_dump()
