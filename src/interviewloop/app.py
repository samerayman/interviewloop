"""Gradio UI for InterviewLoop.

Run:
    python -m interviewloop.app

The app exposes one tab: paste your CV + a JD, click "Start interview",
then answer each round in the chat box. The final report renders as Markdown.
"""
from __future__ import annotations

import json
import os
import uuid

import gradio as gr
from dotenv import load_dotenv
from langgraph.types import Command

from .graph import build_graph

load_dotenv()

# Per-session graph + thread map. Gradio gives us a session-scoped state dict.
def _new_session():
    return {
        "graph": build_graph(checkpoint_path=None),  # in-memory for UI sessions
        "thread_id": str(uuid.uuid4()),
        "started": False,
        "done": False,
    }


def _format_report_md(report: dict) -> str:
    if not report:
        return "_(no report)_"
    lines = [
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
    return "\n".join(lines)


def start_interview(cv_text, jd_text, session):
    if not os.getenv("OPENAI_API_KEY"):
        return [("system", "Set OPENAI_API_KEY in your environment first.")], "", session
    if not cv_text.strip() or not jd_text.strip():
        return [("system", "Paste both a CV and a JD to start.")], "", session

    sess = session or _new_session()
    graph = sess["graph"]
    config = {"configurable": {"thread_id": sess["thread_id"]}}
    result = graph.invoke({"cv_text": cv_text, "jd_text": jd_text}, config=config)

    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return [("system", "Graph ended unexpectedly.")], "", sess

    data = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
    sess["started"] = True
    return (
        [(None, f"**Round: {data['round']}**\n\n{data['question']}")],
        "",
        sess,
    )


def submit_answer(answer, history, session):
    if not session or not session.get("started"):
        return history + [(answer, "Click 'Start interview' first.")], "", session
    if session.get("done"):
        return history, "", session

    graph = session["graph"]
    config = {"configurable": {"thread_id": session["thread_id"]}}
    history = history + [(answer, None)]

    result = graph.invoke(Command(resume=answer), config=config)
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None

    if interrupts:
        data = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        history[-1] = (answer, f"**Round: {data['round']}**\n\n{data['question']}")
        return history, "", session

    report = result.get("final_report") if isinstance(result, dict) else None
    if report:
        session["done"] = True
        history[-1] = (answer, "_Interview complete. See report below._")
        return history, _format_report_md(report), session

    history[-1] = (answer, "_(graph ended without a report)_")
    return history, "", session


def build_ui():
    with gr.Blocks(title="InterviewLoop", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# InterviewLoop\n"
            "A multi-agent mock-interview panel. Paste your CV and a job description, "
            "then answer 4 rounds. Built with LangChain + LangGraph."
        )

        session = gr.State()

        with gr.Row():
            with gr.Column(scale=1):
                cv = gr.Textbox(label="Your CV", lines=14, placeholder="Paste your CV as plain text…")
                jd = gr.Textbox(label="Job description", lines=14, placeholder="Paste the JD…")
                start_btn = gr.Button("Start interview", variant="primary")

            with gr.Column(scale=2):
                chat = gr.Chatbot(label="Interview", height=520)
                answer = gr.Textbox(label="Your answer", lines=4)
                submit_btn = gr.Button("Submit answer", variant="primary")
                report_md = gr.Markdown(label="Final report")

        start_btn.click(start_interview, inputs=[cv, jd, session], outputs=[chat, report_md, session])
        submit_btn.click(submit_answer, inputs=[answer, chat, session], outputs=[chat, report_md, session])
        answer.submit(submit_answer, inputs=[answer, chat, session], outputs=[chat, report_md, session])

    return demo


def main():
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
