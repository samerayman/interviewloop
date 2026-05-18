# InterviewLoop

> A multi-agent mock-interview panel. Paste your CV and a job description; four AI personas — recruiter, hiring manager, technical interviewer, and a bar-raiser who picks at your weakest answer — run a realistic 4-round interview and produce a scored, actionable report. Built with **LangChain + LangGraph**.

**Stack:** Python 3.12 · LangChain · LangGraph (StateGraph + checkpointing + HITL interrupts) · OpenAI `gpt-4o-mini` · LangSmith · Gradio · Typer

---

## Why this project exists

Every AI-engineering job posting in 2026 asks for some flavor of: *"You have shipped agentic workflows in production — LangGraph or similar."* Most portfolio projects on the topic are toy single-step chains. InterviewLoop is deliberately the opposite — it uses the parts of LangGraph that you only reach for when you actually need them:

- **Stateful multi-agent panel** with five distinct personas as graph nodes.
- **Human-in-the-loop interrupts** (`interrupt()`) so the graph pauses mid-flow to wait for the candidate's real answer, then resumes.
- **Persistent checkpointing** with `SqliteSaver` — kill the process mid-interview, restart with the same `--thread-id`, and pick up at the next question.
- **A self-looping node** (`bar_raiser`) that re-enters itself up to twice if it wants to drill deeper on a weak answer, decided by a model-emitted control token in the rubric's `notes` field.
- **Structured output everywhere** via Pydantic + `with_structured_output(...)` — no regex scraping LLM prose.
- **A LangSmith-instrumented eval harness** that drives the graph end-to-end on three fixture candidates (a strong ML engineer, a vague junior, a mid-level with a real weakness) and asserts that overall scores land in the expected band.

---

## Architecture

```
START → init → recruiter → hiring_manager → tech → bar_raiser ─┐
              ↑ HITL       ↑ HITL            ↑ HITL    ↑ HITL  │ (loops up to 2x
                                                       │       on weak answers)
                                                       └───────┘
                                                       │
                                                  synthesizer → END
```

[Full architecture, mermaid diagram, and design rationale →](docs/architecture.md)

The interesting parts in code:

| Where                                                                 | What                                                              |
|-----------------------------------------------------------------------|-------------------------------------------------------------------|
| [src/interviewloop/graph.py](src/interviewloop/graph.py)              | StateGraph wiring, conditional self-loop, checkpointer setup      |
| [src/interviewloop/state.py](src/interviewloop/state.py)              | TypedDict graph state + Pydantic models for structured output     |
| [src/interviewloop/prompts.py](src/interviewloop/prompts.py)          | Five persona system prompts + their scoring rubrics               |
| [evals/run_evals.py](evals/run_evals.py)                              | Driver that runs fixtures end-to-end and checks score ranges      |
| [evals/fixtures.json](evals/fixtures.json)                            | Three reference candidates with reference answers + expected score bands |

---

## Quickstart

### Option A — Docker (recommended)

```bash
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up --build
```

Open <http://localhost:7860>.

### Option B — local Python

```bash
python -m venv .venv
. .venv/Scripts/activate           # PowerShell:  .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env               # add your OPENAI_API_KEY

# Terminal interview
python -m interviewloop.cli run --cv examples/sample_cv.txt --jd examples/sample_jd.txt

# OR the Gradio UI on http://localhost:7860
python -m interviewloop.app
```

### Run the eval suite

```bash
python evals/run_evals.py
```

This drives the graph end-to-end on three fixture candidates with their reference answers and asserts that the overall scores land in the expected band. Exit code 0 = all fixtures within range. Full per-fixture report is written to `evals/last_run.json`.

If you set `LANGSMITH_TRACING=true` in `.env`, every node call shows up in your LangSmith project with a shareable trace URL — drop a screenshot in your README/LinkedIn.

---

## A walk through one interview

1. **`init`** — parses the CV and JD into `CandidateProfile` and `RoleProfile` using `with_structured_output`. No interrupt.
2. **`recruiter`** — generates a screen-style question grounded in BOTH the CV and the JD (no clichés), interrupts for your answer, scores the answer on clarity / motivation / fit_signal / communication.
3. **`hiring_manager`** — probes a specific project from your CV. Scoring rewards STAR-style structure and penalizes vague "we built a thing".
4. **`tech`** — deep-dive on one technical area present in both the CV and the JD. Scoring rewards tradeoff thinking, not just correctness.
5. **`bar_raiser`** — reads the full transcript, identifies your weakest answer, and presses. Emits `CONTINUE_PROBING` or `STOP` in the rubric's `notes`. The graph's conditional edge re-enters this node (max twice) if it wants to drill deeper.
6. **`synthesizer`** — emits a `FinalReport`: weighted overall score, blunt fit summary, top strengths/weaknesses across rounds, specific talking points to rehearse, and the follow-up questions the real interviewer is most likely to ask given your weak spots.

---

## What this project is meant to demonstrate

To anyone reading this as part of a hiring evaluation:

- **LangGraph fluency, not just LangChain wrappers** — every part of LangGraph that's actually useful in production (state, reducers, interrupts, checkpointers, conditional edges, structured outputs) is used here for a real reason, not for show.
- **Eval discipline** — there's a fixture-driven eval harness with expected score bands, not just a happy-path demo. Three candidate profiles cover the strong/weak/mixed cases.
- **Production-shape engineering** — Pydantic-typed state, structured LLM outputs, separated prompt files, a CLI and a UI driven by the same graph, Docker, MIT license, `.env` discipline, pytest smoke tests.
- **Cost-aware** — full interview runs in pennies on `gpt-4o-mini`; cost breakdown is in [docs/architecture.md](docs/architecture.md).

---

## Repo layout

```
interviewloop/
├── README.md
├── LICENSE                              # MIT
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── src/interviewloop/
│   ├── __init__.py
│   ├── state.py                         # TypedDict + Pydantic models
│   ├── prompts.py                       # 5 persona prompts + scoring rubrics
│   ├── parsing.py                       # CV/JD → structured profiles
│   ├── graph.py                         # the StateGraph
│   ├── cli.py                           # `python -m interviewloop.cli run ...`
│   └── app.py                           # `python -m interviewloop.app` (Gradio)
├── evals/
│   ├── fixtures.json                    # 3 candidates × CV + JD + ref answers
│   └── run_evals.py                     # end-to-end driver + score-band asserts
├── tests/
│   └── test_graph.py                    # offline smoke tests (no API key needed)
├── examples/
│   ├── sample_cv.txt
│   └── sample_jd.txt
└── docs/
    └── architecture.md                  # graph diagram, design choices, cost
```

---

## Author

Built by **Samer Farid** as a portfolio piece for AI/ML engineer roles. Feedback welcome.
