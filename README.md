# Agentic Data Prep

An AI agent that automatically cleans and preprocesses messy tabular data.
Upload a CSV, Excel, or JSON file — the agent profiles the data, plans a
fix using an LLM, executes each step, and explains what it did.

**[Live demo](#)** · Built with LangGraph + Groq + FastAPI

---

## What it does

1. **Profiles** the data — finds missing values, duplicates, outliers,
   inconsistent casing, and ambiguous date formats (pure Python, no AI).
2. **Plans** a fix — an LLM reads the findings and decides which tools to
   run, in what order.
3. **Executes** the plan — each tool (impute, deduplicate, scale, encode,
   etc.) actually runs and logs what it did.
4. **Reports** — the LLM writes a plain-English summary of the whole run.

The output includes the plan, a step-by-step execution log, a written
report, and the cleaned data ready to download.

---

## Why this exists

Most "AI agent" tutorials skip the hard part: making the agent's
decisions actually correct. This project is built around that problem —
every function is backed by real test data, and a small evaluation suite
(`eval_suite.py`) validates the agent against 3 independent test datasets
covering 12 distinct data-quality issues (missing values, duplicates,
outliers, casing problems), with a 100% resolution rate on the current
suite.

Along the way, testing surfaced and fixed over a dozen real bugs — a
pandas date-parsing issue that silently corrupted unambiguous dates,
false-positive duplicate detection, and a case where naive imputation
could fabricate fake duplicate records. Details in `PROJECT_GUIDE.md`.

---

## Tech stack

- **Agent**: LangGraph (plan-then-execute pattern) + Groq (Llama 3.3)
- **Data processing**: pandas, NumPy
- **Backend**: FastAPI
- **Frontend**: plain HTML/CSS/JS (no framework)
- **Deployment**: Docker, Render

---

## Project structure

```
profiling.py       # Detects data quality issues (no LLM — pure Python)
tools.py            # Cleaning functions: impute, dedupe, fix types, outliers, casing
preprocessing.py    # ML-prep functions: encode categories, scale numbers
tool_registry.py    # Maps tool names -> functions, describes them to the LLM
state.py            # Shared state object passed through the agent graph
agent.py            # The LangGraph agent: profile -> plan -> execute -> report
main.py             # FastAPI backend, wraps the agent as a web API
static/index.html   # Frontend demo page
eval_suite.py        # Validates the agent against test datasets
sample_data.csv, test_data_*.csv   # Test datasets with known, verified issues
PROJECT_GUIDE.md    # Full walkthrough of every file and concept
```

---

## Running it locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Groq API key (free at console.groq.com)
cp .env.example .env
# edit .env and paste your real key in

# 3a. Run the agent directly from the command line
python agent.py

# 3b. Or run it as a web app
uvicorn main:app --reload
# then open http://localhost:8000
```

To check the agent's accuracy against the test datasets:

```bash
python eval_suite.py
```

---

## Deployment

Containerized with the included `Dockerfile`. Deployed on Render — push to
GitHub, connect the repo, set the `GROQ_API_KEY` environment variable, and
Render builds and serves it automatically.

---

## Honest limitations

- **Reflection is simplified.** The agent currently checks "are there more
  planned steps left," not "did the last step actually work." A fuller
  version would re-profile after each step and let the LLM decide whether
  to retry.
- **Tested on small, hand-built datasets.** The 100% resolution rate is
  real, but it's on 3 datasets I designed to test specific issues — not
  yet validated against a large, real-world messy dataset.