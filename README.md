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

## Why I built this

I wanted a project that proved I understand agentic AI, not just that I
can call an LLM API. So instead of stopping once the agent "worked," I
built a small evaluation suite (`eval_suite.py`) with 3 test datasets I
designed myself, each with specific known issues, and checked whether the
agent actually fixed them.

It didn't, at first. Testing surfaced over a dozen real bugs — a pandas
date-parsing bug that silently flipped `2023-03-10` into October 3rd, a
duplicate-detection method that missed rows unless they were byte-for-byte
identical, a classification heuristic that mislabeled personal names as
categories, and an imputation strategy that could fabricate a fake
duplicate by copying one row's name into another. I traced each one to
its root cause, fixed it, and re-ran the suite to confirm.

After that process, the agent resolves all 12 known issues across my 3
test datasets. 

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

## What I'd build next

- **Real reflection.** Right now the agent checks "are there more planned
  steps left," not "did the last step actually work." I'd add a
  re-profile-and-retry loop so the agent verifies its own results instead
  of trusting the plan blindly.
- **Testing at scale.** My 100% resolution rate is real, but it's against
  3 datasets I designed myself to test specific issues. The next step is
  running it against a large, messy, real-world dataset (a public Kaggle
  one) to see what breaks that my test data didn't cover.