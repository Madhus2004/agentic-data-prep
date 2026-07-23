"""
agent.py

The actual LangGraph agent. Graph shape (matches the architecture diagram
we designed earlier):

  ingest -> profile -> plan (LLM) -> execute_step -> reflect -+
                                          ^                    |
                                          +---- (more steps) --+
                                                                |
                                                  (plan done) -> report

Run with:
    GROQ_API_KEY=your_key python agent.py
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from state import AgentState
from tool_registry import TOOL_REGISTRY, TOOL_DESCRIPTIONS
import profiling

load_dotenv()  # reads .env in the current directory into os.environ, if present

LLM_MODEL = "llama-3.3-70b-versatile"  # check console.groq.com/docs/models for current options


def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at console.groq.com, then either:\n"
            "  1. Create a .env file in this folder containing:\n"
            "       GROQ_API_KEY=your_key\n"
            "  2. Or export it directly:\n"
            "       export GROQ_API_KEY=your_key   (mac/linux)\n"
            "       set GROQ_API_KEY=your_key      (windows cmd)\n"
        )
    return ChatGroq(model=LLM_MODEL, temperature=0, api_key=api_key)


# ---------- Nodes ----------

def profile_node(state: AgentState) -> dict:
    profile = profiling.profile_data(state["df"])
    return {"profile": profile}


def plan_node(state: AgentState) -> dict:
    llm = get_llm()
    issues = profiling.summarize_issues(state["profile"])

    prompt = f"""You are a data cleaning and preprocessing planner.

Below is a list of specific issues already found in this dataset. Each
issue tells you what's wrong and, where relevant, exactly which tool and
parameters to use. Turn this list into an ordered plan. Do not invent
issues that aren't listed. Do not skip an issue that IS listed, including
any "Do NOT encode or scale this column" instructions — those are hard
constraints, not suggestions.

Issues found:
{chr(10).join(f"- {i}" for i in issues)}

Available tools:
{TOOL_DESCRIPTIONS}

For exact numeric bounds/means/dtypes if needed, here is the full profile:
{json.dumps(state["profile"], indent=2, default=str)}

Respond with ONLY a JSON array, no other text, no markdown fences. Each
element must look like:
{{"tool": "<tool_name>", "params": {{"column": "...", ...other params...}}}}

Order matters: cleaning steps (impute, dedupe, fix_dtype, outliers,
standardize_text) must come before preprocessing steps (encode_categorical,
scale_numeric) on the same column. Never include encode_categorical or
scale_numeric for a column marked "Do NOT encode or scale".
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # strip markdown fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        raw = raw.removeprefix("json").strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Planner returned invalid JSON: {e}\nRaw output:\n{raw}")

    return {"plan": plan, "current_step": 0, "log": []}


def execute_step_node(state: AgentState) -> dict:
    step = state["plan"][state["current_step"]]
    tool_name = step["tool"]
    params = step.get("params", {})

    if tool_name not in TOOL_REGISTRY:
        log_entry = {"tool": tool_name, "column": params.get("column"),
                     "status": "failed", "detail": f"unknown tool '{tool_name}'"}
        return {"log": state["log"] + [log_entry], "current_step": state["current_step"] + 1}

    fn = TOOL_REGISTRY[tool_name]
    try:
        new_df, log_entry = fn(state["df"], **params)
    except Exception as e:
        log_entry = {"tool": tool_name, "column": params.get("column"),
                     "status": "failed", "detail": f"execution error: {e}"}
        return {"log": state["log"] + [log_entry], "current_step": state["current_step"] + 1}

    return {"df": new_df, "log": state["log"] + [log_entry], "current_step": state["current_step"] + 1}


def reflect_router(state: AgentState) -> str:
    """Conditional edge: loop back to execute_step if steps remain, else move to report."""
    if state["current_step"] < len(state["plan"]):
        return "execute_step"
    return "report"


def report_node(state: AgentState) -> dict:
    llm = get_llm()

    prompt = f"""Summarize this data cleaning run for a non-technical reader.
Be concise (5-8 sentences). Mention what was wrong with the data and what
was done about it. Here is the action log:

{json.dumps(state["log"], indent=2, default=str)}
"""
    response = llm.invoke(prompt)
    return {"report": response.content.strip()}


# ---------- Graph assembly ----------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("profile", profile_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "profile")
    graph.add_edge("profile", "plan")
    graph.add_edge("plan", "execute_step")
    graph.add_conditional_edges("execute_step", reflect_router, {
        "execute_step": "execute_step",
        "report": "report",
    })
    graph.add_edge("report", END)

    return graph.compile()


if __name__ == "__main__":
    df = pd.read_csv("sample_data.csv")
    app = build_graph()

    initial_state: AgentState = {
        "df": df, "profile": {}, "plan": [], "current_step": 0, "log": [], "report": ""
    }

    final_state = app.invoke(initial_state)

    print("=== PLAN ===")
    for step in final_state["plan"]:
        print(step)

    print("\n=== EXECUTION LOG ===")
    for entry in final_state["log"]:
        print(entry)

    print("\n=== REPORT ===")
    print(final_state["report"])

    print("\n=== FINAL DATA (head) ===")
    print(final_state["df"].head())