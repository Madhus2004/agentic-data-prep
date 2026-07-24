"""
main.py

FastAPI wrapper around the LangGraph agent. One real endpoint: upload a
file, get back the agent's plan, execution log, report, and the cleaned
data — all in one response, so a frontend can render the whole story.
"""

import io
import json
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agent import build_graph
from state import AgentState

app = FastAPI(
    title="Agentic Data Prep",
    description="An LLM agent that profiles, cleans, and preprocesses tabular data.",
)

# Permissive CORS since this is a public demo, not an internal tool with
# sensitive data — anyone should be able to hit the API from the deployed
# frontend or their own client.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_graph = None


def get_graph():
    """Build the LangGraph agent once and reuse it across requests."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _load_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    buf = io.BytesIO(content)

    if ext == "csv":
        return pd.read_csv(buf)
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(buf)
    elif ext == "json":
        return pd.read_json(buf)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Upload a .csv, .xlsx, or .json file.",
        )


@app.get("/health")
def health():
    return {"status": "ok", "service": "agentic-data-prep"}


@app.post("/clean")
async def clean_data(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    df = _load_dataframe(file.filename, content)
    if df.empty:
        raise HTTPException(status_code=400, detail="File parsed but contains no rows.")

    initial_state: AgentState = {
        "df": df, "profile": {}, "plan": [], "current_step": 0, "log": [], "report": ""
    }

    try:
        final_state = get_graph().invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}")

    return JSONResponse({
        "original_rows": len(df),
        "original_columns": list(df.columns),
        "plan": final_state["plan"],
        "log": final_state["log"],
        "report": final_state["report"],
        "cleaned_rows": len(final_state["df"]),
        "cleaned_columns": list(final_state["df"].columns),
        "cleaned_data_csv": final_state["df"].to_csv(index=False, date_format="%Y-%m-%d"),
    })


# Serve the minimal frontend at "/". Must be mounted last so it doesn't
# shadow the API routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")