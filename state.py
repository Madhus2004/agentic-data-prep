"""
state.py

The state object LangGraph passes between every node. Think of it as the
agent's working memory for one cleaning run.
"""

from typing import TypedDict, Any
import pandas as pd


class AgentState(TypedDict):
    df: pd.DataFrame                 # the dataframe, mutated as tools run
    profile: dict                    # output of profiling.profile_data()
    plan: list[dict]                 # ordered list of steps the LLM decided on
    current_step: int                # index into plan
    log: list[dict]                  # log_entry from each tool call so far
    report: str                      # final LLM-written summary