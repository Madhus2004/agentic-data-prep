"""
tool_registry.py

Maps tool names (strings the LLM will output) to the actual Python
functions in tools.py / preprocessing.py, plus a text description of each
tool's purpose and parameters. This description is what gets embedded in
the planner's prompt — it's the LLM's only knowledge of what these tools
do, so keep it accurate and specific.

Why not LangChain's @tool decorator + native tool-calling here? Native
tool-calling expects the LLM to call one tool per turn and get a real
tool result back before continuing. Our tools operate on a DataFrame,
which isn't something you can round-trip through a chat message. Instead
we use a "plan-then-execute" pattern: the LLM sees the profile once and
proposes an entire ordered plan as JSON, then LangGraph executes each
step deterministically. This is a standard, well-documented LangGraph
pattern (plan-and-execute agents) and it's easier to debug because the
plan is fully visible before anything runs.
"""

import tools
import preprocessing

TOOL_REGISTRY = {
    "impute_missing": tools.impute_missing,
    "remove_duplicates": tools.remove_duplicates,
    "fix_dtype": tools.fix_dtype,
    "handle_outliers": tools.handle_outliers,
    "standardize_text": tools.standardize_text,
    "encode_categorical": preprocessing.encode_categorical,
    "scale_numeric": preprocessing.scale_numeric,
}

# Fed directly into the planner's prompt. Keep param names exactly matching
# each function's signature — the planner's JSON output will be passed as
# **kwargs to the real function.
TOOL_DESCRIPTIONS = """
- impute_missing(column, strategy): fill missing values.
  strategy: "median" | "mean" | "mode" | "drop_rows"
  Use "median"/"mean" for numeric columns with outliers or skew, "mode" for
  categorical columns, "drop_rows" only if missing_pct is very small.

- remove_duplicates(subset): remove exact duplicate rows.
  subset: list of column names to compare on (e.g. ["email"]), or null for all columns.
  Check profile.duplicates.count_excluding_id_columns — if it's greater than
  profile.duplicates.count, rows differ only by an id-like column and are
  true duplicates; use subset = profile.duplicates.id_columns_excluded's
  complement (i.e. all non-id columns) to catch them.
  ALSO check profile.duplicates.natural_key_duplicates — if a column like
  "email" appears there, rows share that value even though other columns
  differ (e.g. one row has missing data the other has filled in). These
  are the same real-world entity recorded twice; use subset=[that_column]
  to deduplicate on it, keeping the more complete row.

- fix_dtype(column, target_type, dayfirst): convert a column's dtype.
  target_type: "int" | "float" | "datetime" | "str"
  dayfirst: true|false, only relevant for target_type="datetime". Use the
  date_ambiguity info in the profile to decide — if dayfirst_evidence is
  true or the data looks non-US, prefer true.

- handle_outliers(column, method): handle numeric outliers found via IQR.
  method: "cap" (clip to bounds, preserves row count) | "remove" (drop rows)
  Prefer "cap" unless the outlier count is tiny and clearly an error.

- standardize_text(column, case, strip_whitespace): fix casing/whitespace
  in a text column so equivalent values merge (e.g. "USA"/"usa").
  case: "lower" | "upper" | "title". Use when casing_inconsistent is true.

- encode_categorical(column, method): convert a category column to numeric.
  method: "onehot" (few categories, no order) | "label" (many categories or ordinal)
  Only use on columns that are is_likely_categorical, AFTER cleaning them.

- scale_numeric(column, method): rescale a numeric column.
  method: "standard" (zero mean/unit variance) | "minmax" (rescale to [0,1])
  Do NOT scale id-like columns (is_likely_id=true).
"""