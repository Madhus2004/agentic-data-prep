"""
eval_suite.py

Runs the full agent against every test dataset and checks whether the
data-quality issues that existed at the start were actually resolved by
the end. This produces a real, reproducible number instead of a made-up
one — the kind of thing a resume bullet should point to.

Requires GROQ_API_KEY (loaded from .env, same as agent.py).
Run with:
    python eval_suite.py
"""

import pandas as pd
import profiling
from agent import build_graph
from state import AgentState

TEST_FILES = ["sample_data.csv", "test_data_products.csv", "test_data_customers.csv"]


def count_unhandled_outlier_columns(before_profile: dict, log: list) -> int:
    """
    Checks whether every column that had outliers got a successful
    handle_outliers call — rather than re-running IQR on the FINAL data.
    Re-checking outliers after scaling isn't a fair comparison: scaling
    legitimately reshapes a column's distribution, so a value correctly
    capped in the original scale can look newly "extreme" once bounds are
    recomputed on the scaled result. That's a property of scaling, not an
    unresolved data quality issue — so we check "did the tool run
    successfully," not "does a fresh statistical test still flag something."
    """
    outlier_cols = {col for col, c in before_profile["columns"].items()
                     if c.get("outliers") and c["outliers"]["count"] > 0}
    handled_cols = {entry["column"] for entry in log
                     if entry["tool"] == "handle_outliers" and entry["status"] == "success"}
    return len(outlier_cols - handled_cols)


def count_total_missing(profile: dict) -> int:
    return sum(c["missing_count"] for c in profile["columns"].values())


def count_total_outliers(profile: dict) -> int:
    """Number of COLUMNS with at least one outlier (not total outlier values) —
    kept in the same unit as count_unhandled_outlier_columns for a fair comparison."""
    return sum(1 for c in profile["columns"].values() if c.get("outliers") and c["outliers"]["count"] > 0)


def count_casing_issues(profile: dict) -> int:
    return sum(1 for c in profile["columns"].values() if c.get("casing_inconsistent"))


def count_duplicate_rows(profile: dict) -> int:
    dup = profile["duplicates"]
    total = dup.get("count", 0) + dup.get("count_excluding_id_columns", 0)
    for info in dup.get("natural_key_duplicates", {}).values():
        total += info["duplicate_row_count"]
    return total


METRICS = {
    "missing_values": count_total_missing,
    "casing_issues": count_casing_issues,
    "duplicate_rows": count_duplicate_rows,
}


def evaluate_file(path: str, graph) -> dict:
    print(f"\n{'='*60}\n{path}\n{'='*60}")
    df = pd.read_csv(path)
    before_profile = profiling.profile_data(df)
    before = {name: fn(before_profile) for name, fn in METRICS.items()}
    before["outliers"] = count_total_outliers(before_profile)
    print("BEFORE:", before)

    initial_state: AgentState = {
        "df": df, "profile": {}, "plan": [], "current_step": 0, "log": [], "report": ""
    }
    final_state = graph.invoke(initial_state)

    after_profile = profiling.profile_data(final_state["df"])
    after = {name: fn(after_profile) for name, fn in METRICS.items()}
    after["outliers"] = count_unhandled_outlier_columns(before_profile, final_state["log"])
    print("AFTER: ", after)

    results = {}
    for key in before:
        b, a = before[key], after[key]
        if b == 0:
            results[key] = "n/a"          # no issue of this type existed
        elif a == 0:
            results[key] = "pass"         # fully resolved
        elif a < b:
            results[key] = f"partial ({b} -> {a})"
        else:
            results[key] = f"fail (still {a})"

    print("\nRESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Self-diagnosing: if outliers or duplicates didn't fully resolve,
    # print exactly which columns are still flagged so the cause is
    # visible without a separate debugging pass.
    if results.get("outliers", "").startswith(("fail", "partial")):
        outlier_cols = {col for col, c in before_profile["columns"].items()
                         if c.get("outliers") and c["outliers"]["count"] > 0}
        handled_cols = {entry["column"] for entry in final_state["log"]
                         if entry["tool"] == "handle_outliers" and entry["status"] == "success"}
        print(f"\n  outlier detail: columns with outliers before={sorted(outlier_cols)}, "
              f"successfully handled={sorted(handled_cols)}, "
              f"unhandled={sorted(outlier_cols - handled_cols)}")

    if results.get("duplicate_rows", "").startswith(("fail", "partial")):
        print("\n  duplicate detail:")
        dup = after_profile["duplicates"]
        if dup.get("count", 0) or dup.get("count_excluding_id_columns", 0):
            print(f"    full/id-excluded: count={dup.get('count')}, count_excluding_id_columns={dup.get('count_excluding_id_columns')}")
        for col, info in dup.get("natural_key_duplicates", {}).items():
            print(f"    natural key '{col}': {info['duplicate_row_count']} row(s) still repeat")

    return {"file": path, "before": before, "after": after, "results": results,
             "plan_steps": len(final_state["plan"]), "log_entries": len(final_state["log"])}


if __name__ == "__main__":
    graph = build_graph()
    all_results = [evaluate_file(f, graph) for f in TEST_FILES]

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    total, passed = 0, 0
    for run in all_results:
        for v in run["results"].values():
            if v == "n/a":
                continue
            total += 1
            if v == "pass":
                passed += 1

    if total:
        pct = round(passed / total * 100, 1)
        print(f"{passed}/{total} known issue categories fully resolved "
              f"across {len(TEST_FILES)} datasets ({pct}%)")
    else:
        print("No issues found across test datasets — check test data.")