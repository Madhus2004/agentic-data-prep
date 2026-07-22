"""
profiling.py

Deterministic data profiling — no LLM involved here.
Takes a DataFrame, returns a structured dict describing what's wrong with it.
This dict is what gets handed to the planner (LLM) later, NOT the raw data,
so the agent reasons over a compact summary instead of burning tokens on
every row.
"""

import re
import pandas as pd
import numpy as np

_SLASH_DATE_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")


def profile_data(df: pd.DataFrame) -> dict:
    """
    Run a full profile pass on a DataFrame.

    Returns a dict shaped like:
    {
        "shape": {"rows": int, "columns": int},
        "duplicates": {"count": int, "pct": float},
        "columns": {
            "<col_name>": {
                "dtype": str,
                "missing_count": int,
                "missing_pct": float,
                "unique_count": int,
                "is_likely_id": bool,
                "is_likely_categorical": bool,
                "outliers": {...} or None,       # only for numeric columns
                "casing_inconsistent": bool or None,  # only for string columns
                "sample_values": list
            },
            ...
        }
    }
    """
    profile = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "duplicates": _profile_duplicates(df),
        "columns": {}
    }

    for col in df.columns:
        profile["columns"][col] = _profile_column(df, col)

    return profile


def _profile_duplicates(df: pd.DataFrame) -> dict:
    dup_count = int(df.duplicated().sum())
    return {
        "count": dup_count,
        "pct": round(dup_count / len(df) * 100, 2) if len(df) else 0.0
    }


def _profile_column(df: pd.DataFrame, col: str) -> dict:
    series = df[col]
    missing_count = int(series.isna().sum())
    n = len(series)

    result = {
        "dtype": str(series.dtype),
        "missing_count": missing_count,
        "missing_pct": round(missing_count / n * 100, 2) if n else 0.0,
        "unique_count": int(series.nunique(dropna=True)),
        "is_likely_id": _is_likely_id(series, col),
        "is_likely_categorical": _is_likely_categorical(series),
        "outliers": None,
        "casing_inconsistent": None,
        "sample_values": series.dropna().unique()[:5].tolist()
    }

    if pd.api.types.is_numeric_dtype(series):
        result["outliers"] = _detect_outliers_iqr(series)
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        result["casing_inconsistent"] = _check_casing_inconsistent(series)
        result["date_ambiguity"] = _check_date_ambiguity(series, col)

    return result


def _is_likely_id(series: pd.Series, col_name: str) -> bool:
    name_hints = col_name.lower() in ("id", "index", "uuid") or col_name.lower().endswith("_id")
    high_cardinality = series.nunique(dropna=True) >= len(series) * 0.95
    return bool(name_hints or high_cardinality)


def _is_likely_categorical(series: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    n_unique = series.nunique(dropna=True)
    return 1 < n_unique <= 20


def _detect_outliers_iqr(series: pd.Series) -> dict:
    """IQR method: flags values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    clean = series.dropna()
    if len(clean) < 4:
        return {"count": 0, "bounds": None, "values": []}

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = clean[(clean < lower) | (clean > upper)]

    return {
        "count": int(len(outliers)),
        "bounds": {"lower": round(float(lower), 2), "upper": round(float(upper), 2)},
        "values": outliers.tolist()[:10]  # cap so the profile stays compact
    }


def _check_casing_inconsistent(series: pd.Series) -> bool:
    """
    Flags columns where the same logical value appears in different cases,
    e.g. 'USA' and 'usa' being treated as distinct when they shouldn't be.
    """
    clean = series.dropna().astype(str)
    if clean.empty:
        return False
    lowered_unique = clean.str.lower().nunique()
    original_unique = clean.nunique()
    return lowered_unique < original_unique


def _check_date_ambiguity(series: pd.Series, col_name: str) -> dict | None:
    """
    Flags date-like string columns where DD/MM vs MM/DD can't be told apart
    from the data alone (e.g. '03/04/2023'), and columns that mix ISO
    ('2023-01-15') with slash-separated formats. Only runs on columns that
    look date-like, either by name or by matching a date-shaped pattern in
    most of their values.
    """
    clean = series.dropna().astype(str)
    if clean.empty:
        return None

    name_hint = any(k in col_name.lower() for k in ("date", "dob", "day", "_at", "time"))
    slash_matches = clean.apply(lambda v: _SLASH_DATE_RE.match(v))
    iso_matches = clean.apply(lambda v: bool(_ISO_DATE_RE.match(v)))

    slash_hit_rate = slash_matches.notna().mean()
    looks_date_like = name_hint or slash_hit_rate > 0.3 or iso_matches.mean() > 0.3

    if not looks_date_like:
        return None

    ambiguous_values = []
    forced_dayfirst = False
    for v, m in zip(clean, slash_matches):
        if m is None:
            continue
        first, second = int(m.group(1)), int(m.group(2))
        if first > 12:
            forced_dayfirst = True  # unambiguous: only valid as day-first
        elif first <= 12 and second <= 12:
            ambiguous_values.append(v)  # could be read either way

    mixed_formats = bool(slash_matches.notna().any()) and bool(iso_matches.any())

    return {
        "is_likely_date": True,
        "mixed_formats": mixed_formats,
        "ambiguous_count": len(ambiguous_values),
        "ambiguous_sample": ambiguous_values[:5],
        "dayfirst_evidence": forced_dayfirst,
    }


if __name__ == "__main__":
    df = pd.read_csv("sample_data.csv")
    import json
    result = profile_data(df)
    print(json.dumps(result, indent=2, default=str))