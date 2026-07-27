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
        "columns": {}
    }

    for col in df.columns:
        profile["columns"][col] = _profile_column(df, col)

    id_like_cols = [c for c in df.columns if profile["columns"][c]["is_likely_id"]]
    structural_id_cols = [c for c in df.columns if _is_structural_id(c)]
    profile["duplicates"] = _profile_duplicates(df, structural_id_cols, profile["columns"])

    return profile


def _profile_duplicates(df: pd.DataFrame, id_like_cols: list, profile_columns: dict) -> dict:
    dup_count = int(df.duplicated().sum())
    result = {
        "count": dup_count,
        "pct": round(dup_count / len(df) * 100, 2) if len(df) else 0.0,
    }

    compare_cols = [c for c in df.columns if c not in id_like_cols]
    if compare_cols:
        dup_excl_id = int(df.duplicated(subset=compare_cols).sum())
        result["count_excluding_id_columns"] = dup_excl_id
        result["id_columns_excluded"] = id_like_cols
        if dup_excl_id > dup_count:
            result["note"] = (
                f"{dup_excl_id} row(s) are identical except for an id-like column "
                f"({id_like_cols}) — likely true duplicates. Use remove_duplicates "
                f"with subset={compare_cols} to catch these."
            )

    # Natural-key check: a column can repeat (same person, same email) even
    # when OTHER columns differ (e.g. salary filled in on a later row and
    # missing on an earlier one) — a full-row comparison misses this
    # entirely. Check high-uniqueness, non-id text columns individually.
    key_candidates = {}
    for col, col_profile in profile_columns.items():
        if col in id_like_cols:
            continue
        if col_profile["is_likely_categorical"]:
            continue  # values are SUPPOSED to repeat in a category column

        is_text = col_profile["dtype"] in ("str", "object")
        if not is_text:
            # Numeric columns only qualify if they're identifier-like by
            # name (e.g. "phone", "account_number"). A plain numeric
            # measurement (age, salary, price) coincidentally sharing a
            # value across two rows is completely normal in a small
            # dataset and is NOT evidence those rows are duplicates.
            if not col_profile["is_likely_id"]:
                continue
        n = len(df)
        if n == 0:
            continue
        uniqueness_ratio = col_profile["unique_count"] / n
        if uniqueness_ratio < 0.7:  # not unique enough to plausibly be a key
            continue
        non_null = df[col].notna()
        key_dup_count = int(df.loc[non_null, col].duplicated(keep=False).sum())
        if key_dup_count > 0:
            key_candidates[col] = {
                "duplicate_row_count": key_dup_count,
                "note": (
                    f"'{col}' repeats in {key_dup_count} row(s) even though other "
                    f"columns differ — check if these are the same entity recorded "
                    f"twice, e.g. with one row missing data the other has filled in."
                )
            }
    if key_candidates:
        result["natural_key_duplicates"] = key_candidates

    return result


def _profile_column(df: pd.DataFrame, col: str) -> dict:
    series = df[col]
    missing_count = int(series.isna().sum())
    n = len(series)

    result = {
        "dtype": str(series.dtype),
        "missing_count": missing_count,
        "missing_pct": round(missing_count / n * 100, 2) if n else 0.0,
        "unique_count": int(series.nunique(dropna=True)),
        "unique_count_normalized": _normalized_unique_count(series),
        "is_likely_id": _is_likely_id(series, col),
        "is_likely_categorical": _is_likely_categorical(series),
        "outliers": None,
        "casing_inconsistent": None,
        "sample_values": series.dropna().unique()[:5].tolist()
    }

    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if non_null.nunique() <= 2:
            # Binary/near-constant column (e.g. a one-hot encoded flag).
            # IQR outlier detection is meaningless here — the rare class
            # in a 0/1 column isn't a data quality problem, it's just an
            # imbalanced category.
            result["outliers"] = {"count": 0, "bounds": None, "values": [],
                                   "note": "binary/near-constant column, IQR outlier check skipped"}
        else:
            result["outliers"] = _detect_outliers_iqr(series)
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        result["casing_inconsistent"] = _check_casing_inconsistent(series)
        result["date_ambiguity"] = _check_date_ambiguity(series, col)

    return result


def _normalized_unique_count(series: pd.Series) -> int:
    """
    Counts unique values after lowercasing/stripping text — so 'Chennai'
    and 'chennai' count as ONE value, not two. Without this, a column with
    casing inconsistency looks artificially more unique than it really is,
    which can wrongly make a genuine category column (like city or
    country) look like a near-unique identifier. Numeric columns are
    unaffected — casing doesn't apply to them.
    """
    clean = series.dropna()
    if pd.api.types.is_object_dtype(clean) or pd.api.types.is_string_dtype(clean):
        return int(clean.astype(str).str.strip().str.lower().nunique())
    return int(clean.nunique())


def _is_structural_id(col_name: str) -> bool:
    """
    True only for meaningless, auto-increment-style identifiers (id, uuid,
    index, foo_id) — the kind of column that's expected to be unique on
    every single row, including for two rows that are otherwise true
    duplicates. These are useless as a duplicate-detection key (comparing
    on them would never find a match) so they're excluded from that check.
    This is deliberately narrower than is_likely_id: a natural identifier
    like 'phone' or 'email' should NOT be scaled/encoded as a model
    feature, but SHOULD still be usable to catch the same real-world
    entity being entered twice — so it stays eligible for the natural-key
    duplicate check even though is_likely_id is true for it.
    """
    name = col_name.lower()
    return name in ("id", "index", "uuid") or name.endswith("_id")


def _is_likely_id(series: pd.Series, col_name: str) -> bool:
    name_hints = (
        col_name.lower() in ("id", "index", "uuid", "phone", "zip", "zipcode",
                              "pincode", "postal_code", "account_number")
        or col_name.lower().endswith("_id")
        or col_name.lower().endswith("_number")
    )
    n = len(series.dropna())
    high_cardinality = n > 0 and _normalized_unique_count(series) >= n * 0.95
    return bool(name_hints or high_cardinality)


def _is_likely_categorical(series: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    clean = series.dropna()
    n = len(clean)
    if n == 0:
        return False
    n_unique = _normalized_unique_count(series)
    if n_unique <= 1:
        return False
    uniqueness_ratio = n_unique / n
    return n_unique <= 20 and uniqueness_ratio <= 0.5


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


def summarize_issues(profile: dict) -> list[str]:
    """
    Converts the profile dict into explicit, unambiguous findings in plain
    English. This exists because handing an LLM a big nested JSON blob and
    expecting it to correctly infer "these two fields together mean X" is
    unreliable in practice — it's easy for a model to encode an id-like
    column anyway, or miss a duplicate signal buried three keys deep, even
    when temperature=0. Pre-computing the conclusion in Python and stating
    it as a sentence removes that failure mode: the LLM's job becomes
    "turn this finding into a tool call", not "notice the finding exists".
    """
    issues = []
    cols = profile["columns"]
    dup = profile["duplicates"]

    if dup["count"] > 0:
        issues.append(f"{dup['count']} exact duplicate row(s) found across all columns.")

    if dup.get("count_excluding_id_columns", 0) > dup["count"]:
        non_id_cols = [c for c in cols if c not in dup.get("id_columns_excluded", [])]
        issues.append(
            f"{dup['count_excluding_id_columns']} row(s) are identical except for an "
            f"id-like column {dup.get('id_columns_excluded')}. Call remove_duplicates "
            f"with subset={non_id_cols} to remove these."
        )

    for key_col, info in dup.get("natural_key_duplicates", {}).items():
        issues.append(
            f"Column '{key_col}' has the same value repeated across "
            f"{info['duplicate_row_count']} rows even though other columns differ "
            f"(likely the same real-world entity entered twice). Call "
            f"remove_duplicates with subset=['{key_col}'] to deduplicate on it."
        )

    for col, c in cols.items():
        if c["missing_count"] > 0:
            if c["is_likely_categorical"]:
                suggestion = "Use impute_missing with strategy='mode' — this is a low-cardinality category, so a most-common value makes sense."
            elif c["dtype"] in ("str", "object"):
                suggestion = (
                    "Do NOT use strategy='mode' — this column is near-unique (like a name "
                    "or email), so there's no real 'most common' value, and copying another "
                    "row's value in would fabricate a fake duplicate. Use strategy='placeholder' instead."
                )
            else:
                suggestion = "Use impute_missing with strategy='median' (or 'mean') since this is numeric."
            issues.append(f"Column '{col}' has {c['missing_count']} missing value(s) ({c['missing_pct']}%). {suggestion}")

        if c.get("casing_inconsistent"):
            issues.append(f"Column '{col}' has inconsistent text casing (e.g. mixed case for the same value).")

        da = c.get("date_ambiguity")
        if da and da.get("is_likely_date"):
            if da.get("mixed_formats"):
                issues.append(f"Column '{col}' looks like a date but mixes formats (e.g. ISO and DD/MM/YYYY).")
            if da.get("ambiguous_count", 0) > 0:
                issues.append(
                    f"Column '{col}' has {da['ambiguous_count']} genuinely ambiguous date value(s) "
                    f"(could be DD/MM or MM/DD). Convert with fix_dtype, choosing dayfirst explicitly."
                )

        if c.get("outliers") and c["outliers"]["count"] > 0:
            issues.append(f"Column '{col}' has {c['outliers']['count']} outlier value(s) detected via IQR.")

        if c["is_likely_id"]:
            issues.append(f"Column '{col}' is an identifier column. Do NOT encode or scale it — exclude it from preprocessing entirely.")
        elif c["is_likely_categorical"]:
            issues.append(f"Column '{col}' is categorical ({c['unique_count_normalized']} unique values after normalizing casing) and is a good candidate for encode_categorical, once cleaned.")
        elif c["dtype"] in ("str", "object"):
            issues.append(
                f"Column '{col}' is free text with high uniqueness ({c['unique_count']} unique values) "
                f"and is NOT categorical — likely a name, email, or other identifier-like field. "
                f"Do NOT encode or scale this column."
            )
        elif c["dtype"] in ("int64", "float64", "Int64", "int32", "float32"):
            issues.append(f"Column '{col}' is numeric and a good candidate for scale_numeric, once cleaned (missing values filled, outliers handled).")

    return issues


if __name__ == "__main__":
    df = pd.read_csv("sample_data.csv")
    import json
    result = profile_data(df)
    print(json.dumps(result, indent=2, default=str))