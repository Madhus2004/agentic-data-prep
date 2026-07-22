"""
tools.py

The cleaning "actions" the agent can take. Each function:
  - takes the DataFrame (and sometimes a column name / params)
  - returns (new_df, log_entry)

log_entry is a dict describing what happened — this feeds directly into
the final report, so it should be human-readable, not just a status code.

These are plain functions for now. Once the LangGraph agent is wired up,
each one gets wrapped with a docstring-based tool schema so the LLM can
call it by name.
"""

import re
import pandas as pd
import numpy as np


def impute_missing(df: pd.DataFrame, column: str, strategy: str = "median") -> tuple[pd.DataFrame, dict]:
    """
    Fill missing values in a numeric or categorical column.
    strategy: 'median', 'mean', 'mode', or 'drop_rows'
    """
    df = df.copy()
    missing_before = int(df[column].isna().sum())

    if missing_before == 0:
        return df, _log("impute_missing", column, "skipped", "no missing values found")

    if strategy == "drop_rows":
        df = df.dropna(subset=[column])
        detail = f"dropped {missing_before} rows missing '{column}'"
    elif strategy == "mode":
        fill_value = df[column].mode(dropna=True)
        fill_value = fill_value.iloc[0] if not fill_value.empty else None
        df[column] = df[column].fillna(fill_value)
        detail = f"filled {missing_before} missing values with mode ({fill_value!r})"
    elif strategy in ("median", "mean"):
        if not pd.api.types.is_numeric_dtype(df[column]):
            return df, _log("impute_missing", column, "failed",
                             f"strategy '{strategy}' requires a numeric column")
        fill_value = df[column].median() if strategy == "median" else df[column].mean()
        fill_value = round(float(fill_value), 2)
        df[column] = df[column].fillna(fill_value)
        detail = f"filled {missing_before} missing values with {strategy} ({fill_value})"
    else:
        return df, _log("impute_missing", column, "failed", f"unknown strategy '{strategy}'")

    return df, _log("impute_missing", column, "success", detail)


def remove_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Remove exact duplicate rows. Optionally restrict comparison to a subset
    of columns (e.g. ignore an 'id' column that's always unique).
    """
    df = df.copy()
    dup_count = int(df.duplicated(subset=subset).sum())

    if dup_count == 0:
        return df, _log("remove_duplicates", subset or "all columns", "skipped", "no duplicates found")

    df = df.drop_duplicates(subset=subset, keep="first")
    detail = f"removed {dup_count} duplicate rows"
    return df, _log("remove_duplicates", subset or "all columns", "success", detail)


def fix_dtype(df: pd.DataFrame, column: str, target_type: str, dayfirst: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Convert a column to the correct dtype.
    target_type: 'int', 'float', 'datetime', 'str'

    dayfirst only applies to target_type='datetime'. Ambiguous dates like
    '03/04/2023' can validly mean either March 4 or April 3 — there is no
    way to infer this correctly from the data alone. dayfirst defaults to
    True (day/month/year), matching common non-US conventions; pass
    dayfirst=False for US-style month/day/year data. This choice is always
    recorded in the log so it's never a silent guess.
    """
    df = df.copy()
    original_dtype = str(df[column].dtype)

    try:
        if target_type == "datetime":
            df[column] = _parse_dates(df[column], dayfirst=dayfirst)
        elif target_type == "int":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        elif target_type == "float":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif target_type == "str":
            df[column] = df[column].astype(str)
        else:
            return df, _log("fix_dtype", column, "failed", f"unknown target_type '{target_type}'")
    except Exception as e:
        return df, _log("fix_dtype", column, "failed", f"conversion error: {e}")

    new_failures = int(df[column].isna().sum())
    detail = f"converted from {original_dtype} to {target_type}"
    if target_type == "datetime":
        detail += f" (assumed day-first format: {dayfirst})"
    if new_failures:
        detail += f" ({new_failures} values could not be parsed and became missing)"

    return df, _log("fix_dtype", column, "success", detail)


def handle_outliers(df: pd.DataFrame, column: str, method: str = "cap") -> tuple[pd.DataFrame, dict]:
    """
    Handle outliers in a numeric column using the IQR bounds.
    method: 'cap' (clip to bounds) or 'remove' (drop those rows)
    """
    df = df.copy()
    clean = df[column].dropna()
    if len(clean) < 4:
        return df, _log("handle_outliers", column, "skipped", "not enough data to compute bounds")

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_mask = (df[column] < lower) | (df[column] > upper)
    outlier_count = int(outlier_mask.sum())

    if outlier_count == 0:
        return df, _log("handle_outliers", column, "skipped", "no outliers found")

    if method == "cap":
        df[column] = df[column].clip(lower=lower, upper=upper)
        detail = f"capped {outlier_count} outlier(s) to range [{round(lower,2)}, {round(upper,2)}]"
    elif method == "remove":
        df = df[~outlier_mask]
        detail = f"removed {outlier_count} row(s) with outlier values"
    else:
        return df, _log("handle_outliers", column, "failed", f"unknown method '{method}'")

    return df, _log("handle_outliers", column, "success", detail)


def standardize_text(df: pd.DataFrame, column: str, case: str = "lower", strip_whitespace: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Normalize text formatting in a column: fix casing and stray whitespace
    so values like 'USA' and 'usa ' are treated as the same category.
    case: 'lower', 'upper', or 'title'
    """
    df = df.copy()
    before_unique = int(df[column].nunique(dropna=True))

    series = df[column].astype(str)
    if strip_whitespace:
        series = series.str.strip()
    if case == "lower":
        series = series.str.lower()
    elif case == "upper":
        series = series.str.upper()
    elif case == "title":
        series = series.str.title()
    else:
        return df, _log("standardize_text", column, "failed", f"unknown case '{case}'")

    df[column] = series
    after_unique = int(df[column].nunique(dropna=True))
    merged = before_unique - after_unique

    detail = f"standardized casing/whitespace"
    if merged > 0:
        detail += f" (merged {merged} duplicate category values, e.g. 'USA'/'usa' -> one value)"

    return df, _log("standardize_text", column, "success", detail)


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")


def _parse_dates(series: pd.Series, dayfirst: bool) -> pd.Series:
    """
    Parse a mixed-format date column safely.

    Workaround for a pandas quirk (observed on pandas 3.0.2): passing
    dayfirst=True to pd.to_datetime(..., format="mixed") can incorrectly
    swap month/day on unambiguous ISO strings too (e.g. '2023-03-10'
    becoming Oct 3 instead of Mar 10). ISO 'YYYY-MM-DD' is never ambiguous,
    so it's parsed with an explicit fixed format, immune to dayfirst.
    dayfirst is only applied to the genuinely ambiguous remainder
    (e.g. '15/04/2023').
    """
    s = series.astype(str)
    iso_mask = s.str.match(_ISO_DATE_RE)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if iso_mask.any():
        result.loc[iso_mask] = pd.to_datetime(s[iso_mask], format="%Y-%m-%d", errors="coerce")
    if (~iso_mask).any():
        result.loc[~iso_mask] = pd.to_datetime(s[~iso_mask], errors="coerce", dayfirst=dayfirst)

    return result


def _log(tool: str, column, status: str, detail: str) -> dict:
    return {"tool": tool, "column": column, "status": status, "detail": detail}


if __name__ == "__main__":
    df = pd.read_csv("sample_data.csv")

    df, log1 = impute_missing(df, "age", strategy="median")
    df, log2 = impute_missing(df, "salary", strategy="median")
    df, log3 = handle_outliers(df, "age", method="cap")
    df, log4 = handle_outliers(df, "salary", method="cap")
    df, log5 = standardize_text(df, "country", case="upper")
    df, log6 = remove_duplicates(df, subset=["email"])

    for log in [log1, log2, log3, log4, log5, log6]:
        print(log)

    print("\nfinal shape:", df.shape)
    print(df[["age", "salary", "country"]])