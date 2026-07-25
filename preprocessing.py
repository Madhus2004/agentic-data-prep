"""
preprocessing.py

Preprocessing actions — the step after cleaning. Where tools.py fixes
*correctness* (missing values, duplicates, wrong types, outliers), this
module prepares clean data for modeling: encoding categories into numbers,
scaling numeric ranges. Same (df, log_entry) contract as tools.py so the
agent calls these identically.
"""

import pandas as pd
import numpy as np


def encode_categorical(df: pd.DataFrame, column: str, method: str = "onehot") -> tuple[pd.DataFrame, dict]:
    """
    Convert a categorical column into a numeric representation.
    method: 'onehot' (one column per category) or 'label' (single integer column)

    Use 'onehot' for nominal categories with no order (e.g. country).
    Use 'label' for ordinal categories, or when cardinality is too high
    for one-hot to be practical.
    """
    df = df.copy()
    n_categories = df[column].nunique(dropna=True)

    if method == "onehot":
        dummies = pd.get_dummies(df[column], prefix=column, dtype=int)
        df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
        detail = f"one-hot encoded into {len(dummies.columns)} columns ({n_categories} categories)"
    elif method == "label":
        codes, uniques = pd.factorize(df[column]) #output is like [0,1,2,...] and [unique values in that column]
        df[column] = codes
        mapping = {val: i for i, val in enumerate(uniques)}
        detail = f"label encoded {n_categories} categories, mapping: {mapping}"
    else:
        return df, _log("encode_categorical", column, "failed", f"unknown method '{method}'")

    return df, _log("encode_categorical", column, "success", detail)


def scale_numeric(df: pd.DataFrame, column: str, method: str = "standard") -> tuple[pd.DataFrame, dict]:
    """
    Rescale a numeric column.
    method: 'standard' (zero mean, unit variance) or 'minmax' (rescale to [0, 1])
    """
    df = df.copy()

    if not pd.api.types.is_numeric_dtype(df[column]):
        return df, _log("scale_numeric", column, "failed", "column is not numeric")

    if method == "standard":
        mean, std = df[column].mean(), df[column].std()
        if std == 0:
            return df, _log("scale_numeric", column, "skipped", "zero variance, scaling would divide by zero")
        df[column] = ((df[column] - mean) / std).round(4)
        detail = f"standard scaled (mean={round(mean,2)}, std={round(std,2)})"
    elif method == "minmax":
        col_min, col_max = df[column].min(), df[column].max()
        if col_min == col_max:
            return df, _log("scale_numeric", column, "skipped", "constant column, scaling would divide by zero")
        df[column] = ((df[column] - col_min) / (col_max - col_min)).round(4)
        detail = f"min-max scaled (min={round(col_min,2)}, max={round(col_max,2)})"
    else:
        return df, _log("scale_numeric", column, "failed", f"unknown method '{method}'")

    return df, _log("scale_numeric", column, "success", detail)


def _log(tool: str, column, status: str, detail: str) -> dict:
    return {"tool": tool, "column": column, "status": status, "detail": detail}


if __name__ == "__main__":
    import tools

    df = pd.read_csv("sample_data.csv")

    # run the full cleaning pipeline first, so preprocessing starts from clean data
    df, log1 = tools.impute_missing(df, "age", strategy="median")
    df, log2 = tools.impute_missing(df, "salary", strategy="median")
    df, log3 = tools.handle_outliers(df, "age", method="cap")
    df, log4 = tools.handle_outliers(df, "salary", method="cap")
    df, log5 = tools.standardize_text(df, "country", case="upper")
    df, log6 = tools.remove_duplicates(df, subset=["email"])
    df, log7 = tools.fix_dtype(df, "signup_date", "datetime", dayfirst=True)

    # then preprocess
    df, log8 = scale_numeric(df, "age", method="standard")
    df, log9 = scale_numeric(df, "salary", method="minmax")
    df, log10 = encode_categorical(df, "country", method="onehot")

    for log in [log1, log2, log3, log4, log5, log6, log7, log8, log9, log10]:
        print(log)

    print("\nfinal shape:", df.shape)
    print(df.head())