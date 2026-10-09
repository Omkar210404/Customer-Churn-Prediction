"""Cleaning, input validation and the sklearn preprocessing transformer.

Two kinds of steps live here:

* Deterministic cleaning (`clean_raw`, `validate_features`): type fixes and
  value normalisation that learn nothing from the data, so they are safe to
  apply before the train/test split and identically at prediction time.
* Learned steps (`build_preprocessor`): imputers, scaler and one-hot encoder.
  These only ever live inside a Pipeline that is fit on training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


@dataclass
class CleaningReport:
    """Human-readable log of every change `clean_raw` makes."""

    steps: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.steps.append(message)


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def clean_raw(raw: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    """Clean the raw Telco CSV (all-text columns) into model-ready features + target.

    No rows are dropped. Every change is recorded in `report`.
    """
    report = report if report is not None else CleaningReport()
    expected = set(config.FEATURE_COLS) | {config.ID_COL, config.TARGET}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")

    df = _strip_strings(raw)
    report.add(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")

    dup_ids = int(df[config.ID_COL].duplicated().sum())
    report.add(f"Duplicate customerIDs: {dup_ids}.")
    dup_rows = int(df.drop(columns=[config.ID_COL]).duplicated().sum())
    report.add(
        f"Rows identical to an earlier row on every column except customerID: {dup_rows}. "
        "Kept: they have distinct customer IDs, so they are separate customers."
    )

    # TotalCharges is text in the raw file; blanks belong to customers with tenure 0.
    blank_tc = df["TotalCharges"].eq("")
    tenure_num = pd.to_numeric(df["tenure"], errors="raise")
    n_blank = int(blank_tc.sum())
    if n_blank and not (tenure_num[blank_tc] == 0).all():
        raise ValueError("Found blank TotalCharges for customers with tenure > 0; rule does not hold.")
    df.loc[blank_tc, "TotalCharges"] = "0"
    report.add(
        f"TotalCharges blank in {n_blank} rows, all with tenure 0 (not yet billed): set to 0.0."
    )

    for col in config.NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)

    df["SeniorCitizen"] = df["SeniorCitizen"].map({"0": "No", "1": "Yes"})
    report.add("SeniorCitizen 0/1 mapped to No/Yes to match the other binary columns.")

    for col, allowed in config.CATEGORICAL_VALUES.items():
        bad = set(df[col].dropna().unique()) - set(allowed)
        if bad:
            raise ValueError(f"Unexpected values in {col}: {sorted(bad)}")

    df[config.TARGET] = df[config.TARGET].map({"No": 0, "Yes": 1})
    if df[config.TARGET].isna().any():
        raise ValueError("Target column contains values other than Yes/No.")
    df[config.TARGET] = df[config.TARGET].astype(int)
    report.add("Target Churn mapped Yes -> 1, No -> 0.")

    df = df.drop(columns=[config.ID_COL])
    report.add("Dropped customerID (identifier, not a feature).")
    report.add(f"Result: {len(df):,} rows, {len(config.FEATURE_COLS)} features + target.")
    return df[config.FEATURE_COLS + [config.TARGET]]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.FEATURE_COLS].copy(), df[config.TARGET].copy()


_MISSING_TOKENS = {"", "nan", "none", "null", "na"}


def validate_features(data: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
    """Validate and normalise customer records for prediction.

    Accepts a DataFrame, one dict or a list of dicts. Missing values (None, NaN,
    empty string) are allowed and left as NaN for the pipeline's imputers.
    Raises ValueError for unknown columns, invalid categories, non-numeric
    numbers or negative amounts.
    """
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
    if df.empty:
        raise ValueError("No records to validate.")

    df = df.drop(columns=[c for c in (config.ID_COL, config.TARGET) if c in df.columns])
    unknown = set(df.columns) - set(config.FEATURE_COLS)
    if unknown:
        raise ValueError(f"Unknown columns: {sorted(unknown)}")
    for col in config.FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = _strip_strings(df)
    df = df.map(lambda v: np.nan if isinstance(v, str) and v.lower() in _MISSING_TOKENS else v)

    if "SeniorCitizen" in df:
        df["SeniorCitizen"] = df["SeniorCitizen"].map(
            lambda v: {0: "No", 1: "Yes", "0": "No", "1": "Yes"}.get(v, v)
        )

    for col, allowed in config.CATEGORICAL_VALUES.items():
        values = df[col].dropna()
        bad = sorted({str(v) for v in values if v not in allowed})
        if bad:
            raise ValueError(f"Invalid value(s) for {col}: {bad}. Allowed: {allowed}")
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    for col in config.NUMERIC_COLS:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{col} must be numeric: {exc}") from None
        if (df[col] < 0).any():
            raise ValueError(f"{col} cannot be negative.")
    if (df["tenure"] > 120).any():
        raise ValueError("tenure is in months and cannot exceed 120.")

    return df[config.FEATURE_COLS]


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot encode categoricals."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, categorical_cols)],
        verbose_feature_names_out=True,
    )
