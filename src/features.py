"""Engineered features, applied inside the model Pipeline.

All three are row-wise formulas: they learn nothing from the data, so they
cannot leak information between train and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from src import config

ENGINEERED_NUMERIC = ["num_addon_services", "avg_monthly_spend"]
ENGINEERED_CATEGORICAL = ["tenure_bucket"]

TENURE_BINS = [-0.5, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49m+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the engineered columns appended."""
    out = df.copy()
    out["num_addon_services"] = (
        (out[config.ADDON_SERVICES] == "Yes").sum(axis=1).astype(float)
    )
    # Average spend per month of tenure. Tenure 0 is treated as 1 month.
    out["avg_monthly_spend"] = out["TotalCharges"] / out["tenure"].clip(lower=1)
    bucket = pd.cut(out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    out["tenure_bucket"] = bucket.astype(object).where(bucket.notna(), np.nan)
    return out


def feature_engineering_step() -> FunctionTransformer:
    return FunctionTransformer(add_features, validate=False)


def model_input_columns(engineered: bool) -> tuple[list[str], list[str]]:
    """(numeric, categorical) columns the preprocessor should see."""
    numeric = list(config.NUMERIC_COLS)
    categorical = list(config.CATEGORICAL_COLS)
    if engineered:
        numeric += ENGINEERED_NUMERIC
        categorical += ENGINEERED_CATEGORICAL
    return numeric, categorical
