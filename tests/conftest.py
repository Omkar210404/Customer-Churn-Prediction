"""Shared fixtures.

The records below are hand-written TEST FIXTURES in the Telco schema, not real
customers. They exist so the unit tests run without downloading the dataset.
"""

import pandas as pd
import pytest

BASE_RECORD = {
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": "12",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": "55.0",
    "TotalCharges": "660.0",
}

# (overrides, churn) pairs: churners are short-tenure, month-to-month, fibre.
_VARIANTS = [
    ({}, "No"),
    ({"tenure": "60", "Contract": "Two year", "TotalCharges": "3300.0"}, "No"),
    ({"tenure": "0", "TotalCharges": " "}, "No"),
    ({"InternetService": "No", "OnlineSecurity": "No internet service",
      "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
      "TechSupport": "No internet service", "StreamingTV": "No internet service",
      "StreamingMovies": "No internet service", "MonthlyCharges": "20.0",
      "TotalCharges": "240.0"}, "No"),
    ({"PhoneService": "No", "MultipleLines": "No phone service", "gender": "Male"}, "No"),
    ({"tenure": "36", "PaymentMethod": "Credit card (automatic)", "TotalCharges": "1980.0"}, "No"),
    ({"tenure": "2", "Contract": "Month-to-month", "InternetService": "Fiber optic",
      "OnlineSecurity": "No", "TechSupport": "No", "PaymentMethod": "Electronic check",
      "PaperlessBilling": "Yes", "MonthlyCharges": "90.0", "TotalCharges": "180.0"}, "Yes"),
    ({"tenure": "1", "Contract": "Month-to-month", "InternetService": "Fiber optic",
      "OnlineSecurity": "No", "TechSupport": "No", "SeniorCitizen": "1",
      "MonthlyCharges": "85.0", "TotalCharges": "85.0", "Partner": "No"}, "Yes"),
    ({"tenure": "5", "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
      "StreamingTV": "Yes", "MonthlyCharges": "75.0", "TotalCharges": "375.0"}, "Yes"),
    ({"tenure": "8", "Contract": "Month-to-month", "InternetService": "Fiber optic",
      "MultipleLines": "Yes", "Dependents": "Yes", "MonthlyCharges": "99.0",
      "TotalCharges": "792.0"}, "Yes"),
]


def make_raw_frame() -> pd.DataFrame:
    rows = []
    for i, (overrides, churn) in enumerate(_VARIANTS):
        row = {"customerID": f"TEST-{i:04d}", **BASE_RECORD, **overrides, "Churn": churn}
        rows.append(row)
    return pd.DataFrame(rows).astype(str)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def valid_record() -> dict:
    """One prediction-time record with typed values, like the app sends."""
    rec = dict(BASE_RECORD)
    rec.update(tenure=12, MonthlyCharges=55.0, TotalCharges=660.0, SeniorCitizen="No")
    return rec


@pytest.fixture
def fixture_model(raw_df):
    """A ChurnModel trained in memory on the test fixture (no files written)."""
    from sklearn.linear_model import LogisticRegression

    from src.predict import ChurnModel
    from src.preprocessing import clean_raw, split_xy
    from src.train import make_pipeline

    X, y = split_xy(clean_raw(raw_df))
    pipe = make_pipeline(LogisticRegression(max_iter=1000), engineered=True).fit(X, y)
    means = dict(zip(pipe.named_steps["preprocess"].get_feature_names_out().tolist(),
                     pipe[:-1].transform(X).mean(axis=0).tolist()))
    return ChurnModel(pipe, {"model_name": "test", "threshold": 0.3, "high_risk_from": 0.5,
                             "train_feature_means": means})
