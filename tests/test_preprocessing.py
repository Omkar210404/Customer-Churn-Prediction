import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import add_features, model_input_columns
from src.preprocessing import (
    CleaningReport,
    build_preprocessor,
    clean_raw,
    split_xy,
    validate_features,
)


# ------------------------------------------------------------------ clean_raw

def test_clean_raw_keeps_every_row(raw_df):
    out = clean_raw(raw_df)
    assert len(out) == len(raw_df)


def test_clean_raw_fixes_blank_total_charges(raw_df):
    report = CleaningReport()
    out = clean_raw(raw_df, report)
    zero_tenure = out["tenure"] == 0
    assert zero_tenure.sum() == 1
    assert out.loc[zero_tenure, "TotalCharges"].iloc[0] == 0.0
    assert any("TotalCharges blank in 1 rows" in s for s in report.steps)


def test_clean_raw_types_and_mappings(raw_df):
    out = clean_raw(raw_df)
    for col in config.NUMERIC_COLS:
        assert out[col].dtype == float
    assert set(out[config.TARGET].unique()) == {0, 1}
    assert set(out["SeniorCitizen"].unique()) <= {"No", "Yes"}
    assert config.ID_COL not in out.columns
    assert out.isna().sum().sum() == 0


def test_clean_raw_strips_whitespace(raw_df):
    raw_df.loc[0, "Contract"] = "  One year "
    assert clean_raw(raw_df).loc[0, "Contract"] == "One year"


def test_clean_raw_rejects_blank_total_charges_with_tenure(raw_df):
    raw_df.loc[0, "TotalCharges"] = ""  # tenure is 12 here
    with pytest.raises(ValueError, match="tenure > 0"):
        clean_raw(raw_df)


def test_clean_raw_rejects_missing_column(raw_df):
    with pytest.raises(ValueError, match="missing columns"):
        clean_raw(raw_df.drop(columns=["Contract"]))


def test_clean_raw_rejects_unknown_category(raw_df):
    raw_df.loc[0, "InternetService"] = "Satellite"
    with pytest.raises(ValueError, match="InternetService"):
        clean_raw(raw_df)


def test_split_xy(raw_df):
    X, y = split_xy(clean_raw(raw_df))
    assert list(X.columns) == config.FEATURE_COLS
    assert len(X) == len(y)


# ------------------------------------------------------------ validate_features

def test_validate_accepts_dict_and_list(valid_record):
    assert len(validate_features(valid_record)) == 1
    assert len(validate_features([valid_record, valid_record])) == 2


def test_validate_maps_senior_citizen_numbers(valid_record):
    valid_record["SeniorCitizen"] = 1
    assert validate_features(valid_record)["SeniorCitizen"].iloc[0] == "Yes"


def test_validate_drops_id_and_target(valid_record):
    valid_record.update(customerID="X-1", Churn="Yes")
    out = validate_features(valid_record)
    assert list(out.columns) == config.FEATURE_COLS


@pytest.mark.parametrize("missing", [None, np.nan, "", "  "])
def test_validate_missing_values_become_nan(valid_record, missing):
    valid_record["MonthlyCharges"] = missing
    valid_record["PaymentMethod"] = missing
    out = validate_features(valid_record)
    assert np.isnan(out["MonthlyCharges"].iloc[0])
    assert pd.isna(out["PaymentMethod"].iloc[0])


def test_validate_fills_absent_columns_with_nan():
    out = validate_features({"tenure": 3, "Contract": "Month-to-month"})
    assert list(out.columns) == config.FEATURE_COLS
    assert pd.isna(out["gender"].iloc[0])


def test_validate_rejects_invalid_category(valid_record):
    valid_record["Contract"] = "Weekly"
    with pytest.raises(ValueError, match="Contract"):
        validate_features(valid_record)


def test_validate_rejects_non_numeric(valid_record):
    valid_record["tenure"] = "twelve"
    with pytest.raises(ValueError, match="tenure must be numeric"):
        validate_features(valid_record)


@pytest.mark.parametrize("col", ["tenure", "MonthlyCharges", "TotalCharges"])
def test_validate_rejects_negative(valid_record, col):
    valid_record[col] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_features(valid_record)


def test_validate_rejects_unrealistic_tenure(valid_record):
    valid_record["tenure"] = 500
    with pytest.raises(ValueError, match="tenure"):
        validate_features(valid_record)


def test_validate_rejects_unknown_column(valid_record):
    valid_record["favourite_colour"] = "blue"
    with pytest.raises(ValueError, match="Unknown columns"):
        validate_features(valid_record)


def test_validate_rejects_empty():
    with pytest.raises(ValueError):
        validate_features([])


# --------------------------------------------------------- build_preprocessor

def test_preprocessor_imputes_missing_values(raw_df, valid_record):
    X, _ = split_xy(clean_raw(raw_df))
    numeric, categorical = model_input_columns(engineered=True)
    pre = build_preprocessor(numeric, categorical).fit(add_features(X))
    valid_record["MonthlyCharges"] = None
    valid_record["Contract"] = None
    Z = pre.transform(add_features(validate_features(valid_record)))
    assert Z.shape[0] == 1
    assert not np.isnan(Z).any()


def test_preprocessor_scales_with_train_statistics_only(raw_df):
    X, _ = split_xy(clean_raw(raw_df))
    numeric, categorical = model_input_columns(engineered=False)
    pre = build_preprocessor(numeric, categorical).fit(X.iloc[:5])
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(X["tenure"].iloc[:5].mean())
