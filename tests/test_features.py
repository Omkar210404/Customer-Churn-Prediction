import numpy as np
import pandas as pd
import pytest

from src.features import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, add_features
from src.preprocessing import validate_features


def test_add_features_does_not_mutate_input(valid_record):
    X = validate_features(valid_record)
    before = X.copy()
    out = add_features(X)
    pd.testing.assert_frame_equal(X, before)
    for col in ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL:
        assert col in out.columns


def test_num_addon_services_counts_yes(valid_record):
    # BASE_RECORD has OnlineSecurity=Yes and TechSupport=Yes
    assert add_features(validate_features(valid_record))["num_addon_services"].iloc[0] == 2


def test_avg_monthly_spend_handles_zero_tenure(valid_record):
    valid_record.update(tenure=0, TotalCharges=0.0)
    assert add_features(validate_features(valid_record))["avg_monthly_spend"].iloc[0] == 0.0
    valid_record.update(tenure=10, TotalCharges=500.0)
    assert add_features(validate_features(valid_record))["avg_monthly_spend"].iloc[0] == 50.0


@pytest.mark.parametrize(
    "tenure,bucket",
    [(0, "0-12m"), (12, "0-12m"), (13, "13-24m"), (24, "13-24m"), (48, "25-48m"), (72, "49m+")],
)
def test_tenure_bucket_boundaries(valid_record, tenure, bucket):
    valid_record["tenure"] = tenure
    assert add_features(validate_features(valid_record))["tenure_bucket"].iloc[0] == bucket


def test_missing_tenure_propagates_as_nan(valid_record):
    valid_record["tenure"] = None
    out = add_features(validate_features(valid_record))
    assert pd.isna(out["tenure_bucket"].iloc[0])
    assert np.isnan(out["avg_monthly_spend"].iloc[0])
