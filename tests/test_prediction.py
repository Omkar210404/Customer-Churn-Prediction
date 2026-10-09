import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src import evaluate as ev
from src.predict import ChurnModel, ModelNotFoundError, load_model, risk_band
from src.preprocessing import clean_raw, split_xy
from src.train import make_pipeline


@pytest.fixture
def tiny_model(raw_df, tmp_path):
    """A pipeline trained on the test fixture and saved like the real one."""
    X, y = split_xy(clean_raw(raw_df))
    pipe = make_pipeline(LogisticRegression(max_iter=1000), engineered=True).fit(X, y)
    model_path, meta_path = tmp_path / "m.joblib", tmp_path / "m.json"
    joblib.dump(pipe, model_path)
    means = dict(zip(pipe.named_steps["preprocess"].get_feature_names_out().tolist(),
                     pipe[:-1].transform(X).mean(axis=0).tolist()))
    meta_path.write_text(json.dumps({"model_name": "test", "threshold": 0.3,
                                     "high_risk_from": 0.5, "train_feature_means": means}))
    return load_model(model_path, meta_path)


# ------------------------------------------------------------------ risk bands

@pytest.mark.parametrize(
    "p,band", [(0.0, "LOW"), (0.29, "LOW"), (0.30, "MEDIUM"), (0.49, "MEDIUM"),
               (0.50, "HIGH"), (1.0, "HIGH")],
)
def test_risk_band_boundaries(p, band):
    assert risk_band(p, threshold=0.30, high_from=0.50) == band


# ---------------------------------------------------------- loading / saving

def test_load_model_missing_files_raises(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m src.train"):
        load_model(tmp_path / "nope.joblib", tmp_path / "nope.json")


def test_loaded_model_round_trip(tiny_model):
    assert isinstance(tiny_model, ChurnModel)
    assert tiny_model.threshold == 0.3
    assert tiny_model.high_from == 0.5


# ------------------------------------------------------------------ predict

def test_predict_single_record(tiny_model, valid_record):
    out = tiny_model.predict(valid_record)
    assert list(out.columns) == ["churn_probability", "risk_band", "flagged"]
    p = out["churn_probability"].iloc[0]
    assert 0.0 <= p <= 1.0
    assert out["risk_band"].iloc[0] == risk_band(p, 0.3, 0.5)
    assert out["flagged"].iloc[0] == (p >= 0.3)


def test_predict_batch(tiny_model, raw_df):
    out = tiny_model.predict(raw_df)  # raw strings, incl. customerID / Churn columns
    assert len(out) == len(raw_df)
    assert out["churn_probability"].between(0, 1).all()


def test_predict_with_missing_values(tiny_model, valid_record):
    valid_record.update(MonthlyCharges=None, InternetService="", PaymentMethod=np.nan)
    p = tiny_model.predict(valid_record)["churn_probability"].iloc[0]
    assert 0.0 <= p <= 1.0


def test_predict_invalid_input_raises(tiny_model, valid_record):
    valid_record["PaymentMethod"] = "Bitcoin"
    with pytest.raises(ValueError, match="PaymentMethod"):
        tiny_model.predict(valid_record)


def test_predict_is_deterministic(tiny_model, valid_record):
    a = tiny_model.predict(valid_record)["churn_probability"].iloc[0]
    b = tiny_model.predict(valid_record)["churn_probability"].iloc[0]
    assert a == b


def test_explain_contributions_add_up_to_log_odds(tiny_model, valid_record):
    contrib = tiny_model.explain(valid_record)
    assert "Contract" in contrib.index and "tenure" in contrib.index
    assert not any("_" in name and name.startswith("Contract") for name in contrib.index)
    lr = tiny_model.pipeline.named_steps["model"]
    names = tiny_model.pipeline.named_steps["preprocess"].get_feature_names_out()
    means = np.array([tiny_model.metadata["train_feature_means"][n] for n in names])
    reference_logit = lr.intercept_[0] + lr.coef_[0] @ means
    p = tiny_model.predict(valid_record)["churn_probability"].iloc[0]
    assert reference_logit + contrib.sum() == pytest.approx(np.log(p / (1 - p)))


# ------------------------------------------------------------------ metrics

def test_compute_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = ev.compute_metrics(y, p, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["roc_auc"] == 0.75


def test_band_stats_shares_sum_to_one():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.05, 0.2, 0.3, 0.6, 0.9])
    stats = ev.band_stats(y, p, medium_from=0.15, high_from=0.5)
    assert sum(s["share"] for s in stats.values()) == pytest.approx(1.0)
    assert stats["HIGH"]["churn_rate"] == 1.0


# ------------------------------------------------- real saved model (if trained)

real_model_missing = not (config.MODEL_PATH.exists() and config.METADATA_PATH.exists())


@pytest.mark.skipif(real_model_missing, reason="run `python -m src.train` first")
def test_real_model_loads_and_scores(valid_record):
    model = load_model()
    assert 0 < model.threshold < model.high_from <= 1
    out = model.predict(valid_record)
    assert out["churn_probability"].between(0, 1).all()


@pytest.mark.skipif(real_model_missing, reason="run `python -m src.train` first")
def test_real_model_ranks_obvious_cases(valid_record):
    """Sanity check: a new month-to-month fibre customer on electronic check
    should score higher than a 5-year two-year-contract customer."""
    risky = dict(valid_record, tenure=1, Contract="Month-to-month",
                 InternetService="Fiber optic", OnlineSecurity="No", TechSupport="No",
                 PaymentMethod="Electronic check", PaperlessBilling="Yes",
                 MonthlyCharges=90.0, TotalCharges=90.0)
    safe = dict(valid_record, tenure=60, Contract="Two year",
                PaymentMethod="Bank transfer (automatic)", TotalCharges=3300.0)
    out = load_model().predict([risky, safe])
    assert out["churn_probability"].iloc[0] > out["churn_probability"].iloc[1]
