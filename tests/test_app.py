"""Tests for the Streamlit app: headless page runs (need the trained model) and UI helpers."""

import io
import re

import pandas as pd
import pytest

from src import config

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import ui  # noqa: E402

APP = str(config.ROOT / "app" / "streamlit_app.py")
PAGES = ["views/home.py", "views/predict.py", "views/batch.py", "views/insights.py",
         "views/drivers.py", "views/model.py", "views/actions.py"]

needs_model = pytest.mark.skipif(
    not (config.MODEL_PATH.exists() and config.METRICS_PATH.exists()),
    reason="run `python -m src.train` first",
)


def open_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.switch_page(page).run()
    return at


# ------------------------------------------------------------------ pages

@needs_model
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = open_page(page)
    assert not at.exception, at.exception


@needs_model
def test_predict_page_scores_immediately():
    at = open_page("views/predict.py")
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "big-number" in text and "%" in text
    assert "Suggested next step" in text


@needs_model
def test_presets_change_the_score():
    at = open_page("views/predict.py")

    def score(at) -> float:
        html = next(m.value for m in at.markdown if "big-number" in m.value)
        return float(re.search(r"([\d.]+)%", html).group(1))

    risky = next(b for b in at.button if b.label == "New fibre customer")
    risky.click().run()
    high = score(at)
    safe = next(b for b in at.button if b.label == "Loyal long-term customer")
    safe.click().run()
    low = score(at)
    assert not at.exception, at.exception
    assert high > low


@needs_model
def test_no_internet_clears_and_disables_addons():
    at = open_page("views/predict.py")
    at.pills(key="addons").set_value(["OnlineSecurity"]).run()
    at.segmented_control(key="internet").set_value("No").run()
    assert not at.exception, at.exception
    addons = at.pills(key="addons")
    assert addons.disabled
    assert addons.value == []


# ------------------------------------------------------------------ helpers

def test_pretty_feature():
    assert ui.pretty_feature("tenure") == "Tenure"
    assert ui.pretty_feature("Contract_Month-to-month") == "Contract: Month-to-month"
    assert ui.pretty_feature("tenure_bucket_49m+") == "Tenure group: 49m+"
    assert ui.pretty_feature("unknown_thing") == "unknown_thing"


def test_template_has_all_columns():
    header = pd.read_csv(io.StringIO(ui.template_csv())).columns.tolist()
    assert header == [config.ID_COL] + config.FEATURE_COLS


def test_score_table_keeps_ids_and_sorts(fixture_model, raw_df):
    out = ui.score_table(fixture_model, raw_df)
    assert list(out.columns) == [config.ID_COL, "churn_probability", "risk_band",
                                 "flagged_for_retention"]
    assert set(out[config.ID_COL]) == set(raw_df[config.ID_COL])
    assert out["churn_probability"].is_monotonic_decreasing


def test_score_table_rejects_bad_rows(fixture_model, raw_df):
    raw_df.loc[0, "Contract"] = "Weekly"
    with pytest.raises(ValueError, match="Contract"):
        ui.score_table(fixture_model, raw_df)
