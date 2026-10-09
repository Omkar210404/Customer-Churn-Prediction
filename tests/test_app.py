"""Headless smoke tests for the Streamlit app (needs the trained model)."""

import pytest

from src import config

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(config.ROOT / "app" / "streamlit_app.py")
SECTIONS = ["Overview", "Predict a customer", "Model information", "Dataset insights",
            "Feature importance", "Business recommendations"]

pytestmark = pytest.mark.skipif(
    not (config.MODEL_PATH.exists() and config.METRICS_PATH.exists()),
    reason="run `python -m src.train` first",
)


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_renders(section):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(section).run()
    assert not at.exception, at.exception


def test_prediction_flow_shows_probability_and_band():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Predict a customer").run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert at.metric[0].label == "Churn probability"
    assert at.metric[0].value.endswith("%")
    banners = [e.value for e in (*at.success, *at.warning, *at.error)]
    assert any("risk" in b for b in banners)


def test_no_internet_locks_addon_fields():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Predict a customer").run()
    internet = next(s for s in at.selectbox if s.label == "Internet service")
    internet.set_value("No").run()
    security = next(s for s in at.selectbox if s.label == "OnlineSecurity")
    assert security.value == "No internet service"
    assert security.disabled
    at.button[0].click().run()
    assert not at.exception, at.exception
