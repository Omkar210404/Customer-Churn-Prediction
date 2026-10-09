"""Streamlit app for the churn model.

Run from the project root:
    streamlit run app/streamlit_app.py

Loads the saved pipeline and the reports written by `python -m src.train`.
It never retrains. Each page lives in app/views/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import ui  # noqa: E402

st.set_page_config(page_title="Customer Churn Prediction", page_icon=":material/insights:",
                   layout="wide")
ui.inject_css()

VIEWS = Path(__file__).parent / "views"
pages = [
    st.Page(VIEWS / "home.py", title="Overview", icon=":material/home:", default=True),
    st.Page(VIEWS / "predict.py", title="Score a customer", icon=":material/person_search:"),
    st.Page(VIEWS / "batch.py", title="Score a list", icon=":material/upload_file:"),
    st.Page(VIEWS / "insights.py", title="Data insights", icon=":material/bar_chart:"),
    st.Page(VIEWS / "drivers.py", title="Churn drivers", icon=":material/troubleshoot:"),
    st.Page(VIEWS / "model.py", title="Model performance", icon=":material/model_training:"),
    st.Page(VIEWS / "actions.py", title="Recommendations", icon=":material/lightbulb:"),
]
st.navigation(pages, position="top").run()
