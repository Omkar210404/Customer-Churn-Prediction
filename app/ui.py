"""Shared helpers for the Streamlit pages: data loading, labels, styling and charts."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.predict import ChurnModel, ModelNotFoundError, load_model

# Colours (reference data-viz palette). Status colours are only used for risk bands,
# and always next to an icon + label.
BLUE = "#2a78d6"
RED = "#e34948"
INK_2 = "#52514e"
MUTED = "#898781"
STATUS = {"LOW": "#0ca30c", "MEDIUM": "#fab219", "HIGH": "#d03b3b"}

BANDS = {
    "LOW": {
        "badge": "green", "icon": ":material/check_circle:",
        "headline": "Low churn risk",
        "action": "No action needed. Keep up regular service and check again next cycle.",
    },
    "MEDIUM": {
        "badge": "orange", "icon": ":material/warning:",
        "headline": "Medium churn risk",
        "action": "Worth a light-touch retention step: a check-in call, a loyalty perk, "
                  "or an offer to move to a longer contract.",
    },
    "HIGH": {
        "badge": "red", "icon": ":material/error:",
        "headline": "High churn risk",
        "action": "Prioritise this customer: personal outreach and a concrete retention offer "
                  "(discount for an annual contract, service review, support add-on trial).",
    },
}

LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "tenure": "Tenure",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "num_addon_services": "Number of add-ons",
    "avg_monthly_spend": "Average monthly spend",
    "tenure_bucket": "Tenure group",
}


def pretty_feature(name: str) -> str:
    """'Contract_Month-to-month' -> 'Contract: Month-to-month'; 'tenure' -> 'Tenure'."""
    if name in LABELS:
        return LABELS[name]
    for col in sorted(LABELS, key=len, reverse=True):
        if name.startswith(col + "_"):
            return f"{LABELS[col]}: {name[len(col) + 1:]}"
    return name


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


# ------------------------------------------------------------------ data

@st.cache_resource
def get_model() -> ChurnModel:
    return load_model()


@st.cache_data
def get_metrics() -> dict | None:
    if not config.METRICS_PATH.exists():
        return None
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


def require_model() -> ChurnModel:
    try:
        return get_model()
    except ModelNotFoundError as exc:
        st.error(f"{exc}", icon=":material/error:")
        st.stop()


def require_metrics() -> dict:
    metrics = get_metrics()
    if metrics is None:
        st.info("Run `python -m src.train` to generate the reports this page needs.",
                icon=":material/info:")
        st.stop()
    return metrics


def group_rates(metrics: dict, col: str) -> pd.DataFrame:
    data = metrics["dataset_summary"]["churn_rate_by"][col]
    return pd.DataFrame(
        [{"group": k, "churn_rate": v["churn_rate"], "customers": v["customers"]}
         for k, v in data.items()]
    )


# ------------------------------------------------------------------ layout

CSS = """
<style>
[data-testid="stAppDeployButton"] {display: none;}
[data-testid="stMainBlockContainer"] {padding-top: 4.5rem; max-width: 1200px;}
.hero {
  background: linear-gradient(120deg, #104281 0%, #2a78d6 100%);
  border-radius: 14px; padding: 28px 32px; margin-bottom: 8px; color: #ffffff;
}
.hero h1 {color: #ffffff; font-size: 2.1rem; margin: 0 0 6px 0; padding: 0;}
.hero p {color: #dbe8fb; font-size: 1.05rem; margin: 0; max-width: 820px;}
.hero .chips {margin-top: 14px;}
.hero .chip {
  display: inline-block; background: rgba(255,255,255,0.14); color: #ffffff;
  border: 1px solid rgba(255,255,255,0.25); border-radius: 999px;
  padding: 3px 12px; margin: 0 6px 6px 0; font-size: 0.85rem;
}
.page-title h2 {margin-bottom: 0;}
.subtle {color: #52514e;}
.big-number {font-size: 3.2rem; font-weight: 700; line-height: 1.05; margin: 4px 0 2px 0;}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def hero(title: str, subtitle: str, chips: list[str]) -> None:
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.html(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>'
            f'<div class="chips">{chip_html}</div></div>')


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.markdown(f'<p class="subtle">{subtitle}</p>', unsafe_allow_html=True)


def footer(model: ChurnModel) -> None:
    st.divider()
    st.caption(
        f"Data: IBM Telco Customer Churn sample (a fictional telco, 7,043 customers). "
        f"Model: {model.metadata['model_name']}, trained {model.metadata['trained_at'][:10]}. "
        "Patterns shown are associations, not proven causes."
    )


# ------------------------------------------------------------------ charts

def risk_meter(p: float, threshold: float, high_from: float) -> alt.LayerChart:
    """Track split into the three bands with a marker at the customer's probability."""
    zones = pd.DataFrame([
        {"band": "LOW", "start": 0.0, "end": threshold},
        {"band": "MEDIUM", "start": threshold, "end": high_from},
        {"band": "HIGH", "start": high_from, "end": 1.0},
    ])
    x_scale = alt.Scale(domain=[0, 1], nice=False)
    colour = alt.Color("band:N", scale=alt.Scale(domain=list(STATUS), range=list(STATUS.values())),
                       legend=None)
    track = alt.Chart(zones).mark_bar(height=16, opacity=0.85).encode(
        x=alt.X("start:Q", scale=x_scale,
                axis=alt.Axis(values=[0, threshold, high_from, 1], format=".0%", title=None,
                              grid=False, domain=False, ticks=False, labelColor=MUTED)),
        x2="end:Q",
        color=colour,
        tooltip=[alt.Tooltip("band:N", title="Band"),
                 alt.Tooltip("start:Q", format=".0%", title="From"),
                 alt.Tooltip("end:Q", format=".0%", title="To")],
    )
    point = pd.DataFrame({"p": [p], "label": [pct(p)]})
    marker = alt.Chart(point).mark_point(shape="triangle-down", size=260, filled=True,
                                         color="#0b0b0b", yOffset=-16).encode(x="p:Q")
    text = alt.Chart(point).mark_text(dy=-34, fontSize=13, fontWeight="bold",
                                      color="#0b0b0b").encode(x="p:Q", text="label:N")
    return (track + marker + text).properties(height=86)


def rate_bars(df: pd.DataFrame, overall: float, sort: bool = True) -> alt.LayerChart:
    """Churn rate per group: one blue series, value labels, dashed overall-average rule."""
    order = list(df.sort_values("churn_rate", ascending=False)["group"]) if sort else list(df["group"])
    y = alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelColor=INK_2))
    x = alt.X("churn_rate:Q", title=None, scale=alt.Scale(domain=[0, max(0.6, df.churn_rate.max() + 0.08)]),
              axis=alt.Axis(format=".0%", grid=True, gridColor="#eeeeec", domain=False,
                            labelColor=MUTED, tickCount=4))
    base = alt.Chart(df).encode(y=y, x=x)
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, size=22).encode(
        tooltip=[alt.Tooltip("group:N", title="Group"),
                 alt.Tooltip("churn_rate:Q", format=".1%", title="Churn rate"),
                 alt.Tooltip("customers:Q", format=",", title="Customers")])
    labels = base.mark_text(align="left", dx=6, color=INK_2, fontSize=12).encode(
        text=alt.Text("churn_rate:Q", format=".1%"))
    rule_df = pd.DataFrame({"x": [overall], "label": [f"average {pct(overall)}"]})
    rule = alt.Chart(rule_df).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(x="x:Q")
    rule_label = alt.Chart(rule_df).mark_text(align="left", dx=4, dy=-6, color=MUTED, fontSize=11,
                                              baseline="bottom").encode(
        x="x:Q", y=alt.value(0), text="label:N")
    return (bars + labels + rule + rule_label).properties(height=alt.Step(34))


def signed_bars(values: pd.Series, title: str, value_format: str = ".2f") -> alt.LayerChart:
    """Diverging bars: red = towards churn, blue = away from churn."""
    df = values.rename_axis("feature").reset_index(name="value")
    df["direction"] = df["value"].map(lambda v: "raises churn risk" if v > 0 else "lowers churn risk")
    y = alt.Y("feature:N", sort=list(df["feature"]), title=None,
              axis=alt.Axis(labelLimit=260, labelColor=INK_2))
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, size=20).encode(
        y=y,
        x=alt.X("value:Q", title=title, axis=alt.Axis(grid=True, gridColor="#eeeeec", tickCount=6,
                                                       labelColor=MUTED, titleColor=MUTED)),
        color=alt.Color("direction:N",
                        scale=alt.Scale(domain=["raises churn risk", "lowers churn risk"],
                                        range=[RED, BLUE]),
                        legend=alt.Legend(orient="top", title=None, labelColor=INK_2)),
        tooltip=[alt.Tooltip("feature:N", title="Input"),
                 alt.Tooltip("value:Q", format=value_format, title="Effect"),
                 alt.Tooltip("direction:N", title="Direction")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#c3c2b7").encode(x="x:Q")
    return (bars + zero).properties(height=alt.Step(30))


def importance_bars(df: pd.DataFrame) -> alt.Chart:
    """Permutation importance: one blue series with value labels."""
    y = alt.Y("feature:N", sort=list(df["feature"]), title=None,
              axis=alt.Axis(labelLimit=220, labelColor=INK_2))
    x = alt.X("mean:Q", title="Drop in ROC-AUC when this input is shuffled",
              axis=alt.Axis(grid=True, gridColor="#eeeeec", labelColor=MUTED, titleColor=MUTED,
                            tickCount=5))
    base = alt.Chart(df).encode(y=y, x=x)
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, size=20).encode(
        tooltip=[alt.Tooltip("feature:N", title="Input"),
                 alt.Tooltip("mean:Q", format=".4f", title="Mean drop in AUC"),
                 alt.Tooltip("std:Q", format=".4f", title="Std over 10 shuffles")])
    labels = base.mark_text(align="left", dx=6, color=INK_2, fontSize=12).encode(
        text=alt.Text("mean:Q", format=".3f"))
    return (bars + labels).properties(height=alt.Step(30))


# ------------------------------------------------------------------ batch scoring

def score_table(model: ChurnModel, df: pd.DataFrame) -> pd.DataFrame:
    """Score an uploaded table. Keeps customerID if present; sorted by probability."""
    scored = model.predict(df)
    out = pd.DataFrame(index=df.index)
    if config.ID_COL in df.columns:
        out[config.ID_COL] = df[config.ID_COL]
    out["churn_probability"] = scored["churn_probability"].round(4)
    out["risk_band"] = scored["risk_band"]
    out["flagged_for_retention"] = scored["flagged"]
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def template_csv() -> str:
    """Header-only CSV with the expected columns."""
    return ",".join([config.ID_COL] + config.FEATURE_COLS) + "\n"
