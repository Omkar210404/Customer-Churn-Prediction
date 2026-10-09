"""Shared helpers for the Streamlit pages: data loading, labels, styling and charts."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.predict import ChurnModel, ModelNotFoundError, load_model

# Colours: dark-mode steps of the reference data-viz palette, validated against the app
# card surface #161922 (validate_palette.js: all checks pass). Status colours are only used
# for risk bands, and always next to an icon + label.
BLUE = "#3987e5"
RED = "#e66767"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]  # categorical, fixed order
INK = "#f2f2f0"
INK_2 = "#c3c2b7"
MUTED = "#9a9890"
GRID = "#2a2d33"
BASELINE = "#4a4d55"
SURFACE = "#161922"
STATUS = {"LOW": "#0ca30c", "MEDIUM": "#fab219", "HIGH": "#d03b3b"}
FONT = 14  # chart label size

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

# UI accent colours (chrome only: banners, card edges, KPI tiles). Data colours stay in the
# validated chart palette above.
ACCENTS = {
    "violet": ("#8b7cf6", "139,124,246"),
    "teal": ("#22c3a6", "34,195,166"),
    "blue": ("#3d8bfd", "61,139,253"),
    "amber": ("#f5b041", "245,176,65"),
    "pink": ("#f06292", "240,98,146"),
}
ACCENT_CYCLE = ["violet", "teal", "blue", "amber"]
BAND_TINT = {"LOW": "12,163,12", "MEDIUM": "250,178,25", "HIGH": "208,59,59"}


def _card_css() -> str:
    rules = []
    for name, (hex_, rgb) in ACCENTS.items():
        rules.append(
            f'[class*="st-key-card-{name}"] {{'
            f"border-left: 4px solid {hex_} !important;"
            f"background: linear-gradient(135deg, rgba({rgb},0.16) 0%, rgba({rgb},0.03) 45%, "
            f"#151821 100%) !important;}}"
        )
    for band, rgb in BAND_TINT.items():
        rules.append(
            f'[class*="st-key-card-risk-{band}"] {{'
            f"border: 1px solid rgba({rgb},0.55) !important;"
            f"background: linear-gradient(160deg, rgba({rgb},0.22) 0%, rgba({rgb},0.04) 55%, "
            f"#151821 100%) !important;"
            f"box-shadow: 0 0 0 1px rgba({rgb},0.15), 0 12px 40px rgba({rgb},0.18) !important;}}"
        )
    # KPI tiles: accent rotates by column position.
    for i, name in enumerate(ACCENT_CYCLE, start=1):
        hex_, rgb = ACCENTS[name]
        rules.append(
            f'[data-testid="stColumn"]:nth-child({i}) [data-testid="stMetric"] {{'
            f"border-top: 3px solid {hex_} !important;"
            f"background: linear-gradient(160deg, rgba({rgb},0.20) 0%, rgba({rgb},0.04) 60%, "
            f"#151821 100%) !important;}}"
        )
    # Batch-scoring KPI row: band counts wear their band colour, not the rotating accent.
    for i, band in enumerate(["HIGH", "MEDIUM", "LOW"], start=2):
        rgb = BAND_TINT[band]
        rules.append(
            f'[class*="st-key-bandkpis"] [data-testid="stColumn"]:nth-child({i}) '
            f'[data-testid="stMetric"] {{border-top: 3px solid {STATUS[band]} !important;'
            f"background: linear-gradient(160deg, rgba({rgb},0.22) 0%, rgba({rgb},0.04) 60%, "
            f"#151821 100%) !important;}}"
        )
    return "\n".join(rules)


CSS = """
<style>
[data-testid="stAppDeployButton"] {display: none;}

/* page background: soft colour glows on deep navy */
[data-testid="stApp"] {
  background:
    radial-gradient(1100px 520px at -5% -10%, rgba(139,124,246,0.20), transparent 60%),
    radial-gradient(900px 480px at 105% -5%, rgba(34,195,166,0.14), transparent 60%),
    radial-gradient(800px 600px at 50% 120%, rgba(61,139,253,0.10), transparent 60%),
    #0d0f14;
  background-attachment: fixed;
}

/* use the full window width */
[data-testid="stMainBlockContainer"] {
  max-width: 100% !important;
  padding: 4.6rem 3.2rem 2.5rem 3.2rem !important;
}

/* frosted top navigation with a gradient pill for the active page */
[data-testid="stHeader"] {
  background: rgba(13,15,20,0.72) !important;
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stTopNavLink"] {border-radius: 999px !important; padding: 4px 14px !important;}
[data-testid="stTopNavLink"][aria-current="page"] {
  background: linear-gradient(90deg, #6d5ff0, #3d8bfd) !important;
  color: #ffffff !important;
}

/* every bordered container becomes a card */
[data-testid="stVerticalBlock"][class*="st-key-card"] {
  background: linear-gradient(180deg, #171a24 0%, #13151c 100%);
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 18px !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  padding: 1.1rem 1.3rem !important;
}

/* KPI tiles */
[data-testid="stMetric"] {
  border-radius: 16px !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  padding: 16px 20px !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.30);
}
[data-testid="stMetricValue"] {font-weight: 700;}

/* banners */
.hero, .banner {
  position: relative; overflow: hidden; color: #ffffff;
  border: 1px solid rgba(255,255,255,0.10); border-radius: 22px;
  box-shadow: 0 18px 50px rgba(0,0,0,0.40);
}
.hero {
  padding: 38px 42px; margin-bottom: 6px;
  background:
    radial-gradient(600px 300px at 90% 10%, rgba(34,195,166,0.45), transparent 60%),
    linear-gradient(120deg, #2a1f7a 0%, #4c3fd1 45%, #1c6fd1 100%);
}
.hero h1 {color: #ffffff; font-size: 2.9rem; margin: 0 0 8px 0; padding: 0; letter-spacing: -0.5px;}
.hero p {color: #e6e3ff; font-size: 1.2rem; margin: 0; max-width: 900px;}
.hero .chips {margin-top: 18px;}
.chip {
  display: inline-block; background: rgba(255,255,255,0.14); color: #ffffff;
  border: 1px solid rgba(255,255,255,0.28); border-radius: 999px;
  padding: 4px 14px; margin: 0 8px 8px 0; font-size: 0.9rem; font-weight: 600;
}
.banner {padding: 24px 30px; margin-bottom: 4px; display: flex; gap: 20px; align-items: center;}
.banner .icon {
  font-family: 'Material Symbols Rounded'; font-size: 34px; line-height: 1;
  width: 58px; height: 58px; display: flex; align-items: center; justify-content: center;
  border-radius: 16px; background: rgba(255,255,255,0.16); flex: none;
}
.banner h2 {color: #ffffff; font-size: 2rem; margin: 0; padding: 0;}
.banner p {color: rgba(255,255,255,0.86); font-size: 1.08rem; margin: 4px 0 0 0;}

/* section titles with an accent bar */
.section {display: flex; align-items: center; gap: 12px; margin: 18px 0 6px 0;}
.section .bar {width: 6px; height: 28px; border-radius: 4px;
  background: linear-gradient(180deg, #8b7cf6, #22c3a6);}
.section h3 {margin: 0; padding: 0; font-size: 1.55rem;}

.subtle {color: #c3c2b7; font-size: 1.05rem;}
.big-number {
  font-size: 3.6rem; font-weight: 800; line-height: 1.05; margin: 4px 0 6px 0;
  background: linear-gradient(90deg, #ffffff 0%, #d9d3ff 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.stat-value {font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin: 6px 0 4px 0;}
</style>
"""

BANNER_GRADIENTS = {
    "violet": "linear-gradient(120deg, #2a1f7a 0%, #5b4be0 100%)",
    "teal": "linear-gradient(120deg, #0b4a45 0%, #14a08a 100%)",
    "blue": "linear-gradient(120deg, #0d2e66 0%, #2f7de8 100%)",
    "amber": "linear-gradient(120deg, #5a3a06 0%, #c98500 100%)",
    "pink": "linear-gradient(120deg, #5a1238 0%, #d0487f 100%)",
}


def inject_css() -> None:
    st.html(CSS.replace("</style>", _card_css() + "\n</style>"))


def card(name: str, accent: str | None = None, parent=None, **kwargs):
    """Bordered container styled as a card. `accent` adds a coloured edge and tint;
    `parent` is a column (or other container) to place it in."""
    key = f"card-{accent}-{name}" if accent else f"card-{name}"
    return (parent or st).container(border=True, key=key, **kwargs)


def hero(title: str, subtitle: str, chips: list[str]) -> None:
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.html(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>'
            f'<div class="chips">{chip_html}</div></div>')


def page_header(title: str, subtitle: str, icon: str = "insights", accent: str = "violet") -> None:
    """Gradient banner at the top of a page. `icon` is a Material Symbols name."""
    st.html(
        f'<div class="banner" style="background: {BANNER_GRADIENTS[accent]};">'
        f'<div class="icon">{icon}</div><div><h2>{title}</h2><p>{subtitle}</p></div></div>'
    )


def section(title: str) -> None:
    st.html(f'<div class="section"><div class="bar"></div><h3>{title}</h3></div>')


def footer(model: ChurnModel) -> None:
    st.divider()
    st.caption(
        f"Data: IBM Telco Customer Churn sample (a fictional telco, 7,043 customers). "
        f"Model: {model.metadata['model_name']}, trained {model.metadata['trained_at'][:10]}. "
        "Patterns shown are associations, not proven causes."
    )


# ------------------------------------------------------------------ charts

def styled(chart):
    """Shared chart styling: larger labels, no frame, transparent background."""
    return (chart.configure_axis(labelFontSize=FONT, titleFontSize=FONT, titleFontWeight="normal",
                                 domainColor=BASELINE, tickColor=BASELINE)
            .configure_legend(labelFontSize=FONT, titleFontSize=FONT, symbolSize=120)
            .configure_view(stroke=None)
            .properties(background="transparent"))


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
    marker = alt.Chart(point).mark_point(shape="triangle-down", size=520, filled=True,
                                         color=INK, yOffset=-18).encode(
        x="p:Q", tooltip=[alt.Tooltip("label:N", title="This customer")])
    return styled((track + marker).properties(height=70))


def rate_bars(df: pd.DataFrame, overall: float, sort: bool = True) -> alt.LayerChart:
    """Churn rate per group: one blue series, value labels, dashed overall-average rule."""
    order = list(df.sort_values("churn_rate", ascending=False)["group"]) if sort else list(df["group"])
    y = alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelColor=INK_2))
    x = alt.X("churn_rate:Q", title=None, scale=alt.Scale(domain=[0, max(0.6, df.churn_rate.max() + 0.08)]),
              axis=alt.Axis(format=".0%", grid=True, gridColor=GRID, domain=False,
                            labelColor=MUTED, tickCount=4))
    base = alt.Chart(df).encode(y=y, x=x)
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, size=22).encode(
        tooltip=[alt.Tooltip("group:N", title="Group"),
                 alt.Tooltip("churn_rate:Q", format=".1%", title="Churn rate"),
                 alt.Tooltip("customers:Q", format=",", title="Customers")])
    labels = base.mark_text(align="left", dx=6, color=INK_2, fontSize=FONT).encode(
        text=alt.Text("churn_rate:Q", format=".1%"))
    rule_df = pd.DataFrame({"x": [overall], "label": [f"average {pct(overall)}"]})
    rule = alt.Chart(rule_df).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(x="x:Q")
    rule_label = alt.Chart(rule_df).mark_text(align="left", dx=4, dy=-6, color=MUTED, fontSize=FONT - 1,
                                              baseline="bottom").encode(
        x="x:Q", y=alt.value(0), text="label:N")
    return styled((bars + labels + rule + rule_label).properties(height=alt.Step(38)))


def signed_bars(values: pd.Series, title: str, value_format: str = ".2f") -> alt.LayerChart:
    """Diverging bars: red = towards churn, blue = away from churn."""
    df = values.rename_axis("feature").reset_index(name="value")
    df["direction"] = df["value"].map(lambda v: "raises churn risk" if v > 0 else "lowers churn risk")
    y = alt.Y("feature:N", sort=list(df["feature"]), title=None,
              axis=alt.Axis(labelLimit=260, labelColor=INK_2))
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, size=20).encode(
        y=y,
        x=alt.X("value:Q", title=title, axis=alt.Axis(grid=True, gridColor=GRID, tickCount=6,
                                                       labelColor=MUTED, titleColor=MUTED)),
        color=alt.Color("direction:N",
                        scale=alt.Scale(domain=["raises churn risk", "lowers churn risk"],
                                        range=[RED, BLUE]),
                        legend=alt.Legend(orient="top", title=None, labelColor=INK_2, labelLimit=0)),
        tooltip=[alt.Tooltip("feature:N", title="Input"),
                 alt.Tooltip("value:Q", format=value_format, title="Effect"),
                 alt.Tooltip("direction:N", title="Direction")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=BASELINE).encode(x="x:Q")
    return styled((bars + zero).properties(height=alt.Step(34)))


def importance_bars(df: pd.DataFrame) -> alt.Chart:
    """Permutation importance: one blue series with value labels."""
    y = alt.Y("feature:N", sort=list(df["feature"]), title=None,
              axis=alt.Axis(labelLimit=220, labelColor=INK_2))
    x = alt.X("mean:Q", title="Drop in ROC-AUC when this input is shuffled",
              axis=alt.Axis(grid=True, gridColor=GRID, labelColor=MUTED, titleColor=MUTED,
                            tickCount=5))
    base = alt.Chart(df).encode(y=y, x=x)
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, size=20).encode(
        tooltip=[alt.Tooltip("feature:N", title="Input"),
                 alt.Tooltip("mean:Q", format=".4f", title="Mean drop in AUC"),
                 alt.Tooltip("std:Q", format=".4f", title="Std over 10 shuffles")])
    labels = base.mark_text(align="left", dx=6, color=INK_2, fontSize=FONT).encode(
        text=alt.Text("mean:Q", format=".3f"))
    return styled((bars + labels).properties(height=alt.Step(34)))


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


# ------------------------------------------------------------------ model charts

def _axis(fmt: str) -> alt.Axis:
    return alt.Axis(grid=True, gridColor=GRID, labelColor=MUTED, titleColor=MUTED, format=fmt,
                    tickCount=5)


def roc_chart(roc: dict, aucs: dict) -> alt.LayerChart:
    rows = []
    for name, pts in roc.items():
        label = f"{name} (AUC {aucs[name]:.3f})"
        rows += [{"model": label, "fpr": f, "tpr": t} for f, t in zip(pts["fpr"], pts["tpr"])]
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["model"]))
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("fpr:Q", title="False positive rate", axis=_axis(".0%")),
        y=alt.Y("tpr:Q", title="True positive rate (recall)", axis=_axis(".0%")),
        color=alt.Color("model:N", scale=alt.Scale(domain=order, range=SERIES[:len(order)]),
                        legend=alt.Legend(orient="bottom", direction="vertical", title=None,
                                          labelColor=INK_2, labelLimit=0)),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("fpr:Q", format=".1%", title="False positive rate"),
                 alt.Tooltip("tpr:Q", format=".1%", title="Recall")],
    )
    diag = alt.Chart(pd.DataFrame({"a": [0, 1]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(
        x="a:Q", y="a:Q")
    return styled((diag + lines).properties(height=380))


def threshold_chart(curve: dict, chosen: float) -> alt.LayerChart:
    df = (pd.DataFrame(curve).rename(columns={"f1": "F1", "f2": "F2"})
          .melt("threshold", var_name="metric", value_name="score"))
    order = ["precision", "recall", "F1", "F2"]
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["threshold"], empty=False)
    base = alt.Chart(df).encode(
        x=alt.X("threshold:Q", title="Decision threshold", axis=_axis(".1f")),
        y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1]), axis=_axis(".1f")),
        color=alt.Color("metric:N", scale=alt.Scale(domain=order, range=SERIES),
                        legend=alt.Legend(orient="top", title=None, labelColor=INK_2)),
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=70, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("threshold:Q", format=".2f", title="Threshold"),
                 alt.Tooltip("metric:N", title="Metric"),
                 alt.Tooltip("score:Q", format=".3f", title="Score")],
    ).add_params(hover)
    rule_df = pd.DataFrame({"t": [chosen], "label": [f"chosen {chosen:.2f}"]})
    rule = alt.Chart(rule_df).mark_rule(color=INK, strokeDash=[5, 4]).encode(x="t:Q")
    label = alt.Chart(rule_df).mark_text(align="left", dx=6, dy=10, color=INK,
                                         fontSize=FONT).encode(x="t:Q", y=alt.value(0), text="label:N")
    return styled((lines + points + rule + label).properties(height=380))


def confusion_chart(cm: dict) -> alt.LayerChart:
    cells = [("Actually stayed", "Predicted stay", cm["tn"], "correctly not flagged"),
             ("Actually stayed", "Predicted churn", cm["fp"], "false alarm"),
             ("Actually churned", "Predicted stay", cm["fn"], "missed churner"),
             ("Actually churned", "Predicted churn", cm["tp"], "churner caught")]
    df = pd.DataFrame(cells, columns=["actual", "predicted", "count", "meaning"])
    enc = dict(
        x=alt.X("predicted:N", title=None, sort=["Predicted stay", "Predicted churn"],
                axis=alt.Axis(labelColor=INK_2, orient="top", labelAngle=0, domain=False,
                              ticks=False)),
        y=alt.Y("actual:N", title=None, sort=["Actually stayed", "Actually churned"],
                axis=alt.Axis(labelColor=INK_2, titleColor=MUTED, domain=False, ticks=False)),
    )
    rects = alt.Chart(df).mark_rect(cornerRadius=8, stroke=SURFACE, strokeWidth=6).encode(
        **enc,
        color=alt.Color("count:Q", scale=alt.Scale(range=["#1a2a40", "#256abf"]), legend=None),
        tooltip=[alt.Tooltip("actual:N", title="Actual"),
                 alt.Tooltip("predicted:N", title="Predicted"),
                 alt.Tooltip("count:Q", title="Customers"),
                 alt.Tooltip("meaning:N", title="Meaning")],
    )
    counts = alt.Chart(df).mark_text(fontSize=28, fontWeight="bold", color=INK, dy=-12).encode(
        **enc, text="count:Q")
    meaning = alt.Chart(df).mark_text(fontSize=FONT, color=INK_2, dy=18).encode(
        **enc, text="meaning:N")
    return styled((rects + counts + meaning).properties(height=320))


def calibration_chart(cal: dict) -> alt.LayerChart:
    df = pd.DataFrame({"predicted": cal["mean_predicted"], "observed": cal["observed_rate"]})
    diag = alt.Chart(pd.DataFrame({"a": [0, 1]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(
        x=alt.X("a:Q", title="Predicted churn probability", axis=_axis(".0%")),
        y=alt.Y("a:Q", title="Observed churn rate", axis=_axis(".0%")))
    line = alt.Chart(df).mark_line(
        color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=80, filled=True, color=BLUE)
    ).encode(
        x="predicted:Q", y="observed:Q",
        tooltip=[alt.Tooltip("predicted:Q", format=".1%", title="Mean predicted"),
                 alt.Tooltip("observed:Q", format=".1%", title="Actually churned")])
    return styled((diag + line).properties(height=380))
