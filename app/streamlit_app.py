"""Streamlit app for the churn model.

Run from the project root:
    streamlit run app/streamlit_app.py

Loads the saved pipeline and the reports written by `python -m src.train`.
It never retrains.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import ModelNotFoundError, load_model  # noqa: E402

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

NO_INTERNET = "No internet service"
NO_PHONE = "No phone service"
BAND_STYLE = {"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_metrics() -> dict | None:
    if not config.METRICS_PATH.exists():
        return None
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def figure(name: str, caption: str | None = None) -> None:
    path = config.FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure {name} not found. Run `python -m src.train`.")


def rates(summary: dict, col: str) -> pd.Series:
    data = summary["churn_rate_by"][col]
    return pd.Series({k: v["churn_rate"] for k, v in data.items()}).sort_values(ascending=False)


def hbar(series: pd.Series, value_title: str, percent: bool = False, signed: bool = False) -> None:
    """Horizontal bar chart that keeps the series order (st.bar_chart sorts alphabetically)."""
    df = series.rename_axis("label").reset_index(name="value")
    color = (alt.condition("datum.value > 0", alt.value("#C44E52"), alt.value("#4C72B0"))
             if signed else alt.value("#C44E52"))
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y("label:N", sort=list(df["label"]), title=None),
        x=alt.X("value:Q", title=value_title, axis=alt.Axis(format=".0%" if percent else "")),
        color=color,
        tooltip=["label", alt.Tooltip("value:Q", format=".1%" if percent else ".3f")],
    )
    st.altair_chart(chart.properties(height=alt.Step(30)))


# ------------------------------------------------------------------ sections

def section_overview(model, metrics) -> None:
    st.title("Customer Churn Prediction & Analytics")
    st.write(
        "Estimates how likely a telecom customer is to cancel their service (churn), "
        "using a model trained on the public **IBM Telco Customer Churn** sample dataset "
        "(a fictional telco; 7,043 customers). Use the sidebar to score a customer, "
        "inspect the model, or read the data insights."
    )
    meta = model.metadata
    tuned = meta["test_metrics_tuned"]
    st.subheader(f"Final model: {meta['model_name']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test ROC-AUC", f"{tuned['roc_auc']:.3f}")
    c2.metric(f"Recall @ threshold {model.threshold:.2f}", f"{tuned['recall']:.3f}",
              help="Share of actual churners the model flags.")
    c3.metric(f"Precision @ threshold {model.threshold:.2f}", f"{tuned['precision']:.3f}",
              help="Share of flagged customers who actually churned.")
    if metrics:
        s = metrics["dataset_summary"]
        st.subheader("The data at a glance")
        d1, d2, d3 = st.columns(3)
        d1.metric("Customers", f"{s['rows']:,}")
        d2.metric("Churned", f"{s['churned']:,} ({pct(s['churn_rate'])})")
        d3.metric("Median tenure: churned vs stayed",
                  f"{s['median_tenure']['churned']:.0f} vs {s['median_tenure']['stayed']:.0f} months")
    st.caption(
        "All figures come from `python -m src.train` on the held-out test set "
        f"({meta['test_rows']:,} customers) unless stated otherwise."
    )


def customer_form() -> dict:
    cv = config.CATEGORICAL_VALUES
    rec: dict = {}
    st.markdown("**Account**")
    a1, a2, a3 = st.columns(3)
    rec["tenure"] = a1.number_input("Tenure (months)", 0, 120, 5, step=1)
    rec["Contract"] = a2.selectbox("Contract", cv["Contract"])
    rec["PaymentMethod"] = a3.selectbox("Payment method", cv["PaymentMethod"])
    b1, b2, b3 = st.columns(3)
    rec["MonthlyCharges"] = b1.number_input("Monthly charges ($)", 0.0, 500.0, 80.0, step=0.5)
    estimate = round(rec["tenure"] * rec["MonthlyCharges"], 2)
    rec["TotalCharges"] = b2.number_input(
        "Total charges ($)", 0.0, 100000.0, float(estimate), step=10.0,
        help="Defaults to tenure × monthly charges. Overwrite with the real billed total if known.",
    )
    rec["PaperlessBilling"] = b3.selectbox("Paperless billing", cv["PaperlessBilling"], index=1)

    st.markdown("**Customer**")
    c1, c2, c3, c4 = st.columns(4)
    rec["gender"] = c1.selectbox("Gender", cv["gender"])
    rec["SeniorCitizen"] = c2.selectbox("Senior citizen", cv["SeniorCitizen"])
    rec["Partner"] = c3.selectbox("Partner", cv["Partner"])
    rec["Dependents"] = c4.selectbox("Dependents", cv["Dependents"])

    st.markdown("**Services**")
    s1, s2, s3 = st.columns(3)
    rec["PhoneService"] = s1.selectbox("Phone service", cv["PhoneService"], index=1)
    if rec["PhoneService"] == "No":
        rec["MultipleLines"] = s2.selectbox("Multiple lines", [NO_PHONE], disabled=True)
    else:
        rec["MultipleLines"] = s2.selectbox("Multiple lines", ["No", "Yes"])
    rec["InternetService"] = s3.selectbox("Internet service", cv["InternetService"], index=1)

    addon_cols = st.columns(3)
    for i, col in enumerate(config.ADDON_SERVICES):
        target = addon_cols[i % 3]
        if rec["InternetService"] == "No":
            rec[col] = target.selectbox(col, [NO_INTERNET], disabled=True)
        else:
            rec[col] = target.selectbox(col, ["No", "Yes"])
    return rec


def section_predict(model) -> None:
    st.header("Score a customer")
    st.write("Fill in the customer's details. Fields that depend on each other "
             "(e.g. add-ons without internet) are kept consistent automatically.")
    rec = customer_form()
    if not st.button("Predict churn risk", type="primary"):
        return
    try:
        out = model.predict(rec).iloc[0]
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return

    p, band = float(out["churn_probability"]), out["risk_band"]
    st.divider()
    r1, r2 = st.columns([1, 2])
    r1.metric("Churn probability", pct(p))
    BAND_STYLE[band](f"**{band} risk**  ·  {model.metadata['risk_bands'][band]}")
    bands = model.metadata["oof_band_stats"]
    r2.markdown(
        f"""
**How the bands are defined** (cut-offs chosen on training data only):
- **LOW** – below the decision threshold **{model.threshold:.2f}**: not flagged.
  In cross-validation, {pct(bands['LOW']['churn_rate'])} of these customers churned.
- **MEDIUM** – {model.threshold:.2f} to {model.high_from:.2f}: flagged for retention.
  Observed churn {pct(bands['MEDIUM']['churn_rate'])}.
- **HIGH** – {model.high_from:.2f} or above: churn is more likely than not.
  Observed churn {pct(bands['HIGH']['churn_rate'])}.

The threshold {model.threshold:.2f} maximises the F2 score (recall weighted twice as much as
precision), because missing a churner usually costs more than an unneeded retention offer.
"""
    )

    if model.is_linear:
        st.subheader("What moved this score")
        hbar(model.explain(rec).head(8), "contribution to churn log-odds", signed=True)
        st.caption(
            "Contribution of each input to this customer's churn log-odds compared with the "
            "average training customer (positive = towards churn). Exact for the logistic "
            "regression model. Inputs are correlated: monthly charges are largely set by the "
            "services a customer has, so read MonthlyCharges together with InternetService and "
            "the add-ons. These are model associations, not causes."
        )


def section_model(model, metrics) -> None:
    st.header("Model information")
    meta = model.metadata
    st.write(
        f"**Final model:** {meta['model_name']} with parameters `{meta['best_params']}`, "
        f"trained on {meta['train_rows']:,} customers (stratified 80/20 split). "
        f"Engineered features used: {'yes' if meta['engineered_features'] else 'no'}."
    )
    if metrics:
        st.markdown(f"**Why this model:** {metrics['selection_reason']}")
        rows = []
        for name, m in metrics["test"].items():
            c = metrics["cv"][name]
            rows.append({"Model": name,
                         "CV ROC-AUC": f"{c['cv_roc_auc_mean']:.3f} ± {c['cv_roc_auc_std']:.3f}",
                         "Accuracy": m["accuracy"], "Precision": m["precision"],
                         "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"]})
        t = metrics["test_final_tuned"]
        rows.append({"Model": f"{meta['model_name']} @ {t['threshold']:.2f} (final)",
                     "CV ROC-AUC": "", "Accuracy": t["accuracy"], "Precision": t["precision"],
                     "Recall": t["recall"], "F1": t["f1"], "ROC-AUC": t["roc_auc"]})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
        st.caption("Test metrics at threshold 0.50 unless noted. CV = 5-fold stratified on the "
                   "training split. The test set was not used for model or threshold selection.")
    c1, c2 = st.columns(2)
    with c1:
        figure("roc_curves.png")
        figure("threshold_tuning.png")
    with c2:
        figure("confusion_tuned.png", "Final model at the tuned threshold (test set)")
        figure("calibration.png", "Predicted probabilities vs observed churn rates")


def section_insights(metrics) -> None:
    st.header("Dataset insights")
    if not metrics:
        st.info("Run `python -m src.train` to generate the dataset summary.")
        return
    s = metrics["dataset_summary"]
    st.write(f"Overall churn rate: **{pct(s['churn_rate'])}** of {s['rows']:,} customers. "
             "Charts show the share of customers in each group who churned (whole dataset).")
    labels = {"Contract": "Contract type", "tenure_bucket": "Tenure",
              "InternetService": "Internet service", "PaymentMethod": "Payment method",
              "TechSupport": "Tech support", "OnlineSecurity": "Online security",
              "SeniorCitizen": "Senior citizen", "PaperlessBilling": "Paperless billing"}
    cols = st.columns(2)
    for i, (col, label) in enumerate(labels.items()):
        with cols[i % 2]:
            st.markdown(f"**Churn rate by {label.lower()}**")
            r = rates(s, col)
            if col == "tenure_bucket":  # keep the natural order for an ordinal variable
                r = r.reindex(list(s["churn_rate_by"][col]))
            hbar(r, "churn rate", percent=True)
    mt, mc = s["median_tenure"], s["median_monthly_charges"]
    st.markdown(
        f"Customers who churned had a median tenure of **{mt['churned']:.0f} months** "
        f"(vs {mt['stayed']:.0f} for those who stayed) and median monthly charges of "
        f"**\\${mc['churned']:.2f}** (vs \\${mc['stayed']:.2f})."
    )
    st.caption("These are descriptive rates. Groups overlap (e.g. most fibre customers are on "
               "month-to-month contracts), so a single chart does not isolate one factor.")


def section_importance(metrics) -> None:
    st.header("Which factors are associated with churn?")
    st.write(
        "Three views. **Permutation importance** shows how much test ROC-AUC drops when one input "
        "is shuffled, so it measures how much the final model relies on it. **Logistic regression "
        "coefficients** show direction, holding the other inputs fixed. **Random forest "
        "importance** is a second model's view (impurity-based, biased toward numeric columns)."
    )
    figure("permutation_importance.png")
    c1, c2 = st.columns(2)
    with c1:
        figure("logreg_coefficients.png")
    with c2:
        figure("rf_importance.png")
    st.caption(
        "Correlated inputs share credit. For example, tenure, the tenure buckets and total charges "
        "overlap, so an individual coefficient can have a surprising sign once the others are "
        "included. None of this shows causation."
    )


def section_recommendations(metrics) -> None:
    st.header("Business recommendations")
    if not metrics:
        st.info("Run `python -m src.train` first.")
        return
    s = metrics["dataset_summary"]
    c, t = rates(s, "Contract"), rates(s, "tenure_bucket")
    i, p = rates(s, "InternetService"), rates(s, "PaymentMethod")
    sec = rates(s, "OnlineSecurity")
    st.markdown(
        f"""
Each point is a pattern in this dataset, framed as something to **test**, not a proven cause.

1. **Prioritise month-to-month customers.** {pct(c['Month-to-month'])} of them churned, compared
   with {pct(c['One year'])} on one-year and {pct(c['Two year'])} on two-year contracts. A discount
   for moving to an annual contract is a candidate offer to A/B test.
2. **Focus on the first year.** Customers in their first 12 months churned at {pct(t['0-12m'])},
   compared with {pct(t['49m+'])} after four years. Onboarding check-ins in the first months are a
   low-cost intervention to trial.
3. **Look into the fibre-optic experience.** Fibre customers churned at {pct(i['Fiber optic'])}
   compared with {pct(i['DSL'])} for DSL. The data cannot say why (price, quality or competition), so
   this needs customer research.
4. **Electronic-check payers** churned at {pct(p['Electronic check'])}, compared with
   {pct(p['Credit card (automatic)'])} for automatic credit card payments. Encouraging automatic
   payment is worth testing.
5. **Security / support add-ons.** Internet customers without online security churned at
   {pct(sec['No'])}, compared with {pct(sec['Yes'])} with it. Bundling a trial may help, but
   customers who buy add-ons may simply be more committed already.

**Using the model:** score customers regularly. Contact **HIGH** first, then **MEDIUM**, as
budget allows. At the current threshold the model catches about
{pct(metrics['test_final_tuned']['recall'])} of churners on the test set, but only about
{pct(metrics['test_final_tuned']['precision'])} of flagged customers actually churn. Offers should
therefore be cheap relative to the value of a retained customer.
"""
    )


# ------------------------------------------------------------------ main

def main() -> None:
    try:
        model = get_model()
    except ModelNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    metrics = get_metrics()

    pages = {
        "Overview": lambda: section_overview(model, metrics),
        "Predict a customer": lambda: section_predict(model),
        "Model information": lambda: section_model(model, metrics),
        "Dataset insights": lambda: section_insights(metrics),
        "Feature importance": lambda: section_importance(metrics),
        "Business recommendations": lambda: section_recommendations(metrics),
    }
    choice = st.sidebar.radio("Section", list(pages))
    st.sidebar.caption(f"Model: {model.metadata['model_name']}  \n"
                       f"Trained: {model.metadata['trained_at']}")
    pages[choice]()


main()
