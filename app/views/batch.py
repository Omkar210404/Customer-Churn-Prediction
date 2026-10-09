import pandas as pd
import streamlit as st

from app import ui
from src import config

model = ui.require_model()

ui.page_header(
    "Score a list of customers", icon="upload_file", accent="teal", subtitle=
    "Upload a CSV with one customer per row. Every customer is scored and ranked by risk, "
    "ready to download as a contact list.",
)

with ui.card("prepare", "teal"):
    st.markdown("##### :material/description: 1. Prepare the file")
    st.markdown(
        "Use the same columns as the IBM Telco dataset. `customerID` is optional and is kept in "
        "the output. Blank cells are allowed and filled with typical values, but every value "
        "that is given must be valid (e.g. `Contract` must be *Month-to-month*, *One year* or "
        "*Two year*)."
    )
    c1, c2 = st.columns([1, 2])
    c1.download_button("Download empty template", ui.template_csv(), "churn_template.csv",
                       "text/csv", icon=":material/download:")
    with c2.popover("See allowed values", icon=":material/list:"):
        for col, values in config.CATEGORICAL_VALUES.items():
            st.markdown(f"**{col}**: {', '.join(values)}")
        st.markdown("**tenure**, **MonthlyCharges**, **TotalCharges**: numbers ≥ 0 "
                    "(tenure in months)")

with ui.card("upload", "blue"):
    st.markdown("##### :material/upload: 2. Upload and score")
    upload = st.file_uploader("Customer CSV", type=["csv"], label_visibility="collapsed")

if upload is None:
    st.info("Tip: the original `Telco-Customer-Churn.csv` (see data/README.md) works as-is, "
            "since its extra `Churn` column is ignored.", icon=":material/lightbulb:")
    ui.footer(model)
    st.stop()

try:
    raw = pd.read_csv(upload, dtype=str, keep_default_na=False)
    scored = ui.score_table(model, raw)
except ValueError as exc:
    st.error(f"Could not score this file: {exc}", icon=":material/error:")
    ui.footer(model)
    st.stop()

counts = scored["risk_band"].value_counts()
ui.section("3. Results")
k1, k2, k3, k4 = st.container(key="bandkpis").columns(4)
k1.metric("Customers scored", f"{len(scored):,}", border=True)
k2.metric("High risk", f"{counts.get('HIGH', 0):,}", border=True)
k3.metric("Medium risk", f"{counts.get('MEDIUM', 0):,}", border=True)
k4.metric("Low risk", f"{counts.get('LOW', 0):,}", border=True)

bands = st.pills("Show", ["HIGH", "MEDIUM", "LOW"], selection_mode="multi",
                 default=["HIGH", "MEDIUM", "LOW"])
view = scored[scored["risk_band"].isin(bands)]
st.dataframe(
    view,
    hide_index=True,
    width="stretch",
    column_config={
        "churn_probability": st.column_config.ProgressColumn(
            "Churn probability", format="percent", min_value=0.0, max_value=1.0),
        "risk_band": st.column_config.TextColumn("Risk band"),
        "flagged_for_retention": st.column_config.CheckboxColumn("Flagged"),
    },
)
st.download_button("Download scored list (CSV)", scored.to_csv(index=False),
                   "churn_scores.csv", "text/csv", type="primary", icon=":material/download:")
st.caption(f"Flagged = probability ≥ {model.threshold:.2f}. Sorted from highest to lowest risk.")

ui.footer(model)
