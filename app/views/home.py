import streamlit as st

from app import ui

model = ui.require_model()
metrics = ui.require_metrics()
meta = model.metadata
summary = metrics["dataset_summary"]
tuned = metrics["test_final_tuned"]

ui.hero(
    "Customer Churn Prediction",
    "Find the telecom customers most likely to cancel, see why, and decide who to contact first. "
    "Built on the public IBM Telco Customer Churn sample dataset.",
    [f"{summary['rows']:,} customers", f"{meta['model_name']}",
     f"ROC-AUC {tuned['roc_auc']:.3f} (test set)", "scikit-learn · Streamlit"],
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers in the data", f"{summary['rows']:,}", border=True)
c2.metric("Churned", ui.pct(summary["churn_rate"]),
          help=f"{summary['churned']:,} customers left.", border=True)
c3.metric("Churners the model catches", ui.pct(tuned["recall"]),
          help=f"Recall on the held-out test set at threshold {model.threshold:.2f}.", border=True)
c4.metric("Flagged customers who churn", ui.pct(tuned["precision"]),
          help="Precision on the test set: the share of flagged customers who actually left.",
          border=True)

ui.section("What you can do here")
a, b, c = st.columns(3)
with ui.card("do-score", "violet", parent=a, height="stretch"):
    st.markdown("#### :material/person_search: Score one customer")
    st.write("Enter a customer's plan and services, and get their churn probability, "
             "risk band and the main reasons, updated as you type.")
    st.page_link("views/predict.py", label="Open the scorer", icon=":material/arrow_forward:")
with ui.card("do-batch", "teal", parent=b, height="stretch"):
    st.markdown("#### :material/upload_file: Score a whole list")
    st.write("Upload a CSV of customers. Get everyone scored and ranked by risk, "
             "ready to download as a contact list.")
    st.page_link("views/batch.py", label="Upload a list", icon=":material/arrow_forward:")
with ui.card("do-explore", "amber", parent=c, height="stretch"):
    st.markdown("#### :material/bar_chart: Explore the patterns")
    st.write("See which customer groups churn most, which inputs the model relies on, "
             "and what the business could test next.")
    st.page_link("views/insights.py", label="See the insights", icon=":material/arrow_forward:")

ui.section("Key findings")
rates = {col: ui.group_rates(metrics, col).set_index("group")["churn_rate"]
         for col in ["Contract", "tenure_bucket", "InternetService", "PaymentMethod"]}
f1, f2, f3, f4 = st.columns(4)
findings = [
    (f1, "Month-to-month contracts", rates["Contract"]["Month-to-month"],
     f"vs {ui.pct(rates['Contract']['Two year'])} on two-year contracts"),
    (f2, "First 12 months", rates["tenure_bucket"]["0-12m"],
     f"vs {ui.pct(rates['tenure_bucket']['49m+'])} after 4+ years"),
    (f3, "Fibre-optic internet", rates["InternetService"]["Fiber optic"],
     f"vs {ui.pct(rates['InternetService']['DSL'])} on DSL"),
    (f4, "Electronic-check payers", rates["PaymentMethod"]["Electronic check"],
     f"vs {ui.pct(rates['PaymentMethod']['Credit card (automatic)'])} on automatic card"),
]
for i, (col, title, value, compare) in enumerate(findings):
    accent = ["pink", "amber", "blue", "teal"][i]
    with ui.card(f"finding-{i}", accent, parent=col, height="stretch"):
        st.caption(title)
        st.markdown(f'<div class="stat-value" style="color:{ui.ACCENTS[accent][0]}">'
                    f'{ui.pct(value)}</div>', unsafe_allow_html=True)
        st.caption(f"churned, {compare}")

with st.expander("How does it work?", icon=":material/help:"):
    st.markdown(
        f"""
1. **Data.** 7,043 customers with their contract, services, charges and whether they left
   (IBM sample data for a fictional telco).
2. **Cleaning.** Fixed types and filled the 11 blank *Total charges* (all brand-new customers) with 0.
   No rows were removed.
3. **Models.** Logistic regression, random forest and gradient boosting were compared with 5-fold
   cross-validation. All three scored about the same (ROC-AUC ≈ 0.85), so the simplest and most
   explainable one, **logistic regression**, was chosen.
4. **Threshold.** A customer is flagged when their churn probability is at least
   **{model.threshold:.2f}**. That threshold was chosen to catch most churners, because missing
   one usually costs more than an unneeded retention offer.
5. **Honest testing.** Final numbers come from 1,409 customers the model never saw during training.
"""
    )

ui.footer(model)
