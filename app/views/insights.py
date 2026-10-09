import streamlit as st

from app import ui

model = ui.require_model()
metrics = ui.require_metrics()
summary = metrics["dataset_summary"]
overall = summary["churn_rate"]

ui.page_header("Data insights",
               "How churn differs between customer groups across the whole dataset "
               f"({summary['rows']:,} customers).")

mt, mc = summary["median_tenure"], summary["median_monthly_charges"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Overall churn rate", ui.pct(overall), border=True)
k2.metric("Customers who churned", f"{summary['churned']:,}", border=True)
k3.metric("Median tenure, churned", f"{mt['churned']:.0f} months",
          delta=f"{mt['churned'] - mt['stayed']:.0f} vs stayed", delta_color="off", border=True)
k4.metric("Median monthly bill, churned", f"${mc['churned']:.2f}",
          delta=f"+${mc['churned'] - mc['stayed']:.2f} vs stayed", delta_color="off", border=True)

FACTORS = {
    "Contract": ("Contract", False),
    "Tenure": ("tenure_bucket", True),
    "Internet": ("InternetService", False),
    "Payment": ("PaymentMethod", False),
    "Tech support": ("TechSupport", False),
    "Online security": ("OnlineSecurity", False),
    "Senior citizen": ("SeniorCitizen", False),
    "Paperless billing": ("PaperlessBilling", False),
    "Partner": ("Partner", False),
    "Dependents": ("Dependents", False),
}

with st.container(border=True):
    choice = st.segmented_control("Break down churn by", list(FACTORS), default="Contract",
                                  required=True)
    col, keep_order = FACTORS[choice]
    df = ui.group_rates(metrics, col)
    st.altair_chart(ui.rate_bars(df, overall, sort=not keep_order))
    hi = df.loc[df["churn_rate"].idxmax()]
    lo = df.loc[df["churn_rate"].idxmin()]
    st.markdown(
        f"**{hi['group']}** has the highest churn rate (**{ui.pct(hi['churn_rate'])}** of "
        f"{hi['customers']:,} customers), **{hi['churn_rate'] / lo['churn_rate']:.1f}×** the rate "
        f"of **{lo['group']}** ({ui.pct(lo['churn_rate'])}). The dashed line is the overall "
        f"average of {ui.pct(overall)}."
    )

st.markdown("### All factors at a glance")
items = list(FACTORS.items())
for start in range(0, len(items), 2):
    cols = st.columns(2)
    for slot, (name, (col, keep_order)) in zip(cols, items[start:start + 2]):
        with slot.container(border=True, height="stretch"):
            st.markdown(f"**{name}**")
            st.altair_chart(ui.rate_bars(ui.group_rates(metrics, col), overall,
                                         sort=not keep_order))

st.info(
    "These are descriptive rates for single factors. Groups overlap (most fibre customers are "
    "also on month-to-month contracts), so one chart alone does not isolate the effect of one "
    "factor. The **Churn drivers** page shows what the model relies on when all inputs are "
    "considered together.",
    icon=":material/info:",
)

ui.footer(model)
