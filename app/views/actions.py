import streamlit as st

from app import ui

model = ui.require_model()
metrics = ui.require_metrics()
tuned = metrics["test_final_tuned"]


def rate(col: str, group: str) -> float:
    return ui.group_rates(metrics, col).set_index("group").loc[group, "churn_rate"]


ui.page_header("Recommendations",
               "What a retention team could do with these results. Each idea is based on a "
               "pattern in the data, so treat it as something to test, not a proven fix.")

IDEAS = [
    (":material/contract:", "Move customers onto longer contracts",
     f"Month-to-month customers churned at **{ui.pct(rate('Contract', 'Month-to-month'))}**, "
     f"compared with **{ui.pct(rate('Contract', 'Two year'))}** on two-year contracts.",
     "A/B test a discount or perk for switching to an annual contract.",
     "Part of the gap is the contract's lock-in itself, not happier customers."),
    (":material/waving_hand:", "Look after new customers",
     f"Customers in their first year churned at **{ui.pct(rate('tenure_bucket', '0-12m'))}**, "
     f"compared with **{ui.pct(rate('tenure_bucket', '49m+'))}** after four years.",
     "Onboarding calls and check-ins during the first 3 to 6 months.",
     "Tenure is the strongest single input in the model."),
    (":material/wifi:", "Investigate the fibre-optic experience",
     f"Fibre customers churned at **{ui.pct(rate('InternetService', 'Fiber optic'))}**, "
     f"compared with **{ui.pct(rate('InternetService', 'DSL'))}** on DSL, and the gap holds "
     "for every contract type.",
     "Survey fibre leavers: price, reliability or a competitor offer?",
     "The data cannot say why. It only shows that the gap exists."),
    (":material/credit_card:", "Encourage automatic payment",
     f"Electronic-check payers churned at **{ui.pct(rate('PaymentMethod', 'Electronic check'))}**, "
     f"compared with **{ui.pct(rate('PaymentMethod', 'Credit card (automatic)'))}** for "
     "automatic card payments.",
     "Test a small incentive for switching to automatic payment.",
     "Payment method may simply reflect how engaged a customer already is."),
    (":material/shield:", "Offer security and support add-ons",
     f"Internet customers without online security churned at "
     f"**{ui.pct(rate('OnlineSecurity', 'No'))}**, compared with "
     f"**{ui.pct(rate('OnlineSecurity', 'Yes'))}** with it.",
     "Bundle a free trial of online security or tech support.",
     "Customers who buy add-ons may already be more committed."),
]

for start in range(0, len(IDEAS), 2):
    cols = st.columns(2)
    for col, (icon, title, evidence, action, caveat) in zip(cols, IDEAS[start:start + 2]):
        with col.container(border=True, height="stretch"):
            st.markdown(f"##### {icon} {title}")
            st.markdown(evidence)
            st.markdown(f":material/arrow_forward: **Try:** {action}")
            st.caption(f"Caveat: {caveat}")

with st.container(border=True):
    st.markdown("##### :material/checklist: Using the model day to day")
    stats = model.metadata["oof_band_stats"]
    st.markdown(
        f"""
| Band | Share of customers | Actually churned | Suggested action |
|---|---|---|---|
| :red[**HIGH**] | {ui.pct(stats['HIGH']['share'], 0)} | {ui.pct(stats['HIGH']['churn_rate'], 0)} | {ui.BANDS['HIGH']['action']} |
| :orange[**MEDIUM**] | {ui.pct(stats['MEDIUM']['share'], 0)} | {ui.pct(stats['MEDIUM']['churn_rate'], 0)} | {ui.BANDS['MEDIUM']['action']} |
| :green[**LOW**] | {ui.pct(stats['LOW']['share'], 0)} | {ui.pct(stats['LOW']['churn_rate'], 0)} | {ui.BANDS['LOW']['action']} |
"""
    )
    st.markdown(
        f"Score the customer base regularly (the **Score a list** page) and work down from the "
        f"highest probability. At the current threshold the model catches "
        f"**{ui.pct(tuned['recall'])}** of churners, but only **{ui.pct(tuned['precision'])}** of "
        "flagged customers actually leave, so offers should be cheap compared with the value of a "
        "retained customer."
    )

ui.footer(model)
