import streamlit as st

from app import ui
from src import config

model = ui.require_model()
ss = st.session_state

NO_INTERNET, NO_PHONE = "No internet service", "No phone service"
DEFAULTS = {
    "tenure": 12, "contract": "Month-to-month", "payment": "Electronic check",
    "paperless": True, "monthly": 70.0, "exact_total": False, "total": 840.0,
    "phone": True, "multi": False, "internet": "Fiber optic", "addons": [],
    "gender": "Female", "senior": False, "partner": False, "dependents": False,
}
# Example inputs to try the scorer quickly (not real customers).
PRESETS = {
    "New fibre customer": {
        "tenure": 3, "contract": "Month-to-month", "payment": "Electronic check", "paperless": True,
        "monthly": 85.0, "phone": True, "multi": False, "internet": "Fiber optic",
        "addons": ["StreamingTV"], "gender": "Female", "senior": False, "partner": False,
        "dependents": False,
    },
    "Loyal long-term customer": {
        "tenure": 60, "contract": "Two year", "payment": "Bank transfer (automatic)",
        "paperless": False, "monthly": 65.0, "phone": True, "multi": True, "internet": "DSL",
        "addons": ["OnlineSecurity", "OnlineBackup", "TechSupport"], "gender": "Male",
        "senior": False, "partner": True, "dependents": True,
    },
    "Phone-only customer": {
        "tenure": 24, "contract": "One year", "payment": "Mailed check", "paperless": False,
        "monthly": 20.0, "phone": True, "multi": False, "internet": "No", "addons": [],
        "gender": "Male", "senior": False, "partner": True, "dependents": False,
    },
}
for key, value in DEFAULTS.items():
    ss.setdefault(key, value)


def apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        ss[key] = value
    ss["exact_total"] = False


def seed_total() -> None:
    if ss["exact_total"]:
        ss["total"] = round(ss["tenure"] * ss["monthly"], 2)


ui.page_header("Score a customer",
               "Describe a customer on the left. The score on the right updates instantly.")

with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown("**Try an example:**", width="content")
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,), icon=":material/person:", type="tertiary")

left, right = st.columns([1.35, 1], gap="large")

# ------------------------------------------------------------------ inputs
with left:
    with st.container(border=True):
        st.markdown("##### :material/receipt_long: Contract & billing")
        st.segmented_control("Contract", config.CATEGORICAL_VALUES["Contract"],
                             key="contract", required=True)
        st.slider("Tenure (months with the company)", 0, 72, key="tenure",
                  help="The dataset covers 0 to 72 months.")
        c1, c2 = st.columns(2)
        c1.selectbox("Payment method", config.CATEGORICAL_VALUES["PaymentMethod"], key="payment")
        c2.slider("Monthly charges ($)", 18.0, 120.0, step=0.25, key="monthly",
                  help="The dataset ranges from about $18 to $119 per month.")
        c3, c4 = st.columns(2)
        c3.toggle("Paperless billing", key="paperless")
        c4.toggle("I know the exact total charges", key="exact_total", on_change=seed_total)
        if ss["exact_total"]:
            st.number_input("Total charges to date ($)", 0.0, 100000.0, step=10.0, key="total")
            total = float(ss["total"])
        else:
            total = round(ss["tenure"] * ss["monthly"], 2)
            st.caption(f"Total charges estimated as tenure × monthly charges = **${total:,.2f}**")

    with st.container(border=True):
        st.markdown("##### :material/router: Services")
        st.segmented_control("Internet service", config.CATEGORICAL_VALUES["InternetService"],
                             key="internet", required=True,
                             format_func=lambda v: "None" if v == "No" else v)
        has_internet = ss["internet"] != "No"
        if not has_internet:
            ss["addons"] = []
        st.pills("Add-on services", config.ADDON_SERVICES, selection_mode="multi", key="addons",
                 format_func=ui.LABELS.get, disabled=not has_internet,
                 help="Add-ons need an internet service.")
        p1, p2 = st.columns(2)
        p1.toggle("Phone service", key="phone")
        if not ss["phone"]:
            ss["multi"] = False
        p2.toggle("Multiple phone lines", key="multi", disabled=not ss["phone"])

    with st.container(border=True):
        st.markdown("##### :material/person: Customer profile")
        st.segmented_control("Gender", config.CATEGORICAL_VALUES["gender"], key="gender",
                             required=True)
        g1, g2, g3 = st.columns(3)
        g1.toggle("Senior citizen", key="senior")
        g2.toggle("Has partner", key="partner")
        g3.toggle("Has dependents", key="dependents")


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


record = {
    "gender": ss["gender"],
    "SeniorCitizen": yes_no(ss["senior"]),
    "Partner": yes_no(ss["partner"]),
    "Dependents": yes_no(ss["dependents"]),
    "tenure": ss["tenure"],
    "PhoneService": yes_no(ss["phone"]),
    "MultipleLines": yes_no(ss["multi"]) if ss["phone"] else NO_PHONE,
    "InternetService": ss["internet"],
    "Contract": ss["contract"],
    "PaperlessBilling": yes_no(ss["paperless"]),
    "PaymentMethod": ss["payment"],
    "MonthlyCharges": float(ss["monthly"]),
    "TotalCharges": total,
}
for addon in config.ADDON_SERVICES:
    record[addon] = (yes_no(addon in ss["addons"]) if has_internet else NO_INTERNET)

# ------------------------------------------------------------------ result
with right:
    try:
        result = model.predict(record).iloc[0]
    except ValueError as exc:
        st.error(f"Invalid input: {exc}", icon=":material/error:")
        st.stop()

    p, band = float(result["churn_probability"]), result["risk_band"]
    info = ui.BANDS[band]
    band_stats = model.metadata["oof_band_stats"][band]

    with st.container(border=True):
        st.caption("Churn probability")
        st.markdown(f'<div class="big-number">{ui.pct(p)}</div>', unsafe_allow_html=True)
        st.badge(info["headline"], icon=info["icon"], color=info["badge"])
        st.altair_chart(ui.risk_meter(p, model.threshold, model.high_from))
        st.markdown(
            f"Customers the model scored in the **{band}** band churned "
            f"**{ui.pct(band_stats['churn_rate'], 0)}** of the time in cross-validation."
        )
        st.markdown(f"**Suggested next step:** {info['action']}")

    if model.is_linear:
        with st.container(border=True):
            st.markdown("##### What drives this score")
            contrib = model.explain(record).head(6)
            contrib.index = [ui.pretty_feature(i) for i in contrib.index]
            st.altair_chart(ui.signed_bars(contrib, "Effect compared with an average customer"))
            st.caption(
                "Red pushes this customer's risk up, blue pulls it down, compared with the "
                "average customer in the training data. Inputs overlap (monthly charges mostly "
                "reflect the services), so read related inputs together. These are model "
                "associations, not causes."
            )

    with st.expander("How are the risk bands defined?", icon=":material/help:"):
        stats = model.metadata["oof_band_stats"]
        st.markdown(
            f"""
| Band | Churn probability | Customers in band | Actually churned |
|---|---|---|---|
| :green[**LOW**] | below {model.threshold:.2f} | {ui.pct(stats['LOW']['share'], 0)} | {ui.pct(stats['LOW']['churn_rate'], 0)} |
| :orange[**MEDIUM**] | {model.threshold:.2f} – {model.high_from:.2f} | {ui.pct(stats['MEDIUM']['share'], 0)} | {ui.pct(stats['MEDIUM']['churn_rate'], 0)} |
| :red[**HIGH**] | {model.high_from:.2f} or more | {ui.pct(stats['HIGH']['share'], 0)} | {ui.pct(stats['HIGH']['churn_rate'], 0)} |

Shares and churn rates come from cross-validation on the training data. Customers at or above
**{model.threshold:.2f}** are flagged for retention. This threshold catches most churners
(it maximises the F2 score, which weights recall twice as much as precision).
"""
        )

ui.footer(model)
