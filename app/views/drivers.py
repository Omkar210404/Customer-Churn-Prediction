import pandas as pd
import streamlit as st

from app import ui

model = ui.require_model()
metrics = ui.require_metrics()
explain = metrics["explainability"]

ui.page_header("Churn drivers",
               "Which inputs the model relies on, and in which direction they push the risk.")

perm = (pd.DataFrame(explain["permutation_importance"]).T
        .rename_axis("feature").reset_index().sort_values("mean", ascending=False))
perm["feature"] = perm["feature"].map(ui.pretty_feature)
top = perm.iloc[0]

with st.container(border=True):
    st.markdown("##### :material/leaderboard: What the model relies on most")
    st.markdown(
        f"**{top['feature']}** matters far more than anything else. Shuffling it lowers the "
        f"model's ROC-AUC by **{top['mean']:.3f}**. The next four "
        f"({', '.join(perm['feature'].iloc[1:5])}) matter much less, and the rest add very little."
    )
    st.altair_chart(ui.importance_bars(perm.head(10)))
    st.caption(
        "Permutation importance on the 1,409 held-out test customers: each input is shuffled "
        "10 times and the average drop in ROC-AUC is shown. Larger means the model depends on "
        "it more. Top 10 of 19 inputs."
    )

coefs = pd.Series(explain["logreg_coefficients"])
coefs = coefs[~coefs.index.str.endswith(("_No internet service", "_No phone service"))]
coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index[:12]).sort_values(ascending=False)
coefs.index = [ui.pretty_feature(i) for i in coefs.index]

with st.container(border=True):
    st.markdown("##### :material/swap_vert: Which way each input pushes the risk")
    st.altair_chart(ui.signed_bars(coefs, "Logistic regression coefficient (log-odds)"))
    st.caption(
        "The 12 largest coefficients, sorted from most risk-raising to most risk-lowering. "
        "Each one is the effect of an input *with all other inputs held fixed*. Numeric inputs "
        "are standardised, and categories are compared with the other options of the same input."
    )

with st.container(border=True):
    st.markdown("##### :material/forum: In plain language")
    st.markdown(
        """
- **Newer customers are the most likely to leave.** Risk falls steadily as tenure grows.
- **Month-to-month contracts** carry much more risk than one- or two-year contracts.
- **Fibre-optic internet** goes with higher risk than DSL, even at the same contract type.
- Demographics (gender, partner, dependents) add almost nothing once contract, tenure and
  services are known.
"""
    )
    st.warning(
        "Some coefficients look surprising. For example, *Monthly charges* and *Tenure group: 49m+* "
        "have the opposite sign to what the raw data suggests. That's because they overlap with "
        "other inputs (charges mostly reflect the services; tenure group overlaps with tenure), so "
        "the model splits the effect between them. Read related inputs together. None of this "
        "proves cause and effect.",
        icon=":material/warning:",
    )

ui.footer(model)
