import pandas as pd
import streamlit as st

from app import ui

model = ui.require_model()
metrics = ui.require_metrics()
meta = model.metadata
tuned = metrics["test_final_tuned"]
cm = tuned["confusion_matrix"]

ui.page_header("Model performance",
               "How the models compare, why this one was chosen, and how well it does on "
               f"{meta['test_rows']:,} customers it never saw during training.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("ROC-AUC (test)", f"{tuned['roc_auc']:.3f}",
          help="How well the model ranks churners above non-churners. 0.5 = random, 1 = perfect.",
          border=True)
k2.metric("Churners caught", f"{cm['tp']} of {cm['tp'] + cm['fn']}",
          delta=ui.pct(tuned["recall"]) + " recall", delta_color="off", border=True)
k3.metric("Flagged who churned", f"{cm['tp']} of {cm['tp'] + cm['fp']}",
          delta=ui.pct(tuned["precision"]) + " precision", delta_color="off", border=True)
k4.metric("Decision threshold", f"{model.threshold:.2f}",
          help="Customers at or above this probability are flagged.", border=True)

with st.container(border=True):
    st.markdown("##### :material/compare: Model comparison")
    rows = []
    for name, m in metrics["test"].items():
        c = metrics["cv"][name]
        rows.append({"Model": name,
                     "CV ROC-AUC": f"{c['cv_roc_auc_mean']:.3f} ± {c['cv_roc_auc_std']:.3f}",
                     "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
                     "F1": m["f1"], "ROC-AUC": m["roc_auc"]})
    rows.append({"Model": f"★ {meta['model_name']} @ {model.threshold:.2f} (final)",
                 "CV ROC-AUC": "—", "Accuracy": tuned["accuracy"],
                 "Precision": tuned["precision"], "Recall": tuned["recall"], "F1": tuned["f1"],
                 "ROC-AUC": tuned["roc_auc"]})
    num = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={k: num for k in rows[0] if k not in ("Model", "CV ROC-AUC")})
    st.caption("Test-set metrics at threshold 0.50 unless noted. CV = 5-fold cross-validation on "
               "the training data. The test set was not used to choose the model or the threshold.")
    st.markdown(f"**Why {meta['model_name']}?** All three models are within 0.005 ROC-AUC of each "
                "other, which is less than the cross-validation noise. The rule set before "
                "training was: in a tie, pick the most interpretable model. Logistic regression "
                "also gives an exact explanation for every single prediction.")

with st.container(border=True):
    st.markdown("##### :material/tune: The threshold trade-off")
    default = metrics["test"][meta["model_name"]]["confusion_matrix"]
    trade = pd.DataFrame(
        {"Threshold 0.50": [default["tp"], default["fn"], default["fp"], default["tp"] + default["fp"]],
         f"Threshold {model.threshold:.2f} (used)": [cm["tp"], cm["fn"], cm["fp"], cm["tp"] + cm["fp"]]},
        index=["Churners caught", "Churners missed", "False alarms", "Customers flagged"],
    )
    a, b = st.columns([1, 1.2], gap="large")
    a.dataframe(trade, width="stretch")
    b.markdown(
        f"Lowering the threshold to **{model.threshold:.2f}** cuts missed churners from "
        f"**{default['fn']}** to **{cm['fn']}**, at the cost of more false alarms "
        f"({default['fp']} → {cm['fp']}). That's a good deal when a retention offer is cheap "
        "compared with losing a customer. With a small budget, contact customers from the "
        "highest probability down instead."
    )

curves = metrics.get("curves")
if curves is None:
    st.info("Run `python -m src.train` to generate the chart data.", icon=":material/info:")
else:
    aucs = {name: m["roc_auc"] for name, m in metrics["test"].items()}
    r1, r2 = st.columns(2)
    with r1.container(border=True, height="stretch"):
        st.markdown("##### ROC curves (test set)")
        st.altair_chart(ui.roc_chart(curves["roc_test"], aucs))
        st.caption("All three models rank customers almost identically. The dashed diagonal is "
                   "random guessing.")
    with r2.container(border=True, height="stretch"):
        st.markdown("##### Threshold tuning")
        st.altair_chart(ui.threshold_chart(curves["threshold_oof"], model.threshold))
        st.caption("Scores at every threshold, on out-of-fold training predictions. Hover to "
                   "read values. F2 peaks near the chosen threshold.")
    r3, r4 = st.columns(2)
    with r3.container(border=True, height="stretch"):
        st.markdown(f"##### Confusion matrix at {model.threshold:.2f} (test set)")
        st.altair_chart(ui.confusion_chart(cm))
        st.caption(f"Of {meta['test_rows']:,} unseen customers: {cm['tp']} churners caught, "
                   f"{cm['fn']} missed, {cm['fp']} false alarms.")
    with r4.container(border=True, height="stretch"):
        st.markdown("##### Calibration")
        st.altair_chart(ui.calibration_chart(curves["calibration_oof"]))
        st.caption("Predicted probabilities closely match observed churn rates (dashed line = "
                   "perfect), so the percentages in the app can be read at face value.")

ui.footer(model)
