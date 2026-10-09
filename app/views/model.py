import pandas as pd
import streamlit as st

from app import ui
from src import config

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

st.markdown("### Charts")
tabs = st.tabs(["ROC curves", "Threshold tuning", "Confusion matrix", "Calibration"])
figures = [
    ("roc_curves.png", "All three models rank customers almost identically (test set)."),
    ("threshold_tuning.png", "Precision, recall, F1 and F2 at every threshold, on out-of-fold "
                             "training predictions. The dashed line is the chosen threshold."),
    ("confusion_tuned.png", f"Final model at threshold {model.threshold:.2f} on the test set."),
    ("calibration.png", "Predicted probabilities closely match observed churn rates, so the "
                        "percentages shown in the app can be read at face value."),
]
for tab, (name, caption) in zip(tabs, figures):
    with tab:
        c, _ = st.columns([3, 2])
        path = config.FIGURES_DIR / name
        if path.exists():
            c.image(str(path), caption=caption)
        else:
            c.info(f"{name} not found. Run `python -m src.train`.")

ui.footer(model)
