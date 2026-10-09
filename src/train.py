"""Train, compare and save the churn model.

Usage:
    python -m src.train

Steps: load + clean -> stratified split -> feature-engineering ablation (CV) ->
grid search per model (CV ROC-AUC) -> select model -> tune threshold on
out-of-fold predictions -> evaluate once on the test set -> explainability ->
save pipeline, metadata, metrics and figures.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src import config
from src import evaluate as ev
from src.data_loader import download_dataset, load_raw
from src.features import add_features, feature_engineering_step, model_input_columns
from src.preprocessing import CleaningReport, build_preprocessor, clean_raw, split_xy

# Lower = easier to explain to a business audience. Used as the tie-break.
INTERPRETABILITY_RANK = {"Logistic Regression": 0, "Random Forest": 1, "HistGradientBoosting": 2}
TIE_MARGIN = 0.005  # CV ROC-AUC difference treated as a tie
FE_TOLERANCE = 0.001  # keep engineered features unless they lower mean CV AUC by more than this
HIGH_RISK_PROBABILITY = 0.50  # HIGH band: model says churn is more likely than not


def make_pipeline(model, engineered: bool) -> Pipeline:
    numeric, categorical = model_input_columns(engineered)
    steps = []
    if engineered:
        steps.append(("features", feature_engineering_step()))
    steps += [("preprocess", build_preprocessor(numeric, categorical)), ("model", model)]
    return Pipeline(steps)


def base_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=config.RANDOM_STATE, n_jobs=1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }


PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1.0, 10.0]},
    "Random Forest": {
        "model__max_depth": [None, 8, 12],
        "model__min_samples_leaf": [1, 5, 10],
    },
    "HistGradientBoosting": {
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [3, 6],
        "model__max_iter": [100, 300],
    },
}


def feature_ablation(X, y, cv) -> dict:
    """CV ROC-AUC of each base model with and without engineered features."""
    rows = {}
    for name, model in base_models().items():
        scores = {}
        for engineered in (False, True):
            s = cross_val_score(make_pipeline(clone(model), engineered), X, y,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
            scores["with_fe" if engineered else "without_fe"] = float(s.mean())
        scores["delta"] = scores["with_fe"] - scores["without_fe"]
        rows[name] = scores
        print(f"  {name:22s} without {scores['without_fe']:.4f}  with {scores['with_fe']:.4f}"
              f"  delta {scores['delta']:+.4f}")
    return rows


def clean_feature_name(name: str) -> str:
    return name.split("__", 1)[-1]


def dataset_summary(df: pd.DataFrame) -> dict:
    """Descriptive churn rates for the app's insights section (whole dataset)."""
    df = add_features(df)
    groups = ["Contract", "InternetService", "PaymentMethod", "tenure_bucket",
              "SeniorCitizen", "TechSupport", "OnlineSecurity", "PaperlessBilling",
              "Partner", "Dependents"]
    by = {}
    for col in groups:
        g = df.groupby(col, observed=True)[config.TARGET].agg(["mean", "count"])
        by[col] = {str(k): {"churn_rate": float(r["mean"]), "customers": int(r["count"])}
                   for k, r in g.iterrows()}
    churned = df[df[config.TARGET] == 1]
    stayed = df[df[config.TARGET] == 0]
    return {
        "rows": int(len(df)),
        "churn_rate": float(df[config.TARGET].mean()),
        "churned": int(len(churned)),
        "median_tenure": {"churned": float(churned["tenure"].median()),
                          "stayed": float(stayed["tenure"].median())},
        "median_monthly_charges": {"churned": float(churned["MonthlyCharges"].median()),
                                   "stayed": float(stayed["MonthlyCharges"].median())},
        "churn_rate_by": by,
    }


def main() -> None:
    t0 = time.time()
    for d in (config.MODELS_DIR, config.FIGURES_DIR, config.PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data
    download_dataset()
    report = CleaningReport()
    df = clean_raw(load_raw(), report)
    print("Cleaning:\n  " + "\n  ".join(report.steps))
    df.to_csv(config.PROCESSED_DIR / "telco_clean.csv", index=False)
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    print(f"Split: train {len(X_train):,} (churn {y_train.mean():.3f}), "
          f"test {len(X_test):,} (churn {y_test.mean():.3f})")
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    # 2. Feature-engineering ablation
    print("Feature-engineering ablation (5-fold CV ROC-AUC, default params):")
    ablation = feature_ablation(X_train, y_train, cv)
    mean_delta = float(np.mean([r["delta"] for r in ablation.values()]))
    engineered = mean_delta >= -FE_TOLERANCE
    print(f"  mean delta {mean_delta:+.4f} -> engineered features "
          f"{'KEPT' if engineered else 'DROPPED'}")

    # 3. Baseline + grid search
    dummy_cv = cross_val_score(DummyClassifier(strategy="prior"), X_train, y_train,
                               cv=cv, scoring="roc_auc")
    searches = {}
    cv_summary = {"Dummy (prior)": {"cv_roc_auc_mean": float(dummy_cv.mean()),
                                    "cv_roc_auc_std": float(dummy_cv.std()),
                                    "best_params": {}}}
    for name, model in base_models().items():
        ts = time.time()
        gs = GridSearchCV(make_pipeline(model, engineered), PARAM_GRIDS[name],
                          scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        i = gs.best_index_
        searches[name] = gs
        cv_summary[name] = {
            "cv_roc_auc_mean": float(gs.cv_results_["mean_test_score"][i]),
            "cv_roc_auc_std": float(gs.cv_results_["std_test_score"][i]),
            "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
        }
        print(f"  {name:22s} CV AUC {cv_summary[name]['cv_roc_auc_mean']:.4f} "
              f"+/- {cv_summary[name]['cv_roc_auc_std']:.4f}  {cv_summary[name]['best_params']}"
              f"  ({time.time() - ts:.1f}s)")

    # 4. Model selection (CV only; the test set is not consulted)
    best_auc = max(cv_summary[n]["cv_roc_auc_mean"] for n in searches)
    contenders = [n for n in searches if cv_summary[n]["cv_roc_auc_mean"] >= best_auc - TIE_MARGIN]
    final_name = min(contenders, key=INTERPRETABILITY_RANK.get)
    top_name = max(searches, key=lambda n: cv_summary[n]["cv_roc_auc_mean"])
    selection_reason = (
        f"Highest mean CV ROC-AUC was {top_name} ({best_auc:.4f}). Models within "
        f"{TIE_MARGIN} of it: {contenders}. Chose the most interpretable of these: {final_name}."
    )
    print("Selection: " + selection_reason)
    final_pipe = searches[final_name].best_estimator_

    # 5. Threshold tuning on out-of-fold training predictions
    oof = cross_val_predict(clone(final_pipe), X_train, y_train, cv=cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    threshold = ev.best_f2_threshold(y_train, oof)
    # Bands: LOW = below the decision threshold (not flagged); MEDIUM = flagged;
    # HIGH = churn more likely than not. Only meaningful if probabilities are
    # roughly calibrated, which the calibration plot checks.
    high_cut = max(HIGH_RISK_PROBABILITY, threshold)
    oof_metrics = ev.compute_metrics(y_train, oof, threshold)
    bands = ev.band_stats(y_train, oof, threshold, high_cut)
    print(f"Threshold (max F2 on OOF): {threshold:.2f}; HIGH from {high_cut:.2f}")
    for b, s in bands.items():
        print(f"  {b:6s} share {s['share']:.3f}  observed churn {s['churn_rate']:.3f}")
    ev.plot_threshold_curve(ev.threshold_curve(y_train, oof), threshold,
                            config.FIGURES_DIR / "threshold_tuning.png")
    ev.plot_calibration(y_train, oof, f"{final_name}: calibration (out-of-fold, train)",
                        config.FIGURES_DIR / "calibration.png")

    # 6. Test-set evaluation (all models at 0.5, final model also at tuned threshold)
    test_probs = {n: gs.best_estimator_.predict_proba(X_test)[:, 1] for n, gs in searches.items()}
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    test_metrics = {"Dummy (prior)": ev.compute_metrics(y_test, dummy.predict_proba(X_test)[:, 1])}
    for n, p in test_probs.items():
        test_metrics[n] = ev.compute_metrics(y_test, p, 0.5)
    final_tuned = ev.compute_metrics(y_test, test_probs[final_name], threshold)
    ev.plot_roc_curves(y_test, test_probs, config.FIGURES_DIR / "roc_curves.png")
    ev.plot_confusion(test_metrics[final_name]["confusion_matrix"],
                      f"{final_name} @ 0.50", config.FIGURES_DIR / "confusion_default.png")
    ev.plot_confusion(final_tuned["confusion_matrix"],
                      f"{final_name} @ {threshold:.2f}", config.FIGURES_DIR / "confusion_tuned.png")
    print("Test set:")
    for n, m in test_metrics.items():
        print(f"  {n:22s} acc {m['accuracy']:.3f} prec {m['precision']:.3f} "
              f"rec {m['recall']:.3f} f1 {m['f1']:.3f} auc {m['roc_auc']:.3f}")
    print(f"  {final_name + ' @tuned':22s} acc {final_tuned['accuracy']:.3f} "
          f"prec {final_tuned['precision']:.3f} rec {final_tuned['recall']:.3f} "
          f"f1 {final_tuned['f1']:.3f} auc {final_tuned['roc_auc']:.3f}")

    # 7. Explainability
    explain = {}
    lr = searches["Logistic Regression"].best_estimator_
    names = [clean_feature_name(n) for n in lr.named_steps["preprocess"].get_feature_names_out()]
    coefs = pd.Series(lr.named_steps["model"].coef_[0], index=names).sort_values()
    explain["logreg_coefficients"] = coefs.to_dict()
    # "No internet service" / "No phone service" dummies duplicate InternetService_No /
    # PhoneService_No exactly, so they are left out of the chart (kept in metrics.json).
    shown = coefs[~coefs.index.str.endswith(("_No internet service", "_No phone service"))]
    top = shown.reindex(shown.abs().sort_values(ascending=False).index[:15]).sort_values(ascending=False)
    ev.plot_barh(top, "Logistic Regression: largest coefficients (+ = higher churn odds)\n"
                 "duplicate 'No internet/phone service' dummies omitted",
                 "Coefficient (standardised / one-hot inputs)",
                 config.FIGURES_DIR / "logreg_coefficients.png",
                 color=["#C44E52" if v > 0 else "#4C72B0" for v in top.iloc[::-1]])

    rf = searches["Random Forest"].best_estimator_
    names = [clean_feature_name(n) for n in rf.named_steps["preprocess"].get_feature_names_out()]
    rf_imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
    rf_imp = rf_imp.sort_values(ascending=False)
    explain["rf_impurity_importance"] = rf_imp.to_dict()
    ev.plot_barh(rf_imp.head(15), "Random Forest: impurity importance (top 15)",
                 "Mean decrease in impurity", config.FIGURES_DIR / "rf_importance.png")

    perm = permutation_importance(final_pipe, X_test, y_test, scoring="roc_auc",
                                  n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    explain["permutation_importance"] = {
        c: {"mean": float(perm.importances_mean[i]), "std": float(perm.importances_std[i])}
        for i, c in enumerate(X_test.columns)
    }
    ev.plot_barh(perm_s, f"{final_name}: permutation importance (test set)",
                 "Drop in ROC-AUC when the column is shuffled",
                 config.FIGURES_DIR / "permutation_importance.png")

    # 8. Save
    joblib.dump(final_pipe, config.MODEL_PATH, compress=3)
    size_kb = config.MODEL_PATH.stat().st_size / 1024
    metadata = {
        "model_name": final_name,
        "best_params": cv_summary[final_name]["best_params"],
        "engineered_features": engineered,
        "threshold": threshold,
        "high_risk_from": high_cut,
        "risk_bands": {
            "LOW": f"probability < {threshold:.2f}",
            "MEDIUM": f"{threshold:.2f} <= probability < {high_cut:.2f}",
            "HIGH": f"probability >= {high_cut:.2f}",
        },
        "oof_band_stats": bands,
        "test_metrics_tuned": final_tuned,
        "test_metrics_default": test_metrics[final_name],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": config.DATASET_SHA256,
        "input_schema": {"categorical": config.CATEGORICAL_VALUES, "numeric": config.NUMERIC_COLS},
        # Mean of each transformed feature on the training split. For a linear
        # model this is the reference point for per-customer explanations.
        "train_feature_means": dict(zip(
            final_pipe.named_steps["preprocess"].get_feature_names_out().tolist(),
            final_pipe[:-1].transform(X_train).mean(axis=0).tolist(),
        )),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics = {
        "cleaning_report": report.steps,
        "feature_ablation": {"per_model": ablation, "mean_delta": mean_delta, "kept": engineered},
        "cv": cv_summary,
        "selection_reason": selection_reason,
        "threshold": {"value": threshold, "high_risk_from": high_cut,
                      "oof_metrics_at_threshold": oof_metrics, "oof_band_stats": bands},
        "test": test_metrics,
        "test_final_tuned": final_tuned,
        "test_band_stats": ev.band_stats(y_test, test_probs[final_name], threshold, high_cut),
        "explainability": explain,
        "dataset_summary": dataset_summary(df),
        "curves": ev.curve_data(y_train, oof, y_test, test_probs),
        "model_file_kb": round(size_kb, 1),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_comparison_md(cv_summary, test_metrics, final_name, final_tuned, selection_reason)
    print(f"Saved {config.MODEL_PATH.name} ({size_kb:.1f} KB), metadata, metrics, figures. "
          f"Total {time.time() - t0:.0f}s")


def write_comparison_md(cv_summary, test_metrics, final_name, final_tuned, reason) -> None:
    lines = [
        "# Model comparison",
        "",
        "Generated by `python -m src.train`. CV = 5-fold stratified on the training split; "
        "test metrics on the held-out 20% at threshold 0.5 unless noted.",
        "",
        "| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, m in test_metrics.items():
        c = cv_summary[n]
        lines.append(
            f"| {n} | {c['cv_roc_auc_mean']:.3f} ± {c['cv_roc_auc_std']:.3f} | {m['accuracy']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} |"
        )
    t = final_tuned
    lines += [
        f"| **{final_name} @ {t['threshold']:.2f} (final)** | | {t['accuracy']:.3f} | "
        f"{t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} | {t['roc_auc']:.3f} |",
        "",
        f"**Selection:** {reason}",
        "",
    ]
    (config.REPORTS_DIR / "model_comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
