"""Metrics, threshold selection and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Classification metrics at a given probability threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_curve(y_true, y_prob) -> pd.DataFrame:
    """Precision / recall / F1 / F2 across the threshold grid."""
    rows = []
    for t in THRESHOLD_GRID:
        y_pred = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def best_f2_threshold(y_true, y_prob) -> float:
    curve = threshold_curve(y_true, y_prob)
    return float(curve.loc[curve["f2"].idxmax(), "threshold"])


def band_stats(y_true, y_prob, medium_from: float, high_from: float) -> dict:
    """Share of customers and observed churn rate in each risk band."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    masks = {
        "LOW": y_prob < medium_from,
        "MEDIUM": (y_prob >= medium_from) & (y_prob < high_from),
        "HIGH": y_prob >= high_from,
    }
    return {
        band: {"share": float(m.mean()), "churn_rate": float(y_true[m].mean()) if m.any() else None}
        for band, m in masks.items()
    }


# ---------------------------------------------------------------- plots

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc_curves(y_true, probs: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_confusion(cm: dict, title: str, path: Path) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > mat.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Stay", "Churn"])
    ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    _save(fig, path)


def plot_threshold_curve(curve: pd.DataFrame, chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for col in ["precision", "recall", "f1", "f2"]:
        ax.plot(curve["threshold"], curve[col], label=col.upper() if col.startswith("f") else col)
    ax.axvline(chosen, c="black", ls="--", lw=1, label=f"chosen = {chosen:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Score",
           title="Threshold tuning (out-of-fold, training data)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_calibration(y_true, y_prob, title: str, path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title=title)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_barh(series: pd.Series, title: str, xlabel: str, path: Path, color=None) -> None:
    series = series.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(series) + 1.2))
    ax.barh(series.index, series.values, color=color if color is not None else "#4C72B0")
    ax.axvline(0, c="grey", lw=0.8)
    ax.set(title=title, xlabel=xlabel)
    ax.tick_params(axis="y", labelsize=8)
    _save(fig, path)
