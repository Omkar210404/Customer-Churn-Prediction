"""Load the saved pipeline and score customers.

Usage:
    python -m src.predict --json '{"tenure": 2, "Contract": "Month-to-month", ...}'
    python -m src.predict --csv path/to/customers.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import ENGINEERED_CATEGORICAL
from src.preprocessing import validate_features


class ModelNotFoundError(FileNotFoundError):
    pass


def risk_band(probability: float, threshold: float, high_from: float) -> str:
    if probability >= high_from:
        return "HIGH"
    if probability >= threshold:
        return "MEDIUM"
    return "LOW"


@dataclass
class ChurnModel:
    pipeline: object
    metadata: dict

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def high_from(self) -> float:
        return float(self.metadata["high_risk_from"])

    def predict(self, records) -> pd.DataFrame:
        """Score one record (dict) or many (list of dicts / DataFrame).

        Returns churn_probability, risk_band and flagged (probability >= threshold).
        """
        X = validate_features(records)
        proba = self.pipeline.predict_proba(X)[:, 1]
        return pd.DataFrame(
            {
                "churn_probability": proba,
                "risk_band": [risk_band(p, self.threshold, self.high_from) for p in proba],
                "flagged": proba >= self.threshold,
            },
            index=X.index,
        )


    @property
    def is_linear(self) -> bool:
        return hasattr(self.pipeline.named_steps["model"], "coef_")

    def explain(self, record: dict) -> pd.Series:
        """Per-input contribution to one customer's churn log-odds, relative to
        the average training customer. Exact for linear models only.

        Positive = pushes the score towards churn. One-hot columns are summed back
        to their original input (e.g. all Contract_* columns -> Contract).
        """
        if not self.is_linear:
            raise NotImplementedError("Per-customer contributions are only exact for linear models.")
        X = validate_features(record).iloc[[0]]
        names = self.pipeline.named_steps["preprocess"].get_feature_names_out()
        z = self.pipeline[:-1].transform(X)[0]
        means = np.array([self.metadata["train_feature_means"][n] for n in names])
        coef = self.pipeline.named_steps["model"].coef_[0]
        contrib = pd.Series(coef * (z - means), index=[_input_name(n) for n in names])
        return contrib.groupby(level=0).sum().sort_values(key=np.abs, ascending=False)


def _input_name(transformed: str) -> str:
    """'cat__Contract_Two year' -> 'Contract'; 'num__tenure' -> 'tenure'."""
    kind, name = transformed.split("__", 1)
    if kind == "num":
        return name
    for col in sorted(config.CATEGORICAL_COLS + ENGINEERED_CATEGORICAL, key=len, reverse=True):
        if name.startswith(col + "_"):
            return col
    return name


def load_model(
    model_path: Path = config.MODEL_PATH, metadata_path: Path = config.METADATA_PATH
) -> ChurnModel:
    if not model_path.exists() or not metadata_path.exists():
        raise ModelNotFoundError(
            f"Trained model not found ({model_path.name} / {metadata_path.name}). "
            "Run `python -m src.train` first."
        )
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ChurnModel(pipeline=pipeline, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers with the saved churn model.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="one customer as a JSON object")
    src.add_argument("--csv", type=Path, help="CSV with one customer per row")
    args = parser.parse_args()

    try:
        model = load_model()
        if args.json:
            records = json.loads(args.json)
        else:
            records = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
        result = model.predict(records)
    except (ModelNotFoundError, ValueError) as exc:  # JSONDecodeError is a ValueError
        parser.exit(2, f"error: {exc}\n")
    pd.set_option("display.width", 120)
    print(f"Model: {model.metadata['model_name']}  |  threshold {model.threshold:.2f}  |  "
          f"HIGH from {model.high_from:.2f}")
    print(result.round(3).to_string())


if __name__ == "__main__":
    main()
