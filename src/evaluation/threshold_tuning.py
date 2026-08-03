from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.methods import add_baseline_probabilities
from src.evaluation.metrics import binary_classification_metrics
from src.utils.file_io import write_json


BASELINE_METHODS = [
    "image_only",
    "text_only",
    "equal_fusion",
    "confidence_weighted_fusion",
    "reliability_weighted_fusion",
]


def run_threshold_tuning(config: dict[str, Any]) -> dict[str, Any]:
    """Tune decision thresholds on validation and evaluate once on test."""
    paths = config["paths"]
    tuning_config = config.get("threshold_tuning", {})
    thresholds = np.linspace(
        float(tuning_config.get("min_threshold", 0.05)),
        float(tuning_config.get("max_threshold", 0.95)),
        int(tuning_config.get("num_thresholds", 181)),
    )

    rows = []
    rows.extend(_baseline_rows(paths, thresholds))
    rows.extend(_prediction_file_rows(paths, thresholds))

    summary_df = pd.DataFrame(rows)
    summary_path = Path(paths["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    result = {
        "rows": summary_df.to_dict(orient="records"),
        "summary_path": str(summary_path),
        "selection_rule": "threshold selected on validation macro_f1 and applied unchanged to test",
    }
    write_json(result, paths["metrics_path"])
    return result


def tune_threshold(labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    """Select the threshold that maximizes validation macro F1."""
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        metrics = binary_classification_metrics(labels, probabilities, threshold=float(threshold))
        record = {
            "threshold": float(threshold),
            "macro_f1": float(metrics["macro_f1"]),
            "accuracy": float(metrics["accuracy"]),
            "roc_auc": metrics.get("roc_auc"),
        }
        if best is None or record["macro_f1"] > best["macro_f1"]:
            best = record
    if best is None:
        raise ValueError("No thresholds were provided.")
    return best


def _baseline_rows(paths: dict[str, str], thresholds: np.ndarray) -> list[dict[str, Any]]:
    validation_df = pd.read_csv(paths["validation_outputs"])
    test_df = pd.read_csv(paths["test_outputs"])
    validation_probabilities = add_baseline_probabilities(validation_df)
    test_probabilities = add_baseline_probabilities(test_df)

    rows = []
    for method in BASELINE_METHODS:
        selected = tune_threshold(
            validation_probabilities["label"].to_numpy(),
            validation_probabilities[method].to_numpy(),
            thresholds,
        )
        test_metrics = binary_classification_metrics(
            test_probabilities["label"].to_numpy(),
            test_probabilities[method].to_numpy(),
            threshold=float(selected["threshold"]),
        )
        default_metrics = binary_classification_metrics(
            test_probabilities["label"].to_numpy(),
            test_probabilities[method].to_numpy(),
            threshold=0.5,
        )
        rows.append(
            {
                "method": method,
                "source": "baseline",
                "selected_threshold": selected["threshold"],
                "validation_macro_f1_at_selected_threshold": selected["macro_f1"],
                "test_macro_f1_at_0_5": default_metrics["macro_f1"],
                "test_macro_f1_tuned": test_metrics["macro_f1"],
                "test_accuracy_tuned": test_metrics["accuracy"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
        )
    return rows


def _prediction_file_rows(paths: dict[str, str], thresholds: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    prediction_specs = [
        ("rl_adaptive_fusion", "rl_validation_predictions", "rl_test_predictions", "rl"),
        (
            "supervised_mlp_fusion",
            "supervised_validation_predictions",
            "supervised_test_predictions",
            "supervised_fusion",
        ),
    ]
    for method, validation_key, test_key, source in prediction_specs:
        validation_path = paths.get(validation_key)
        test_path = paths.get(test_key)
        if not validation_path or not test_path:
            continue
        if not Path(validation_path).exists() or not Path(test_path).exists():
            continue

        validation_df = pd.read_csv(validation_path)
        test_df = pd.read_csv(test_path)
        selected = tune_threshold(
            validation_df["label"].to_numpy(),
            validation_df["final_probability"].to_numpy(),
            thresholds,
        )
        test_metrics = binary_classification_metrics(
            test_df["label"].to_numpy(),
            test_df["final_probability"].to_numpy(),
            threshold=float(selected["threshold"]),
        )
        default_metrics = binary_classification_metrics(
            test_df["label"].to_numpy(),
            test_df["final_probability"].to_numpy(),
            threshold=0.5,
        )
        rows.append(
            {
                "method": method,
                "source": source,
                "selected_threshold": selected["threshold"],
                "validation_macro_f1_at_selected_threshold": selected["macro_f1"],
                "test_macro_f1_at_0_5": default_metrics["macro_f1"],
                "test_macro_f1_tuned": test_metrics["macro_f1"],
                "test_accuracy_tuned": test_metrics["accuracy"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
        )
    return rows
