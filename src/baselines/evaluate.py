from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.baselines.methods import add_baseline_probabilities
from src.evaluation.metrics import binary_classification_metrics
from src.utils.file_io import write_json


SPLITS = ["train", "validation", "test"]
METHODS = [
    "image_only",
    "text_only",
    "equal_fusion",
    "confidence_weighted_fusion",
    "reliability_weighted_fusion",
]


def evaluate_baselines(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate deterministic fusion baselines on saved modality outputs."""
    paths = config["paths"]
    split_paths = {
        "train": paths["train_outputs"],
        "validation": paths["validation_outputs"],
        "test": paths["test_outputs"],
    }
    rows = []
    summary: dict[str, Any] = {"splits": {}}
    for split, split_path in split_paths.items():
        df = pd.read_csv(split_path)
        probabilities = add_baseline_probabilities(df)
        summary["splits"][split] = {}
        for method in METHODS:
            metrics = binary_classification_metrics(probabilities["label"].to_numpy(), probabilities[method].to_numpy())
            summary["splits"][split][method] = metrics
            rows.append({"split": split, "method": method, **_flatten_metrics(metrics)})

    results_df = pd.DataFrame(rows)
    results_path = Path(paths["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    write_json(summary, paths["summary_path"])
    return summary


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened = {key: value for key, value in metrics.items() if key != "confusion_matrix"}
    flattened["tn"] = metrics["confusion_matrix"][0][0]
    flattened["fp"] = metrics["confusion_matrix"][0][1]
    flattened["fn"] = metrics["confusion_matrix"][1][0]
    flattened["tp"] = metrics["confusion_matrix"][1][1]
    return flattened
