from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.threshold_tuning import run_threshold_tuning, tune_threshold


def test_tune_threshold_selects_validation_macro_f1() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])

    selected = tune_threshold(labels, probabilities, np.array([0.3, 0.5, 0.7]))

    assert selected["threshold"] == 0.5
    assert selected["macro_f1"] == 1.0


def test_run_threshold_tuning_writes_summary(tmp_path) -> None:
    validation = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "image_probability": [0.2, 0.4, 0.6, 0.8],
            "image_confidence": [0.8, 0.6, 0.6, 0.8],
            "image_quality": [0.8, 0.8, 0.8, 0.8],
            "text_probability": [0.1, 0.3, 0.7, 0.9],
            "text_confidence": [0.9, 0.7, 0.7, 0.9],
            "text_quality": [0.9, 0.9, 0.9, 0.9],
        }
    )
    validation.to_csv(tmp_path / "validation.csv", index=False)
    validation.to_csv(tmp_path / "test.csv", index=False)

    result = run_threshold_tuning(
        {
            "threshold_tuning": {"min_threshold": 0.3, "max_threshold": 0.7, "num_thresholds": 5},
            "paths": {
                "validation_outputs": str(tmp_path / "validation.csv"),
                "test_outputs": str(tmp_path / "test.csv"),
                "summary_path": str(tmp_path / "summary.csv"),
                "metrics_path": str(tmp_path / "metrics.json"),
            },
        }
    )
    summary = pd.read_csv(tmp_path / "summary.csv")

    assert result["summary_path"] == str(tmp_path / "summary.csv")
    assert set(summary["method"]) >= {"equal_fusion", "image_only", "text_only"}
