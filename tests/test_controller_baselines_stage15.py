from __future__ import annotations

import pandas as pd

from src.evaluation.controller_baselines import run_controller_baselines


def test_controller_baselines_train_on_reliability_state(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "image_probability": [0.1, 0.2, 0.2, 0.3, 0.1, 0.4, 0.7, 0.8, 0.9, 0.6, 0.7, 0.9],
            "image_confidence": [0.8] * 12,
            "image_quality": [0.7] * 12,
            "text_probability": [0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.8, 0.7, 0.9, 0.8, 0.6, 0.9],
            "text_confidence": [0.9] * 12,
            "text_quality": [0.8] * 12,
        }
    )
    for split in ["train", "validation", "test"]:
        df.to_csv(tmp_path / f"{split}.csv", index=False)

    result = run_controller_baselines(
        {
            "seed": 42,
            "controller_baselines": {"tree_depths": [2], "include_unlimited_tree": False},
            "paths": {
                "train_outputs": str(tmp_path / "train.csv"),
                "validation_outputs": str(tmp_path / "validation.csv"),
                "test_outputs": str(tmp_path / "test.csv"),
                "predictions_dir": str(tmp_path / "predictions"),
                "summary_path": str(tmp_path / "summary.csv"),
                "details_path": str(tmp_path / "details.json"),
            },
        }
    )
    summary = pd.read_csv(tmp_path / "summary.csv")

    assert result["summary_path"] == str(tmp_path / "summary.csv")
    assert {"state_logistic_regression", "state_decision_tree_depth_2"} <= set(summary["method"])
    assert (tmp_path / "predictions" / "state_logistic_regression_test_predictions.csv").exists()
