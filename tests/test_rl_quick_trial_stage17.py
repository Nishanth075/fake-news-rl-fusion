from __future__ import annotations

import json

import pandas as pd

from src.evaluation import rl_quick_trial
from src.evaluation.rl_quick_trial import run_rl_quick_trial


def test_rl_quick_trial_writes_required_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rl_quick_trial, "EXPECTED_EQUAL_F1", 1.0)
    monkeypatch.setattr(rl_quick_trial, "SEEDS", [42])
    monkeypatch.setattr(rl_quick_trial, "REWARD_VARIANTS", ["binary"])
    monkeypatch.setattr(rl_quick_trial, "THRESHOLDS", [0.5])

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = _quick_trial_df()
    for split in ["train", "validation", "test"]:
        frame = df.copy()
        frame["sample_id"] = [f"{split}_{sample_id}" for sample_id in frame["sample_id"]]
        frame.to_csv(data_dir / f"{split}_outputs.csv", index=False)

    metrics_path = tmp_path / "final_rl_fusion_metrics.json"
    metrics_path.write_text(
        json.dumps({"test": {"metrics": {"macro_f1": 0.9, "accuracy": 0.9, "roc_auc": 1.0}}}),
        encoding="utf-8",
    )

    output_dir = tmp_path / "rl_quick_trial"
    result = run_rl_quick_trial(
        {
            "paths": {
                "train_outputs": str(data_dir / "train_outputs.csv"),
                "validation_outputs": str(data_dir / "validation_outputs.csv"),
                "test_outputs": str(data_dir / "test_outputs.csv"),
                "current_rl_metrics": str(metrics_path),
                "output_dir": str(output_dir),
            },
            "checks": {"equal_f1_tolerance": 0.001},
            "discrete": {"batch_size": 2, "epochs": 1, "dropout": 0.0},
            "continuous": {"batch_size": 2, "epochs": 2, "early_stopping_patience": 1},
        }
    )

    assert result["verification"]["equal_fusion_test_macro_f1"] == 1.0
    assert (output_dir / "discrete_rl_trial.csv").exists()
    assert (output_dir / "continuous_controller_trial.csv").exists()
    assert (output_dir / "quick_comparison.csv").exists()
    assert (output_dir / "quick_conclusion.md").exists()


def _quick_trial_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "image_probability": [0.1, 0.9, 0.1, 0.9],
            "image_confidence": [0.9, 0.9, 0.9, 0.9],
            "image_quality": [0.8, 0.8, 0.8, 0.8],
            "text_probability": [0.1, 0.9, 0.1, 0.9],
            "text_confidence": [0.9, 0.9, 0.9, 0.9],
            "text_quality": [0.8, 0.8, 0.8, 0.8],
        }
    )
