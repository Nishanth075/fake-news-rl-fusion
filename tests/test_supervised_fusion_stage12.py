from __future__ import annotations

import pandas as pd

from src.fusion.supervised import train_supervised_fusion


def test_supervised_fusion_trains_and_writes_predictions(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
            "image_probability": [0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6],
            "image_confidence": [0.9, 0.8, 0.8, 0.9, 0.7, 0.7, 0.6, 0.6],
            "image_quality": [0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5],
            "text_probability": [0.2, 0.9, 0.1, 0.8, 0.2, 0.9, 0.3, 0.7],
            "text_confidence": [0.8, 0.9, 0.9, 0.8, 0.8, 0.9, 0.7, 0.7],
            "text_quality": [0.9] * 8,
        }
    )
    for split_name in ["train", "validation", "test"]:
        rows.to_csv(tmp_path / f"{split_name}.csv", index=False)

    config = {
        "seed": 42,
        "supervised_fusion": {
            "state_dim": 9,
            "hidden_dims": [8],
            "dropout": 0.0,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 4,
            "epochs": 2,
        },
        "paths": {
            "train_outputs": str(tmp_path / "train.csv"),
            "validation_outputs": str(tmp_path / "validation.csv"),
            "test_outputs": str(tmp_path / "test.csv"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "metrics_path": str(tmp_path / "metrics.json"),
            "test_predictions_path": str(tmp_path / "predictions.csv"),
        },
    }

    result = train_supervised_fusion(config)
    predictions = pd.read_csv(tmp_path / "predictions.csv")

    assert result["test"]["rows"] == 8
    assert set(predictions.columns) == {"sample_id", "label", "final_probability", "final_prediction"}
