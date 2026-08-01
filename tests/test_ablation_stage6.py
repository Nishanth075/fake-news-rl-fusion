from __future__ import annotations

import pandas as pd

from src.fusion.ablation import run_rl_ablation


def test_run_rl_ablation_writes_summary(tmp_path) -> None:
    outputs_dir = tmp_path / "outputs"
    ablation_dir = tmp_path / "ablations"
    outputs_dir.mkdir()
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "image_probability": [0.1, 0.8, 0.2, 0.7],
            "image_prediction": [0, 1, 0, 1],
            "image_confidence": [0.9, 0.8, 0.8, 0.7],
            "image_quality": [0.8, 0.9, 0.7, 0.6],
            "text_probability": [0.2, 0.9, 0.4, 0.6],
            "text_prediction": [0, 1, 0, 1],
            "text_confidence": [0.8, 0.9, 0.6, 0.6],
            "text_quality": [0.9, 0.8, 0.7, 0.8],
        }
    )
    for split in ["train", "validation", "test"]:
        df.to_csv(outputs_dir / f"{split}_outputs.csv", index=False)

    result = run_rl_ablation(
        {
            "seed": 42,
            "base_fusion": {
                "action_dim": 7,
                "dropout": 0.0,
                "learning_rate": 0.001,
                "batch_size": 2,
                "epochs": 2,
                "epsilon_start": 0.2,
                "epsilon_end": 0.1,
            },
            "paths": {
                "train_outputs": str(outputs_dir / "train_outputs.csv"),
                "validation_outputs": str(outputs_dir / "validation_outputs.csv"),
                "test_outputs": str(outputs_dir / "test_outputs.csv"),
                "output_dir": str(ablation_dir),
                "summary_path": str(tmp_path / "summary.csv"),
            },
            "ablations": [
                {"name": "probabilities_only", "features": ["image_probability", "text_probability"]},
                {
                    "name": "full",
                    "features": [
                        "image_probability",
                        "image_confidence",
                        "image_quality",
                        "text_probability",
                        "text_confidence",
                        "text_quality",
                        "disagreement",
                        "confidence_difference",
                        "quality_difference",
                    ],
                },
            ],
        }
    )

    summary = pd.read_csv(result["summary_path"])
    assert len(summary) == 2
    assert set(summary["state_dim"]) == {2, 9}
    assert (ablation_dir / "ablation_details.json").exists()
