from __future__ import annotations

import pandas as pd
import torch

from src.fusion.q_network import FusionQNetwork
from src.fusion.robustness import run_robustness


def test_run_robustness_writes_summary(tmp_path) -> None:
    outputs_path = tmp_path / "test_outputs.csv"
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "robustness"
    checkpoint_dir.mkdir()
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
    df.to_csv(outputs_path, index=False)
    model = FusionQNetwork(state_dim=9, action_dim=7, dropout=0.0)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_dir / "best_fusion_q_network.pt")

    result = run_robustness(
        {
            "seed": 42,
            "paths": {
                "test_outputs": str(outputs_path),
                "checkpoint_dir": str(checkpoint_dir),
                "output_dir": str(output_dir),
                "summary_path": str(output_dir / "summary.csv"),
                "details_path": str(output_dir / "details.json"),
            },
            "fusion": {"state_dim": 9, "action_dim": 7, "dropout": 0.0},
            "corruptions": {
                "image_quality_drop": {"modality": "image", "levels": [0.5]},
                "text_probability_noise": {"modality": "text", "levels": [0.1]},
            },
        }
    )

    summary = pd.read_csv(result["summary_path"])
    assert len(summary) == 2
    assert "delta_target_weight" in summary.columns
    assert (output_dir / "details.json").exists()
