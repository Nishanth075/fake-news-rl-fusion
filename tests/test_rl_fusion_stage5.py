from __future__ import annotations

import pandas as pd

from src.fusion.actions import FUSION_ACTIONS, get_action_weights, resolve_fusion_actions
from src.fusion.reward import reward_from_predictions
from src.fusion.state_builder import build_states
from src.fusion.train import train_rl_fusion


def test_fusion_actions_sum_to_one() -> None:
    assert len(FUSION_ACTIONS) == 7
    for image_weight, text_weight in FUSION_ACTIONS:
        assert abs((image_weight + text_weight) - 1.0) < 1e-8
    assert get_action_weights(3) == (0.50, 0.50)




def test_custom_fusion_actions_from_config() -> None:
    actions = resolve_fusion_actions({"fusion": {"action_weights": [[0.8, 0.2], [0.2, 0.8]]}})

    assert actions == [(0.8, 0.2), (0.2, 0.8)]
    assert get_action_weights(1, actions) == (0.2, 0.8)

def test_state_builder_has_nine_features() -> None:
    df = _fusion_df()
    states = build_states(df)
    assert states.shape == (4, 9)


def test_reward_from_predictions() -> None:
    rewards = reward_from_predictions(
        predictions=__import__("numpy").array([0, 1, 0]),
        labels=__import__("numpy").array([0, 0, 0]),
    )
    assert rewards.tolist() == [1.0, -1.0, 1.0]


def test_train_rl_fusion_tiny_run(tmp_path) -> None:
    outputs_dir = tmp_path / "outputs"
    checkpoint_dir = tmp_path / "checkpoints"
    metrics_dir = tmp_path / "metrics"
    outputs_dir.mkdir()
    metrics_dir.mkdir()
    df = _fusion_df()
    for split in ["train", "validation", "test"]:
        df.to_csv(outputs_dir / f"{split}_outputs.csv", index=False)

    result = train_rl_fusion(
        {
            "seed": 42,
            "fusion": {
                "state_dim": 9,
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
                "checkpoint_dir": str(checkpoint_dir),
                "metrics_path": str(metrics_dir / "rl.json"),
                "test_predictions_path": str(metrics_dir / "predictions.csv"),
            },
        }
    )

    assert result["test"]["rows"] == 4
    assert (checkpoint_dir / "best_fusion_q_network.pt").exists()
    assert (metrics_dir / "predictions.csv").exists()


def _fusion_df() -> pd.DataFrame:
    return pd.DataFrame(
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

