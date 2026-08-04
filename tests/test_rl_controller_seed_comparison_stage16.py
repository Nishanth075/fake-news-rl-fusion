from __future__ import annotations

import pandas as pd

from src.evaluation.rl_controller_seed_comparison import run_rl_controller_seed_comparison


def test_rl_controller_seed_comparison_uses_best_validation_controller(tmp_path) -> None:
    controller_summary = pd.DataFrame(
        [
            {
                "method": "tree_a",
                "validation_macro_f1": 0.7,
                "test_macro_f1": 0.6,
                "test_accuracy": 0.6,
                "test_roc_auc": 0.7,
                "test_predictions_path": str(tmp_path / "tree_a.csv"),
            },
            {
                "method": "tree_b",
                "validation_macro_f1": 0.8,
                "test_macro_f1": 0.75,
                "test_accuracy": 0.75,
                "test_roc_auc": 0.8,
                "test_predictions_path": str(tmp_path / "tree_b.csv"),
            },
        ]
    )
    controller_predictions = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "final_probability": [0.1, 0.4, 0.7, 0.8],
            "final_prediction": [0, 0, 1, 1],
        }
    )
    rl_predictions = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "final_probability": [0.2, 0.6, 0.7, 0.8],
            "final_prediction": [0, 1, 1, 1],
        }
    )
    controller_summary.to_csv(tmp_path / "summary.csv", index=False)
    controller_predictions.to_csv(tmp_path / "tree_a.csv", index=False)
    controller_predictions.to_csv(tmp_path / "tree_b.csv", index=False)
    rl_predictions.to_csv(tmp_path / "rl_seed_42_test_predictions.csv", index=False)

    result = run_rl_controller_seed_comparison(
        {
            "rl_controller_seed_comparison": {
                "seeds": [42],
                "controller_summary": str(tmp_path / "summary.csv"),
                "rl_predictions_pattern": str(tmp_path / "rl_seed_{seed}_test_predictions.csv"),
                "summary_path": str(tmp_path / "out.csv"),
                "details_path": str(tmp_path / "out.json"),
            }
        }
    )

    assert result["controller_method"] == "tree_b"
    assert result["aggregate"]["num_seeds"] == 1
    assert (tmp_path / "out.csv").exists()
