from __future__ import annotations

import pandas as pd

from src.evaluation.headline_significance import run_headline_significance


def test_headline_significance_compares_rl_to_equal_fusion(tmp_path) -> None:
    test_outputs = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "image_probability": [0.2, 0.7, 0.6, 0.8],
            "image_confidence": [0.8, 0.8, 0.7, 0.7],
            "image_quality": [0.9, 0.9, 0.9, 0.9],
            "text_probability": [0.1, 0.6, 0.7, 0.9],
            "text_confidence": [0.9, 0.9, 0.8, 0.8],
            "text_quality": [0.9, 0.9, 0.9, 0.9],
        }
    )
    rl_predictions = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "final_probability": [0.2, 0.4, 0.6, 0.9],
            "final_prediction": [0, 0, 1, 1],
        }
    )
    test_outputs.to_csv(tmp_path / "test_outputs.csv", index=False)
    rl_predictions.to_csv(tmp_path / "rl.csv", index=False)

    result = run_headline_significance(
        {
            "headline_significance": {
                "baseline_method": "equal_fusion",
                "test_outputs": str(tmp_path / "test_outputs.csv"),
                "rl_predictions": str(tmp_path / "rl.csv"),
                "output_path": str(tmp_path / "headline.json"),
                "summary_path": str(tmp_path / "headline.csv"),
            }
        }
    )

    assert result["rows"] == 4
    assert result["comparison"] == "rl_adaptive_fusion_vs_equal_fusion"
    assert "mcnemar" in result
    assert (tmp_path / "headline.csv").exists()
