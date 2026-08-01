from __future__ import annotations

import pandas as pd

from src.fusion.analysis import analyze_rl_policy


def test_analyze_rl_policy_writes_summary(tmp_path) -> None:
    outputs = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "label": [0, 1, 1],
            "image_probability": [0.2, 0.8, 0.4],
            "image_prediction": [0, 1, 0],
            "image_confidence": [0.8, 0.8, 0.6],
            "image_quality": [0.9, 0.4, 0.2],
            "text_probability": [0.3, 0.7, 0.9],
            "text_prediction": [0, 1, 1],
            "text_confidence": [0.7, 0.7, 0.9],
            "text_quality": [0.8, 0.8, 0.9],
        }
    )
    predictions = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "label": [0, 1, 1],
            "selected_action": [3, 3, 6],
            "image_weight": [0.5, 0.5, 0.1],
            "text_weight": [0.5, 0.5, 0.9],
            "final_probability": [0.25, 0.75, 0.85],
            "final_prediction": [0, 1, 1],
        }
    )
    output_path = tmp_path / "outputs.csv"
    prediction_path = tmp_path / "predictions.csv"
    analysis_path = tmp_path / "analysis.json"
    details_path = tmp_path / "details.csv"
    outputs.to_csv(output_path, index=False)
    predictions.to_csv(prediction_path, index=False)

    analysis = analyze_rl_policy(
        {
            "paths": {
                "test_outputs": str(output_path),
                "predictions_path": str(prediction_path),
                "analysis_path": str(analysis_path),
                "action_details_path": str(details_path),
            }
        }
    )

    assert analysis["rows"] == 3
    assert "oracle_action_upper_bound" in analysis
    assert analysis_path.exists()
    assert details_path.exists()
