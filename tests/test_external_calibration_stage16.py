from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.external_calibration import _method_probabilities, _select_threshold


def test_external_calibration_selects_threshold() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, macro_f1 = _select_threshold(labels, probabilities)

    assert 0.2 < threshold <= 0.8
    assert macro_f1 == 1.0


def test_method_probability_keys() -> None:
    df = pd.DataFrame(
        {
            "image_probability": [0.2, 0.8],
            "text_probability": [0.3, 0.7],
            "image_confidence": [0.8, 0.8],
            "text_confidence": [0.7, 0.7],
            "image_quality": [0.6, 0.4],
            "text_quality": [0.4, 0.6],
            "final_probability": [0.25, 0.75],
        }
    )

    probabilities = _method_probabilities(df)

    assert set(probabilities) == {
        "image_only",
        "text_only",
        "equal_fusion",
        "confidence_weighted_fusion",
        "reliability_weighted_fusion",
        "rl_adaptive_fusion",
    }
