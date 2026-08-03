from __future__ import annotations

import numpy as np

from src.evaluation.seed_significance import mcnemar_exact


def test_mcnemar_exact_counts_discordant_pairs() -> None:
    labels = np.array([0, 1, 0, 1, 1, 0])
    model_a = np.array([0, 1, 1, 1, 0, 0])
    model_b = np.array([1, 1, 0, 0, 0, 0])

    result = mcnemar_exact(labels, model_a, model_b)

    assert result["b"] == 2
    assert result["c"] == 1
    assert result["discordant"] == 3
    assert 0 <= result["p_value"] <= 1


def test_mcnemar_exact_handles_no_disagreement() -> None:
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 0, 1])

    result = mcnemar_exact(labels, predictions, predictions)

    assert result["discordant"] == 0
    assert result["p_value"] == 1.0
