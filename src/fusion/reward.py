from __future__ import annotations

import numpy as np


def fused_probability(
    image_probability: np.ndarray,
    text_probability: np.ndarray,
    image_weight: np.ndarray,
    text_weight: np.ndarray,
) -> np.ndarray:
    return image_weight * image_probability + text_weight * text_probability


def reward_from_predictions(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.where(predictions.astype(int) == labels.astype(int), 1.0, -1.0).astype(np.float32)
