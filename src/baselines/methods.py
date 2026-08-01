from __future__ import annotations

import pandas as pd

EPSILON = 1e-8


def add_baseline_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-sample probabilities for all baseline methods."""
    output = pd.DataFrame({"sample_id": df["sample_id"], "label": df["label"]})
    output["image_only"] = df["image_probability"]
    output["text_only"] = df["text_probability"]
    output["equal_fusion"] = 0.5 * df["image_probability"] + 0.5 * df["text_probability"]

    confidence_sum = df["image_confidence"] + df["text_confidence"] + EPSILON
    image_conf_weight = df["image_confidence"] / confidence_sum
    text_conf_weight = df["text_confidence"] / confidence_sum
    output["confidence_weighted_fusion"] = (
        image_conf_weight * df["image_probability"] + text_conf_weight * df["text_probability"]
    )

    image_reliability = df["image_confidence"] * df["image_quality"]
    text_reliability = df["text_confidence"] * df["text_quality"]
    reliability_sum = image_reliability + text_reliability + EPSILON
    image_rel_weight = image_reliability / reliability_sum
    text_rel_weight = text_reliability / reliability_sum
    output["reliability_weighted_fusion"] = (
        image_rel_weight * df["image_probability"] + text_rel_weight * df["text_probability"]
    )
    return output
