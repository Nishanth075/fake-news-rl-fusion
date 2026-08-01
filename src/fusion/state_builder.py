from __future__ import annotations

import numpy as np
import pandas as pd

STATE_COLUMNS = [
    "image_probability",
    "image_confidence",
    "image_quality",
    "text_probability",
    "text_confidence",
    "text_quality",
    "disagreement",
    "confidence_difference",
    "quality_difference",
]


def build_state_frame(df: pd.DataFrame) -> pd.DataFrame:
    state_df = pd.DataFrame(
        {
            "image_probability": df["image_probability"],
            "image_confidence": df["image_confidence"],
            "image_quality": df["image_quality"],
            "text_probability": df["text_probability"],
            "text_confidence": df["text_confidence"],
            "text_quality": df["text_quality"],
            "disagreement": (df["image_probability"] - df["text_probability"]).abs(),
            "confidence_difference": df["image_confidence"] - df["text_confidence"],
            "quality_difference": df["image_quality"] - df["text_quality"],
        }
    )
    return state_df.clip(-1.0, 1.0)


def build_states(df: pd.DataFrame) -> np.ndarray:
    return build_state_frame(df)[STATE_COLUMNS].to_numpy(dtype=np.float32)
