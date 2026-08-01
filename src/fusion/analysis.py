from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_classification_metrics
from src.fusion.actions import FUSION_ACTIONS
from src.utils.file_io import write_json


def analyze_rl_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Analyze action use, weight behavior and oracle action upper bound."""
    paths = config["paths"]
    outputs_df = pd.read_csv(paths["test_outputs"])
    predictions_df = pd.read_csv(paths["predictions_path"])
    df = outputs_df.merge(predictions_df, on=["sample_id", "label"], how="inner")

    action_details = _action_detail_frame(df)
    action_details.to_csv(paths["action_details_path"], index=False)

    oracle = _oracle_action_predictions(outputs_df)
    oracle_metrics = binary_classification_metrics(
        outputs_df["label"].to_numpy(), oracle["oracle_probability"].to_numpy()
    )

    analysis = {
        "rows": int(len(df)),
        "action_distribution": _value_counts(df["selected_action"]),
        "weight_summary": {
            "average_image_weight": float(df["image_weight"].mean()),
            "average_text_weight": float(df["text_weight"].mean()),
            "image_weight_std": float(df["image_weight"].std(ddof=0)),
            "text_weight_std": float(df["text_weight"].std(ddof=0)),
        },
        "by_label": _group_weight_summary(df, "label"),
        "by_image_quality_group": _quality_group_summary(df, "image_quality"),
        "by_text_quality_group": _quality_group_summary(df, "text_quality"),
        "agreement": _agreement_summary(df),
        "oracle_action_upper_bound": {
            "metrics": oracle_metrics,
            "action_distribution": _value_counts(oracle["oracle_action"]),
        },
        "details_path": str(paths["action_details_path"]),
    }
    write_json(analysis, paths["analysis_path"])
    return analysis


def _action_detail_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "sample_id",
            "label",
            "image_probability",
            "text_probability",
            "image_quality",
            "text_quality",
            "selected_action",
            "image_weight",
            "text_weight",
            "final_probability",
            "final_prediction",
        ]
    ].copy()


def _oracle_action_predictions(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["label"].to_numpy(dtype=int)
    image_prob = df["image_probability"].to_numpy(dtype=float)
    text_prob = df["text_probability"].to_numpy(dtype=float)
    best_actions = []
    best_probabilities = []
    for row_index, label in enumerate(labels):
        row_probs = []
        row_correct = []
        for image_weight, text_weight in FUSION_ACTIONS:
            probability = image_weight * image_prob[row_index] + text_weight * text_prob[row_index]
            prediction = int(probability >= 0.5)
            row_probs.append(probability)
            row_correct.append(prediction == label)
        if any(row_correct):
            candidate_indices = [index for index, correct in enumerate(row_correct) if correct]
            action = min(candidate_indices, key=lambda index: abs(row_probs[index] - 0.5))
        else:
            action = min(range(len(row_probs)), key=lambda index: abs(row_probs[index] - label))
        best_actions.append(action)
        best_probabilities.append(row_probs[action])
    return pd.DataFrame({"oracle_action": best_actions, "oracle_probability": best_probabilities})


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts().sort_index()
    return {str(int(index)): int(value) for index, value in counts.items()}


def _group_weight_summary(df: pd.DataFrame, group_col: str) -> dict[str, dict[str, float]]:
    summary = {}
    for group_value, group in df.groupby(group_col, observed=True):
        if group.empty:
            continue
        summary[str(group_value)] = {
            "rows": int(len(group)),
            "average_image_weight": float(group["image_weight"].mean()),
            "average_text_weight": float(group["text_weight"].mean()),
            "macro_f1": float(
                binary_classification_metrics(group["label"].to_numpy(), group["final_probability"].to_numpy())["macro_f1"]
            ),
        }
    return summary


def _quality_group_summary(df: pd.DataFrame, quality_col: str) -> dict[str, dict[str, float]]:
    work = df.copy()
    work["quality_group"] = pd.cut(
        work[quality_col],
        bins=[-0.001, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"],
    )
    return _group_weight_summary(work.dropna(subset=["quality_group"]), "quality_group")


def _agreement_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    work = df.copy()
    work["modalities_agree"] = work["image_prediction"] == work["text_prediction"]
    return _group_weight_summary(work, "modalities_agree")

