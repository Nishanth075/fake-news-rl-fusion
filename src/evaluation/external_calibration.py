from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils.file_io import write_json
from src.utils.seed import set_seed


def run_external_calibration(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate threshold calibration on an external test-only dataset."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]
    calibration_fraction = float(config["calibration"].get("fraction", 0.2))

    base = pd.read_csv(paths["modality_outputs"])
    rl = pd.read_csv(paths["rl_predictions"])
    df = base.merge(rl[["sample_id", "final_probability"]], on="sample_id", how="inner")

    calibration_df, holdout_df = train_test_split(
        df,
        test_size=1.0 - calibration_fraction,
        random_state=seed,
        stratify=df["label"],
    )

    rows = []
    method_probabilities = _method_probabilities(df)
    for method, probabilities in method_probabilities.items():
        probs = pd.Series(probabilities, index=df.index)
        calibration_probs = probs.loc[calibration_df.index].to_numpy()
        holdout_probs = probs.loc[holdout_df.index].to_numpy()
        threshold, calibration_macro_f1 = _select_threshold(
            calibration_df["label"].to_numpy(), calibration_probs
        )
        rows.append(
            {
                "method": method,
                "threshold_selected_on_20pct_calibration": threshold,
                "holdout_macro_f1": _macro_f1(holdout_df["label"].to_numpy(), holdout_probs, threshold),
                "holdout_accuracy": _accuracy(holdout_df["label"].to_numpy(), holdout_probs, threshold),
                "holdout_roc_auc": _roc_auc(holdout_df["label"].to_numpy(), holdout_probs),
                "holdout_confusion_matrix": confusion_matrix(
                    holdout_df["label"].to_numpy(),
                    (holdout_probs >= threshold).astype(int),
                    labels=[0, 1],
                ).astype(int).tolist(),
                "calibration_macro_f1": calibration_macro_f1,
            }
        )

    results = pd.DataFrame(rows).sort_values("holdout_macro_f1", ascending=False)
    summary_path = Path(paths["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(summary_path, index=False)

    summary = {
        "dataset": config.get("dataset", "OpenFake reddit/test"),
        "rows": int(len(df)),
        "calibration_rows": int(len(calibration_df)),
        "holdout_rows": int(len(holdout_df)),
        "calibration_fraction": calibration_fraction,
        "selection_rule": "Threshold selected on the 20 percent external calibration subset and applied unchanged to the 80 percent holdout subset.",
        "summary_path": str(summary_path),
        "rows_detail": results.to_dict(orient="records"),
    }
    write_json(summary, paths["metrics_path"])
    return summary


def _method_probabilities(df: pd.DataFrame) -> dict[str, np.ndarray]:
    image_probability = df["image_probability"].to_numpy(dtype=float)
    text_probability = df["text_probability"].to_numpy(dtype=float)
    image_confidence = df["image_confidence"].to_numpy(dtype=float)
    text_confidence = df["text_confidence"].to_numpy(dtype=float)
    image_quality = df["image_quality"].to_numpy(dtype=float)
    text_quality = df["text_quality"].to_numpy(dtype=float)

    confidence_denominator = image_confidence + text_confidence
    confidence_image_weight = np.divide(
        image_confidence,
        confidence_denominator,
        out=np.full_like(confidence_denominator, 0.5),
        where=confidence_denominator > 0,
    )
    quality_denominator = image_quality + text_quality
    reliability_image_weight = np.divide(
        image_quality,
        quality_denominator,
        out=np.full_like(quality_denominator, 0.5),
        where=quality_denominator > 0,
    )
    return {
        "image_only": image_probability,
        "text_only": text_probability,
        "equal_fusion": (image_probability + text_probability) / 2.0,
        "confidence_weighted_fusion": confidence_image_weight * image_probability
        + (1.0 - confidence_image_weight) * text_probability,
        "reliability_weighted_fusion": reliability_image_weight * image_probability
        + (1.0 - reliability_image_weight) * text_probability,
        "rl_adaptive_fusion": df["final_probability"].to_numpy(dtype=float),
    }


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_macro_f1 = -1.0
    for threshold in np.linspace(0.001, 0.999, 999):
        macro_f1 = _macro_f1(labels, probabilities, float(threshold))
        if macro_f1 > best_macro_f1:
            best_threshold = float(threshold)
            best_macro_f1 = macro_f1
    return best_threshold, best_macro_f1


def _macro_f1(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    return float(f1_score(labels, (probabilities >= threshold).astype(int), average="macro", zero_division=0))


def _accuracy(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    return float(accuracy_score(labels, (probabilities >= threshold).astype(int)))


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(labels, probabilities))
    except ValueError:
        return None
