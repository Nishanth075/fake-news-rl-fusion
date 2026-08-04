from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_classification_metrics
from src.evaluation.seed_significance import mcnemar_exact
from src.utils.file_io import write_json


def run_rl_controller_seed_comparison(config: dict[str, Any]) -> dict[str, Any]:
    """Compare multi-seed RL fusion outputs with the best same-state controller baseline."""
    comparison = config["rl_controller_seed_comparison"]
    controller_summary = pd.read_csv(comparison["controller_summary"])
    best_controller = controller_summary.sort_values("validation_macro_f1", ascending=False).iloc[0]
    controller_predictions = pd.read_csv(best_controller["test_predictions_path"])
    labels = controller_predictions["label"].to_numpy(dtype=int)
    controller_probs = controller_predictions["final_probability"].to_numpy(dtype=float)
    controller_pred = controller_predictions["final_prediction"].to_numpy(dtype=int)
    controller_metrics = binary_classification_metrics(labels, controller_probs)

    rows: list[dict[str, Any]] = []
    for seed in [int(seed) for seed in comparison.get("seeds", [42, 7, 13])]:
        rl_path = Path(str(comparison["rl_predictions_pattern"]).format(seed=seed))
        if not rl_path.exists():
            raise FileNotFoundError(f"Missing RL predictions for seed {seed}: {rl_path}")
        rl_df = pd.read_csv(rl_path)
        merged = controller_predictions[["sample_id", "label", "final_prediction"]].merge(
            rl_df[["sample_id", "final_probability", "final_prediction"]],
            on="sample_id",
            how="inner",
            suffixes=("_controller", "_rl"),
        )
        if len(merged) != len(controller_predictions):
            raise ValueError(f"RL predictions for seed {seed} do not cover all controller samples.")
        seed_labels = merged["label"].to_numpy(dtype=int)
        rl_probs = merged["final_probability"].to_numpy(dtype=float)
        rl_pred = merged["final_prediction_rl"].to_numpy(dtype=int)
        controller_pred_aligned = merged["final_prediction_controller"].to_numpy(dtype=int)
        rl_metrics = binary_classification_metrics(seed_labels, rl_probs)
        mcnemar = mcnemar_exact(seed_labels, rl_pred, controller_pred_aligned)
        rows.append(
            {
                "seed": seed,
                "controller_method": best_controller["method"],
                "controller_validation_macro_f1": float(best_controller["validation_macro_f1"]),
                "controller_test_macro_f1": controller_metrics["macro_f1"],
                "rl_test_macro_f1": rl_metrics["macro_f1"],
                "delta_macro_f1": rl_metrics["macro_f1"] - controller_metrics["macro_f1"],
                "controller_test_accuracy": controller_metrics["accuracy"],
                "rl_test_accuracy": rl_metrics["accuracy"],
                "delta_accuracy": rl_metrics["accuracy"] - controller_metrics["accuracy"],
                "rl_roc_auc": rl_metrics["roc_auc"],
                "mcnemar_b": mcnemar["b"],
                "mcnemar_c": mcnemar["c"],
                "mcnemar_p_value": mcnemar["p_value"],
                "rl_predictions_path": str(rl_path),
            }
        )

    summary_df = pd.DataFrame(rows)
    aggregate = _aggregate(summary_df)
    summary_path = Path(comparison["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    result = {
        "summary_path": str(summary_path),
        "details_path": comparison["details_path"],
        "controller_method": str(best_controller["method"]),
        "rows": rows,
        "aggregate": aggregate,
        "comparison_note": "RL seed outputs are compared against the best validation-selected same-state controller baseline.",
    }
    write_json(result, comparison["details_path"])
    return result


def _aggregate(summary_df: pd.DataFrame) -> dict[str, Any]:
    if summary_df.empty:
        return {}
    return {
        "num_seeds": int(len(summary_df)),
        "controller_macro_f1": float(summary_df["controller_test_macro_f1"].iloc[0]),
        "rl_macro_f1_mean": float(summary_df["rl_test_macro_f1"].mean()),
        "rl_macro_f1_std": float(summary_df["rl_test_macro_f1"].std(ddof=1)) if len(summary_df) > 1 else 0.0,
        "delta_macro_f1_mean": float(summary_df["delta_macro_f1"].mean()),
        "delta_macro_f1_std": float(summary_df["delta_macro_f1"].std(ddof=1)) if len(summary_df) > 1 else 0.0,
        "mcnemar_p_value_min": float(summary_df["mcnemar_p_value"].min()),
        "mcnemar_p_value_max": float(summary_df["mcnemar_p_value"].max()),
    }
