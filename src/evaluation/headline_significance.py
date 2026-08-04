from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.baselines.methods import add_baseline_probabilities
from src.evaluation.metrics import binary_classification_metrics
from src.evaluation.seed_significance import mcnemar_exact
from src.utils.file_io import write_json


def run_headline_significance(config: dict[str, Any]) -> dict[str, Any]:
    """Compare the headline RL fusion run with the main deterministic baseline.

    This test is intentionally separate from the multi-seed stability check because
    Table 7.1 compares the main final RL run against fixed-fusion baselines.
    """
    comparison = config["headline_significance"]
    baseline_method = str(comparison.get("baseline_method", "equal_fusion"))
    threshold = float(comparison.get("threshold", 0.5))

    test_outputs = pd.read_csv(comparison["test_outputs"])
    rl_predictions = pd.read_csv(comparison["rl_predictions"])
    baseline_outputs = add_baseline_probabilities(test_outputs)
    if baseline_method not in baseline_outputs.columns:
        raise ValueError(f"Unknown baseline method: {baseline_method}")

    labels = baseline_outputs["label"].to_numpy(dtype=int)
    baseline_probs = baseline_outputs[baseline_method].to_numpy(dtype=float)
    baseline_pred = (baseline_probs >= threshold).astype(int)

    merged = baseline_outputs[["sample_id", "label"]].merge(
        rl_predictions[["sample_id", "final_probability", "final_prediction"]],
        on="sample_id",
        how="inner",
        suffixes=("", "_rl"),
    )
    if len(merged) != len(baseline_outputs):
        raise ValueError("RL predictions do not cover all test samples for headline comparison.")

    rl_probs = merged["final_probability"].to_numpy(dtype=float)
    rl_pred = merged["final_prediction"].to_numpy(dtype=int)
    mcnemar = mcnemar_exact(labels, rl_pred, baseline_pred)
    baseline_metrics = binary_classification_metrics(labels, baseline_probs)
    rl_metrics = binary_classification_metrics(labels, rl_probs)

    result = {
        "comparison": f"rl_adaptive_fusion_vs_{baseline_method}",
        "threshold": threshold,
        "rows": int(len(labels)),
        "baseline_method": baseline_method,
        "baseline_metrics": baseline_metrics,
        "rl_metrics": rl_metrics,
        "delta_macro_f1": float(rl_metrics["macro_f1"] - baseline_metrics["macro_f1"]),
        "delta_accuracy": float(rl_metrics["accuracy"] - baseline_metrics["accuracy"]),
        "mcnemar": mcnemar,
    }
    write_json(result, comparison["output_path"])

    summary_path = comparison.get("summary_path")
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "comparison": result["comparison"],
                    "rows": result["rows"],
                    "baseline_macro_f1": baseline_metrics["macro_f1"],
                    "rl_macro_f1": rl_metrics["macro_f1"],
                    "delta_macro_f1": result["delta_macro_f1"],
                    "baseline_accuracy": baseline_metrics["accuracy"],
                    "rl_accuracy": rl_metrics["accuracy"],
                    "delta_accuracy": result["delta_accuracy"],
                    "mcnemar_b": mcnemar["b"],
                    "mcnemar_c": mcnemar["c"],
                    "mcnemar_p_value": mcnemar["p_value"],
                }
            ]
        ).to_csv(summary_path, index=False)
        result["summary_path"] = str(summary_path)
    return result
