from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.methods import add_baseline_probabilities
from src.evaluation.metrics import binary_classification_metrics
from src.fusion.train import train_rl_fusion
from src.utils.config import load_yaml
from src.utils.file_io import write_json


def run_seed_significance(config: dict[str, Any]) -> dict[str, Any]:
    """Run RL fusion over multiple seeds and compare with a deterministic baseline."""
    comparison = config["comparison"]
    seeds = [int(seed) for seed in comparison.get("seeds", [42, 7, 13])]
    baseline_method = str(comparison.get("baseline_method", "equal_fusion"))
    output_dir = Path(comparison.get("output_dir", "outputs/metrics/seed_significance"))
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rl_config_path = comparison["base_rl_config"]
    base_rl_config = load_yaml(base_rl_config_path)
    test_df = pd.read_csv(base_rl_config["paths"]["test_outputs"])
    baseline_probabilities = add_baseline_probabilities(test_df)
    if baseline_method not in baseline_probabilities.columns:
        raise ValueError(f"Unknown baseline method: {baseline_method}")

    baseline_probs = baseline_probabilities[baseline_method].to_numpy(dtype=float)
    labels = baseline_probabilities["label"].to_numpy(dtype=int)
    baseline_metrics = binary_classification_metrics(labels, baseline_probs)
    baseline_predictions = (baseline_probs >= 0.5).astype(int)

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"baseline_method": baseline_method, "baseline_metrics": baseline_metrics, "seeds": {}}
    for seed in seeds:
        run_config = _seeded_rl_config(base_rl_config, seed, output_dir)
        rl_result = train_rl_fusion(run_config)
        predictions_path = rl_result["test"]["predictions_path"]
        rl_predictions_df = pd.read_csv(predictions_path)
        rl_probs = rl_predictions_df["final_probability"].to_numpy(dtype=float)
        rl_predictions = (rl_probs >= 0.5).astype(int)
        rl_metrics = binary_classification_metrics(labels, rl_probs)
        mcnemar = mcnemar_exact(labels, rl_predictions, baseline_predictions)

        row = {
            "seed": seed,
            "baseline_method": baseline_method,
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "rl_macro_f1": rl_metrics["macro_f1"],
            "delta_macro_f1": rl_metrics["macro_f1"] - baseline_metrics["macro_f1"],
            "baseline_accuracy": baseline_metrics["accuracy"],
            "rl_accuracy": rl_metrics["accuracy"],
            "delta_accuracy": rl_metrics["accuracy"] - baseline_metrics["accuracy"],
            "rl_roc_auc": rl_metrics["roc_auc"],
            "mcnemar_b": mcnemar["b"],
            "mcnemar_c": mcnemar["c"],
            "mcnemar_p_value": mcnemar["p_value"],
            "rl_predictions_path": predictions_path,
        }
        rows.append(row)
        details["seeds"][str(seed)] = {"rl_metrics": rl_metrics, "mcnemar": mcnemar, "run_config": run_config}

    summary_df = pd.DataFrame(rows)
    aggregate = _aggregate_rows(summary_df)
    summary_path = Path(comparison["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    result = {
        "summary_path": str(summary_path),
        "details_path": str(comparison["details_path"]),
        "baseline_method": baseline_method,
        "rows": rows,
        "aggregate": aggregate,
    }
    details["aggregate"] = aggregate
    write_json(details, comparison["details_path"])
    return result


def mcnemar_exact(labels: np.ndarray, predictions_a: np.ndarray, predictions_b: np.ndarray) -> dict[str, Any]:
    """Two-sided exact McNemar test for paired binary classifier correctness."""
    y_true = np.asarray(labels).astype(int)
    a_correct = np.asarray(predictions_a).astype(int) == y_true
    b_correct = np.asarray(predictions_b).astype(int) == y_true
    b_count = int(np.logical_and(a_correct, ~b_correct).sum())
    c_count = int(np.logical_and(~a_correct, b_correct).sum())
    discordant = b_count + c_count
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, i) for i in range(0, min(b_count, c_count) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"b": b_count, "c": c_count, "discordant": discordant, "p_value": float(p_value)}


def _seeded_rl_config(base_config: dict[str, Any], seed: int, output_dir: Path) -> dict[str, Any]:
    run_config = copy.deepcopy(base_config)
    run_config["seed"] = seed
    paths = run_config["paths"]
    paths["checkpoint_dir"] = str(output_dir / f"checkpoints_seed_{seed}")
    paths["metrics_path"] = str(output_dir / f"rl_seed_{seed}_metrics.json")
    paths["validation_predictions_path"] = str(output_dir / f"rl_seed_{seed}_validation_predictions.csv")
    paths["test_predictions_path"] = str(output_dir / f"rl_seed_{seed}_test_predictions.csv")
    return run_config


def _aggregate_rows(summary_df: pd.DataFrame) -> dict[str, Any]:
    if summary_df.empty:
        return {}
    return {
        "num_seeds": int(len(summary_df)),
        "rl_macro_f1_mean": float(summary_df["rl_macro_f1"].mean()),
        "rl_macro_f1_std": float(summary_df["rl_macro_f1"].std(ddof=1)) if len(summary_df) > 1 else 0.0,
        "baseline_macro_f1": float(summary_df["baseline_macro_f1"].iloc[0]),
        "delta_macro_f1_mean": float(summary_df["delta_macro_f1"].mean()),
        "delta_macro_f1_std": float(summary_df["delta_macro_f1"].std(ddof=1)) if len(summary_df) > 1 else 0.0,
        "mcnemar_p_value_min": float(summary_df["mcnemar_p_value"].min()),
        "mcnemar_p_value_max": float(summary_df["mcnemar_p_value"].max()),
    }
