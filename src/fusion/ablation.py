from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.fusion.train import train_rl_fusion
from src.utils.file_io import write_json


def run_rl_ablation(config: dict[str, Any]) -> dict[str, Any]:
    """Train/evaluate RL fusion with multiple state feature sets."""
    paths = config["paths"]
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details: dict[str, Any] = {"ablations": {}}

    for ablation in config["ablations"]:
        name = str(ablation["name"])
        features = list(ablation["features"])
        ablation_config = _build_ablation_config(config, name, features, output_dir)
        result = train_rl_fusion(ablation_config)
        test_metrics = result["test"]["metrics"]
        rows.append(
            {
                "name": name,
                "state_dim": len(features),
                "features": ",".join(features),
                "best_validation_macro_f1": result["best_validation_macro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_roc_auc": test_metrics["roc_auc"],
                "average_image_weight": result["test"]["average_image_weight"],
                "average_text_weight": result["test"]["average_text_weight"],
            }
        )
        details["ablations"][name] = result

    summary_df = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    summary_path = Path(paths["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    details["summary_path"] = str(summary_path)
    write_json(details, output_dir / "ablation_details.json")
    return {"summary_path": str(summary_path), "rows": rows}


def _build_ablation_config(
    config: dict[str, Any],
    name: str,
    features: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    ablation_config = {
        "seed": config.get("seed", 42),
        "fusion": deepcopy(config["base_fusion"]),
        "paths": {
            "train_outputs": config["paths"]["train_outputs"],
            "validation_outputs": config["paths"]["validation_outputs"],
            "test_outputs": config["paths"]["test_outputs"],
            "checkpoint_dir": str(output_dir / name / "checkpoint"),
            "metrics_path": str(output_dir / name / "metrics.json"),
            "test_predictions_path": str(output_dir / name / "test_predictions.csv"),
        },
    }
    ablation_config["fusion"]["features"] = features
    ablation_config["fusion"]["state_dim"] = len(features)
    return ablation_config
