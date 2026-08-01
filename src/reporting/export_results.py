from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.file_io import write_json


def export_debug_results(config: dict[str, Any]) -> dict[str, Any]:
    """Collect debug-stage metrics into thesis-friendly summary tables."""
    paths = config["paths"]
    inputs = config["inputs"]
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    collected: dict[str, Any] = {"available_inputs": {}, "missing_inputs": []}
    for key, input_path in inputs.items():
        path = Path(input_path)
        if path.exists():
            collected["available_inputs"][key] = str(path)
        else:
            collected["missing_inputs"].append(key)

    method_rows = []
    method_rows.extend(_baseline_rows(inputs.get("baseline_results")))
    method_rows.extend(_rl_rows(inputs.get("rl_metrics")))
    method_rows.extend(_ablation_rows(inputs.get("ablation_summary")))
    method_table = pd.DataFrame(method_rows)
    if not method_table.empty:
        method_table = method_table.sort_values(["split", "macro_f1"], ascending=[True, False])
    method_table.to_csv(paths["method_table_csv"], index=False)

    stage_rows = _stage_status_rows(inputs)
    pd.DataFrame(stage_rows).to_csv(paths["stage_status_csv"], index=False)

    collected["method_table_csv"] = paths["method_table_csv"]
    collected["stage_status_csv"] = paths["stage_status_csv"]
    collected["dataset_summary"] = _dataset_summary(inputs.get("fakeddit_stats"), inputs.get("image_download_stats"))
    collected["policy_summary"] = _policy_summary(inputs.get("rl_policy_analysis"))
    collected["robustness_summary"] = _robustness_summary(inputs.get("robustness_summary"))
    write_json(collected, paths["summary_json"])
    return collected


def _baseline_rows(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value or not Path(path_value).exists():
        return []
    df = pd.read_csv(path_value)
    return [
        {
            "source": "baseline",
            "split": row.split,
            "method": row.method,
            "macro_f1": row.macro_f1,
            "accuracy": row.accuracy,
            "roc_auc": row.roc_auc,
        }
        for row in df.itertuples(index=False)
    ]


def _rl_rows(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value or not Path(path_value).exists():
        return []
    data = _read_json(path_value)
    test = data.get("test", {})
    metrics = test.get("metrics", {})
    if not metrics:
        return []
    return [
        {
            "source": "rl",
            "split": test.get("split", "test"),
            "method": "rl_full_state",
            "macro_f1": metrics.get("macro_f1"),
            "accuracy": metrics.get("accuracy"),
            "roc_auc": metrics.get("roc_auc"),
        }
    ]


def _ablation_rows(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value or not Path(path_value).exists():
        return []
    df = pd.read_csv(path_value)
    return [
        {
            "source": "ablation",
            "split": "test",
            "method": f"rl_{row.name}",
            "macro_f1": row.test_macro_f1,
            "accuracy": row.test_accuracy,
            "roc_auc": row.test_roc_auc,
        }
        for row in df.itertuples(index=False)
    ]


def _stage_status_rows(inputs: dict[str, str]) -> list[dict[str, Any]]:
    stage_map = {
        "dataset": ["fakeddit_stats", "image_download_stats"],
        "image_branch": ["image_metrics"],
        "text_branch": ["text_metrics"],
        "baselines": ["baseline_results"],
        "rl_fusion": ["rl_metrics", "rl_policy_analysis"],
        "ablation": ["ablation_summary"],
        "robustness": ["robustness_summary"],
    }
    rows = []
    for stage, keys in stage_map.items():
        rows.append(
            {
                "stage": stage,
                "complete": all(Path(inputs[key]).exists() for key in keys),
                "required_files": ";".join(keys),
            }
        )
    return rows


def _dataset_summary(fakeddit_path: str | None, image_download_path: str | None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if fakeddit_path and Path(fakeddit_path).exists():
        data = _read_json(fakeddit_path)
        summary["total_rows"] = data.get("total_rows")
        summary["splits"] = data.get("splits")
        summary["class_distribution"] = data.get("class_distribution")
    if image_download_path and Path(image_download_path).exists():
        summary["debug_image_download"] = _read_json(image_download_path).get("total")
    return summary


def _policy_summary(path_value: str | None) -> dict[str, Any]:
    if not path_value or not Path(path_value).exists():
        return {}
    data = _read_json(path_value)
    return {
        "action_distribution": data.get("action_distribution"),
        "weight_summary": data.get("weight_summary"),
        "oracle_action_upper_bound": data.get("oracle_action_upper_bound", {}).get("metrics"),
    }


def _robustness_summary(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value or not Path(path_value).exists():
        return []
    df = pd.read_csv(path_value)
    cols = [
        "corruption",
        "modality",
        "level",
        "macro_f1",
        "delta_quality",
        "delta_target_weight",
        "adapted_in_expected_direction",
    ]
    return df[cols].to_dict(orient="records")


def _read_json(path_value: str | Path) -> dict[str, Any]:
    with Path(path_value).open("r", encoding="utf-8") as handle:
        return json.load(handle)
