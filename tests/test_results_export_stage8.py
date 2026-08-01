from __future__ import annotations

import json

import pandas as pd

from src.reporting.export_results import export_debug_results


def test_export_debug_results_collects_tables(tmp_path) -> None:
    metrics = tmp_path / "metrics"
    tables = tmp_path / "tables"
    robustness = tmp_path / "robustness"
    metrics.mkdir()
    robustness.mkdir()

    _write_json(metrics / "fakeddit_stats.json", {"total_rows": 10, "splits": {}, "class_distribution": {}})
    _write_json(metrics / "image_outputs.json", {"splits": {}})
    _write_json(metrics / "text_outputs.json", {"splits": {}})
    _write_json(metrics / "rl.json", {"test": {"split": "test", "metrics": {"macro_f1": 0.7, "accuracy": 0.8, "roc_auc": 0.9}}})
    _write_json(
        metrics / "policy.json",
        {
            "action_distribution": {"3": 2},
            "weight_summary": {"average_image_weight": 0.5},
            "oracle_action_upper_bound": {"metrics": {"macro_f1": 0.9}},
        },
    )
    _write_json(tmp_path / "image_download.json", {"total": {"downloaded": 5}})

    pd.DataFrame(
        [{"split": "test", "method": "equal", "macro_f1": 0.6, "accuracy": 0.6, "roc_auc": 0.7}]
    ).to_csv(metrics / "baseline.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "full",
                "test_macro_f1": 0.7,
                "test_accuracy": 0.8,
                "test_roc_auc": 0.9,
            }
        ]
    ).to_csv(metrics / "ablation.csv", index=False)
    pd.DataFrame(
        [
            {
                "corruption": "image_quality_drop",
                "modality": "image",
                "level": 0.5,
                "macro_f1": 0.6,
                "delta_quality": -0.3,
                "delta_target_weight": -0.1,
                "adapted_in_expected_direction": True,
            }
        ]
    ).to_csv(robustness / "summary.csv", index=False)

    result = export_debug_results(
        {
            "paths": {
                "output_dir": str(tables),
                "summary_json": str(tables / "summary.json"),
                "method_table_csv": str(tables / "methods.csv"),
                "stage_status_csv": str(tables / "stages.csv"),
            },
            "inputs": {
                "fakeddit_stats": str(metrics / "fakeddit_stats.json"),
                "image_metrics": str(metrics / "image_outputs.json"),
                "text_metrics": str(metrics / "text_outputs.json"),
                "baseline_results": str(metrics / "baseline.csv"),
                "rl_metrics": str(metrics / "rl.json"),
                "rl_policy_analysis": str(metrics / "policy.json"),
                "ablation_summary": str(metrics / "ablation.csv"),
                "robustness_summary": str(robustness / "summary.csv"),
                "image_download_stats": str(tmp_path / "image_download.json"),
            },
        }
    )

    methods = pd.read_csv(result["method_table_csv"])
    assert (tables / "summary.json").exists()
    assert len(methods) == 3
    assert "dataset_summary" in result


def _write_json(path, data) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
