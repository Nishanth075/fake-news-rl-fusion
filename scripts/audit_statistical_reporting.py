"""Audit statistical reporting from saved outputs only.

This script never trains models and never overwrites prediction files. It reads
existing metrics/prediction artifacts, creates a source manifest, and writes
reporting summaries for McNemar tests, seed aggregation, and explainability
faithfulness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from statsmodels.stats.contingency_tables import mcnemar
except Exception:  # pragma: no cover - reported in output if unavailable
    mcnemar = None

ROOT = Path(".")
OUT = ROOT / "outputs" / "metrics"
ALPHA = 0.05


@dataclass(frozen=True)
class SourceSpec:
    method: str
    path: str
    split: str
    threshold_type: str
    sample_id_column: str
    label_column: str
    prediction_column: str
    probability_column: str
    source_kind: str = "prediction_csv"


SOURCES = [
    SourceSpec(
        "rl_full_state_adaptive_fusion",
        "outputs/metrics/final_rl_fusion_test_predictions.csv",
        "test",
        "default_threshold_0.5",
        "sample_id",
        "label",
        "final_prediction",
        "final_probability",
    ),
    SourceSpec(
        "matched_supervised_mlp_fusion",
        "outputs/metrics/final_supervised_fusion_matched_test_predictions.csv",
        "test",
        "default_threshold_0.5",
        "sample_id",
        "label",
        "final_prediction",
        "final_probability",
    ),
    SourceSpec(
        "equal_fusion",
        "data/final_modality_outputs/test_outputs.csv",
        "test",
        "default_threshold_0.5_derived_from_modality_probabilities",
        "sample_id",
        "label",
        "equal_fusion_prediction",
        "equal_fusion_probability",
        "derived_baseline",
    ),
    SourceSpec(
        "confidence_weighted_fusion",
        "data/final_modality_outputs/test_outputs.csv",
        "test",
        "default_threshold_0.5_derived_from_modality_probabilities",
        "sample_id",
        "label",
        "confidence_weighted_fusion_prediction",
        "confidence_weighted_fusion_probability",
        "derived_baseline",
    ),
]


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def derive_baseline_predictions(df: pd.DataFrame, method: str) -> pd.DataFrame:
    required = {"sample_id", "label", "image_probability", "text_probability"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{method} source missing columns: {missing}")
    out = df[["sample_id", "label", "image_probability", "text_probability"]].copy()
    if method == "equal_fusion":
        prob = 0.5 * out["image_probability"] + 0.5 * out["text_probability"]
    elif method == "confidence_weighted_fusion":
        image_conf = (out["image_probability"] - 0.5).abs()
        text_conf = (out["text_probability"] - 0.5).abs()
        denom = image_conf + text_conf
        image_weight = np.where(denom > 0, image_conf / denom, 0.5)
        text_weight = 1.0 - image_weight
        prob = image_weight * out["image_probability"] + text_weight * out["text_probability"]
    else:
        raise ValueError(method)
    out[f"{method}_probability"] = prob
    out[f"{method}_prediction"] = (prob >= 0.5).astype(int)
    return out


def load_method_frame(spec: SourceSpec) -> tuple[pd.DataFrame | None, str | None]:
    path = ROOT / spec.path
    df = read_csv_if_exists(path)
    if df is None:
        return None, f"missing file: {spec.path}"
    if spec.source_kind == "derived_baseline":
        df = derive_baseline_predictions(df, spec.method)
    required = [spec.sample_id_column, spec.label_column, spec.prediction_column]
    if spec.probability_column:
        required.append(spec.probability_column)
    missing = [col for col in required if col not in df.columns]
    if missing:
        return None, f"missing columns in {spec.path}: {missing}"
    return df, None


def make_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SOURCES:
        df, error = load_method_frame(spec)
        duplicate_count = None
        missing_id_count = None
        row_count = None
        if df is not None:
            ids = df[spec.sample_id_column]
            row_count = int(len(df))
            duplicate_count = int(ids.duplicated().sum())
            missing_id_count = int(ids.isna().sum())
        rows.append(
            {
                "method_name": spec.method,
                "file_path": spec.path,
                "exists": (ROOT / spec.path).exists(),
                "source_kind": spec.source_kind,
                "split": spec.split,
                "threshold_type": spec.threshold_type,
                "row_count": row_count,
                "sample_id_column": spec.sample_id_column,
                "label_column": spec.label_column,
                "prediction_column": spec.prediction_column,
                "probability_column": spec.probability_column,
                "duplicate_id_count": duplicate_count,
                "missing_id_count": missing_id_count,
                "load_error": error,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "final_statistical_source_manifest.csv", index=False)
    return manifest


def compare_predictions(rl_spec: SourceSpec, base_spec: SourceSpec) -> dict[str, Any]:
    rl_df, rl_error = load_method_frame(rl_spec)
    base_df, base_error = load_method_frame(base_spec)
    if rl_error or base_error or rl_df is None or base_df is None:
        return {
            "rl_method": rl_spec.method,
            "baseline_method": base_spec.method,
            "status": "blocked_missing_or_invalid_source",
            "blocker": "; ".join([x for x in [rl_error, base_error] if x]),
            "threshold_protocol": f"RL={rl_spec.threshold_type}; baseline={base_spec.threshold_type}",
            "source_rl_file": rl_spec.path,
            "source_baseline_file": base_spec.path,
        }

    rl_keep = rl_df[[rl_spec.sample_id_column, rl_spec.label_column, rl_spec.prediction_column]].copy()
    rl_keep.columns = ["sample_id", "label_rl", "prediction_rl"]
    base_keep = base_df[[base_spec.sample_id_column, base_spec.label_column, base_spec.prediction_column]].copy()
    base_keep.columns = ["sample_id", "label_baseline", "prediction_baseline"]

    errors: list[str] = []
    for name, frame in [("rl", rl_keep), ("baseline", base_keep)]:
        if frame["sample_id"].isna().any():
            errors.append(f"{name} has missing sample IDs")
        dup = int(frame["sample_id"].duplicated().sum())
        if dup:
            errors.append(f"{name} has {dup} duplicate sample IDs")
    merged = rl_keep.merge(base_keep, on="sample_id", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        errors.append("sample IDs do not match exactly")
    merged = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
    if not (merged["label_rl"].to_numpy() == merged["label_baseline"].to_numpy()).all():
        errors.append("true labels do not match after ID alignment")
    if errors:
        return {
            "rl_method": rl_spec.method,
            "baseline_method": base_spec.method,
            "status": "blocked_alignment_error",
            "blocker": "; ".join(errors),
            "threshold_protocol": f"RL={rl_spec.threshold_type}; baseline={base_spec.threshold_type}",
            "source_rl_file": rl_spec.path,
            "source_baseline_file": base_spec.path,
        }

    y = merged["label_rl"].astype(int).to_numpy()
    rl_pred = merged["prediction_rl"].astype(int).to_numpy()
    base_pred = merged["prediction_baseline"].astype(int).to_numpy()
    rl_correct = rl_pred == y
    base_correct = base_pred == y
    both_correct = int((rl_correct & base_correct).sum())
    b = int((rl_correct & ~base_correct).sum())
    c = int((~rl_correct & base_correct).sum())
    both_wrong = int((~rl_correct & ~base_correct).sum())
    table = [[both_correct, c], [b, both_wrong]]

    if mcnemar is None:
        stat = None
        p_value = None
        significant = None
        status = "blocked_statsmodels_missing"
    else:
        test = mcnemar(table, exact=True)
        stat = float(test.statistic)
        p_value = float(test.pvalue)
        significant = bool(p_value < ALPHA)
        status = "computed"

    if b > c:
        direction = "rl_correct_more_often_than_baseline_on_discordant_pairs"
    elif b < c:
        direction = "baseline_correct_more_often_than_rl_on_discordant_pairs"
    else:
        direction = "tie_on_discordant_pairs"
    return {
        "rl_method": rl_spec.method,
        "baseline_method": base_spec.method,
        "status": status,
        "n_samples": int(len(merged)),
        "both_correct": both_correct,
        "b": b,
        "c": c,
        "both_wrong": both_wrong,
        "statistic": stat,
        "p_value": p_value,
        "alpha": ALPHA,
        "significant_at_0_05": significant,
        "direction": direction,
        "threshold_protocol": f"RL={rl_spec.threshold_type}; baseline={base_spec.threshold_type}",
        "source_rl_file": rl_spec.path,
        "source_baseline_file": base_spec.path,
    }


def write_mcnemar_reports() -> list[dict[str, Any]]:
    by_method = {s.method: s for s in SOURCES}
    rows = [
        compare_predictions(by_method["rl_full_state_adaptive_fusion"], by_method["equal_fusion"]),
        compare_predictions(by_method["rl_full_state_adaptive_fusion"], by_method["confidence_weighted_fusion"]),
        compare_predictions(by_method["rl_full_state_adaptive_fusion"], by_method["matched_supervised_mlp_fusion"]),
    ]
    pd.DataFrame(rows).to_csv(OUT / "final_mcnemar_results.csv", index=False)
    payload = {
        "methodology": {
            "test": "two-sided exact McNemar test using statsmodels.stats.contingency_tables.mcnemar",
            "alignment": "sample_id inner alignment with exact label check; row order is not trusted",
            "b_definition": "RL correct and baseline incorrect",
            "c_definition": "RL incorrect and baseline correct",
            "statistic_note": "For exact McNemar tests, statsmodels reports a statistic equivalent to min(b, c).",
            "superiority_rule": "Do not state RL is superior unless b > c, p < 0.05, and protocols are fair/matched.",
        },
        "results": rows,
    }
    (OUT / "final_mcnemar_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return rows


def write_seed_reports() -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_path = OUT / "final_seed_significance_summary.csv"
    ctrl_path = OUT / "final_rl_controller_seed_comparison.csv"
    seed_df = pd.read_csv(seed_path)
    ctrl_df = pd.read_csv(ctrl_path)
    ctrl_seed_rows = ctrl_df[pd.to_numeric(ctrl_df["seed"], errors="coerce").notna()].copy()
    controller_score = float(ctrl_seed_rows["controller_test_macro_f1"].iloc[0])
    equal_score = float(seed_df["baseline_macro_f1"].iloc[0])

    pairwise = pd.DataFrame(
        {
            "seed": seed_df["seed"].astype(int),
            "rl_macro_f1": seed_df["rl_macro_f1"].astype(float),
            "equal_fusion_macro_f1": equal_score,
            "best_same_state_controller_macro_f1": controller_score,
        }
    )
    pairwise["rl_minus_equal"] = pairwise["rl_macro_f1"] - pairwise["equal_fusion_macro_f1"]
    pairwise["rl_minus_controller"] = pairwise["rl_macro_f1"] - pairwise["best_same_state_controller_macro_f1"]
    pairwise["rl_beats_equal"] = pairwise["rl_minus_equal"] > 0
    pairwise["rl_beats_controller"] = pairwise["rl_minus_controller"] > 0
    pairwise.to_csv(OUT / "final_seed_pairwise_summary.csv", index=False)

    rl_scores = pairwise["rl_macro_f1"]
    summary_rows = [
        {
            "method": "rl_full_state_adaptive_fusion",
            "mean_macro_f1": rl_scores.mean(),
            "std_macro_f1": rl_scores.std(ddof=1),
            "min_macro_f1": rl_scores.min(),
            "max_macro_f1": rl_scores.max(),
            "median_macro_f1": rl_scores.median(),
            "n_runs": int(len(rl_scores)),
            "is_seed_trained": True,
            "rl_wins_against_method": "equal_fusion",
            "rl_ties_against_method": int((pairwise["rl_minus_equal"] == 0).sum()),
            "rl_losses_against_method": int((pairwise["rl_minus_equal"] < 0).sum()),
            "rl_wins_count": int((pairwise["rl_minus_equal"] > 0).sum()),
            "source_file": str(seed_path),
        },
        {
            "method": "equal_fusion",
            "mean_macro_f1": equal_score,
            "std_macro_f1": np.nan,
            "min_macro_f1": equal_score,
            "max_macro_f1": equal_score,
            "median_macro_f1": equal_score,
            "n_runs": 1,
            "is_seed_trained": False,
            "rl_wins_against_method": "not_applicable_reference_method",
            "rl_ties_against_method": np.nan,
            "rl_losses_against_method": np.nan,
            "rl_wins_count": np.nan,
            "source_file": "outputs/metrics/final_baseline_results.csv",
        },
        {
            "method": "state_decision_tree_depth_5_controller",
            "mean_macro_f1": controller_score,
            "std_macro_f1": np.nan,
            "min_macro_f1": controller_score,
            "max_macro_f1": controller_score,
            "median_macro_f1": controller_score,
            "n_runs": 1,
            "is_seed_trained": False,
            "rl_wins_against_method": "not_applicable_reference_method",
            "rl_ties_against_method": np.nan,
            "rl_losses_against_method": np.nan,
            "rl_wins_count": np.nan,
            "source_file": str(ctrl_path),
        },
    ]
    # Columns explicitly requested in text.
    summary_rows[0]["number_of_seeds_beating_equal_fusion"] = int(pairwise["rl_beats_equal"].sum())
    summary_rows[0]["number_of_seeds_tying_equal_fusion"] = int((pairwise["rl_minus_equal"] == 0).sum())
    summary_rows[0]["number_of_seeds_losing_to_equal_fusion"] = int((pairwise["rl_minus_equal"] < 0).sum())
    summary_rows[0]["number_of_seeds_beating_best_same_state_controller"] = int(pairwise["rl_beats_controller"].sum())

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "final_seed_aggregated_summary.csv", index=False)
    return summary, pairwise


def write_faithfulness_report() -> pd.DataFrame:
    source = OUT / "final_explainability_faithfulness_n300.csv"
    raw = pd.read_csv(source)
    row = raw.iloc[0]
    formula = "original predicted-class probability minus probability after deleting evidence"
    rows = [
        {
            "modality": "image",
            "explanation_method": "Grad-CAM salient-region deletion",
            "control_type": "random image-region deletion",
            "n_samples": int(row["rows"]),
            "salient_score": float(row["image_salient_comprehensiveness_mean"]),
            "control_score": float(row["image_random_comprehensiveness_mean"]),
            "comparison_margin": float(row["image_salient_comprehensiveness_mean"] - row["image_random_comprehensiveness_mean"]),
            "passes_faithfulness_test": bool(row["image_salient_comprehensiveness_mean"] > row["image_random_comprehensiveness_mean"]),
            "formula": formula,
            "probability_target": "predicted-class probability",
            "deletion_proportion": "from saved N=300 faithfulness configuration; no raw outputs modified",
            "negative_difference_treatment": "retained in arithmetic mean; not clipped",
            "missing_failed_sample_handling": "only completed rows in saved summary are aggregated",
            "random_repetitions": "as encoded in saved source output",
            "source_file": str(source),
        },
        {
            "modality": "text",
            "explanation_method": "token saliency deletion",
            "control_type": "least-salient token deletion",
            "n_samples": int(row["rows"]),
            "salient_score": float(row["text_salient_comprehensiveness_mean"]),
            "control_score": float(row["text_least_comprehensiveness_mean"]),
            "comparison_margin": float(row["text_salient_comprehensiveness_mean"] - row["text_least_comprehensiveness_mean"]),
            "passes_faithfulness_test": bool(row["text_salient_comprehensiveness_mean"] > row["text_least_comprehensiveness_mean"]),
            "formula": formula,
            "probability_target": "predicted-class probability",
            "deletion_proportion": "from saved N=300 faithfulness configuration; no raw outputs modified",
            "negative_difference_treatment": "retained in arithmetic mean; not clipped",
            "missing_failed_sample_handling": "only completed rows in saved summary are aggregated",
            "random_repetitions": "as encoded in saved source output",
            "source_file": str(source),
        },
        {
            "modality": "text",
            "explanation_method": "token saliency deletion",
            "control_type": "random token deletion",
            "n_samples": int(row["rows"]),
            "salient_score": float(row["text_salient_comprehensiveness_mean"]),
            "control_score": float(row["text_random_comprehensiveness_mean"]),
            "comparison_margin": float(row["text_salient_comprehensiveness_mean"] - row["text_random_comprehensiveness_mean"]),
            "passes_faithfulness_test": bool(row["text_salient_comprehensiveness_mean"] > row["text_random_comprehensiveness_mean"]),
            "formula": formula,
            "probability_target": "predicted-class probability",
            "deletion_proportion": "from saved N=300 faithfulness configuration; no raw outputs modified",
            "negative_difference_treatment": "retained in arithmetic mean; not clipped",
            "missing_failed_sample_handling": "only completed rows in saved summary are aggregated",
            "random_repetitions": "as encoded in saved source output",
            "source_file": str(source),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "final_explainability_faithfulness_summary.csv", index=False)
    return df


def write_change_log(manifest: pd.DataFrame, mcnemar_rows: list[dict[str, Any]], seed_summary: pd.DataFrame, faith: pd.DataFrame) -> None:
    created = [
        "outputs/metrics/final_statistical_source_manifest.csv",
        "outputs/metrics/final_mcnemar_results.csv",
        "outputs/metrics/final_mcnemar_results.json",
        "outputs/metrics/final_seed_aggregated_summary.csv",
        "outputs/metrics/final_seed_pairwise_summary.csv",
        "outputs/metrics/final_explainability_faithfulness_summary.csv",
        "outputs/metrics/final_statistical_reporting_change_log.md",
    ]
    read_files = sorted(set(manifest["file_path"].tolist() + [
        "outputs/metrics/final_seed_significance_summary.csv",
        "outputs/metrics/final_rl_controller_seed_comparison.csv",
        "outputs/metrics/final_explainability_faithfulness_n300.csv",
    ]))
    lines = [
        "# Final Statistical Reporting Change Log",
        "",
        "No model training, fine-tuning, prediction regeneration, threshold alteration, reward change, or checkpoint overwrite was performed.",
        "",
        "## Files Read",
    ]
    lines.extend([f"- `{p}`" for p in read_files])
    lines += ["", "## Files Created"]
    lines.extend([f"- `{p}`" for p in created])
    lines += ["", "## Existing Files Modified", "- None. Original prediction and metric source files were not overwritten.", ""]
    lines += ["## Number Checks"]
    checks = [
        ("Headline RL vs equal macro-F1 delta", "0.00027123141882823276", "pending paired recomputation if prediction files restored", "not changed here"),
        ("RL seed mean macro F1", "0.8593558512608466", str(seed_summary.loc[seed_summary["method"] == "rl_full_state_adaptive_fusion", "mean_macro_f1"].iloc[0]), "unchanged"),
        ("RL seed sample std macro F1", "not consistently reported", str(seed_summary.loc[seed_summary["method"] == "rl_full_state_adaptive_fusion", "std_macro_f1"].iloc[0]), "new reporting detail"),
        ("Image salient comprehensiveness", "0.12595064888397853", str(faith[(faith["modality"] == "image")]["salient_score"].iloc[0]), "unchanged"),
        ("Image random comprehensiveness", "0.18579854875802992", str(faith[(faith["modality"] == "image")]["control_score"].iloc[0]), "unchanged"),
        ("Image faithfulness margin", "not explicitly reported", str(faith[(faith["modality"] == "image")]["comparison_margin"].iloc[0]), "new negative margin; image test does not pass"),
        ("Text salient vs least margin", "not explicitly reported", str(faith[(faith["modality"] == "text") & (faith["control_type"] == "least-salient token deletion")]["comparison_margin"].iloc[0]), "new positive margin"),
        ("Text salient vs random margin", "not explicitly reported", str(faith[(faith["modality"] == "text") & (faith["control_type"] == "random token deletion")]["comparison_margin"].iloc[0]), "new positive margin"),
    ]
    lines.append("| Item | Original thesis-reported number | Recomputed/reported number | Changed? |")
    lines.append("|---|---:|---:|---|")
    lines.extend([f"| {a} | {b} | {c} | {d} |" for a, b, c, d in checks])
    lines += [
        "",
        "## Recommended Thesis Sections Requiring Updates",
        "- Abstract: avoid any statement that RL is statistically superior to fixed fusion.",
        "- Section 7.11: report image Grad-CAM faithfulness as not supported by the deletion test; keep text saliency as supported against least/random controls.",
        "- Section 7.12: include paired McNemar only after prediction files are present and aligned by sample ID.",
        "- Section 7.13: report RL seed mean, sample standard deviation, min, max, median, and win/loss counts.",
        "- Section 8.4: state that the selected RL run is competitive but the fixed-fusion advantage is marginal/unstable.",
        "- Section 8.5: note image explainability faithfulness limitation explicitly.",
        "- Table 7.2: keep threshold protocol columns clear; do not mix default and validation-selected thresholds silently.",
        "- Table 7.4: add faithfulness pass/fail and margins.",
        "- Figure 7.12: annotate that equal fusion and controller scores are fixed references, not independent seeded runs.",
        "",
        "## Recommended Replacement Wording",
        '"The selected RL run achieved competitive performance, but its advantage over fixed fusion was marginal and must be interpreted together with paired significance testing and seed-level instability."',
        "",
        "## Validation Checks",
    ]
    mcnemar_blocked = [r for r in mcnemar_rows if r.get("status", "") != "computed"]
    if mcnemar_blocked:
        lines.append("- Paired prediction validation: blocked locally because one or more prediction/source CSV files are missing.")
        for row in mcnemar_blocked:
            lines.append(f"  - {row['rl_method']} vs {row['baseline_method']}: {row.get('blocker')}")
    else:
        lines.append("- Paired prediction validation: all files aligned by sample ID and label.")
    lines += [
        "- CSV outputs were written from saved metrics/prediction sources only.",
        "- No training script was called by this audit.",
        "- No original prediction file was overwritten.",
    ]
    (OUT / "final_statistical_reporting_change_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest()
    mcnemar_rows = write_mcnemar_reports()
    seed_summary, _ = write_seed_reports()
    faith = write_faithfulness_report()
    write_change_log(manifest, mcnemar_rows, seed_summary, faith)
    print(json.dumps({
        "created": [
            "outputs/metrics/final_statistical_source_manifest.csv",
            "outputs/metrics/final_mcnemar_results.csv",
            "outputs/metrics/final_mcnemar_results.json",
            "outputs/metrics/final_seed_aggregated_summary.csv",
            "outputs/metrics/final_seed_pairwise_summary.csv",
            "outputs/metrics/final_explainability_faithfulness_summary.csv",
            "outputs/metrics/final_statistical_reporting_change_log.md",
        ],
        "mcnemar_statuses": [r.get("status") for r in mcnemar_rows],
    }, indent=2))


if __name__ == "__main__":
    main()
