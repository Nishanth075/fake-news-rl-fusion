"""Check whether expected reproducibility artifacts are present.

The repository intentionally tracks lightweight evidence and source files, while
large datasets/checkpoints live in a separate Drive artifact bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GIT_EVIDENCE = [
    "README.md",
    "requirements.txt",
    "PROJECT_HANDOFF.md",
    "ARTIFACT_MANIFEST.md",
    "REPRODUCIBILITY.md",
    "configs/final_fusion.yaml",
    "configs/final_text_model.yaml",
    "configs/final_image_model.yaml",
    "configs/final_controller_baselines.yaml",
    "configs/final_headline_significance.yaml",
    "configs/final_explainability_faithfulness.yaml",
    "scripts/train_rl_fusion.py",
    "scripts/evaluate_controller_baselines.py",
    "scripts/run_headline_significance.py",
    "scripts/run_explainability_faithfulness.py",
    "src/fusion/train.py",
    "src/evaluation/controller_baselines.py",
    "src/evaluation/headline_significance.py",
    "src/explainability/faithfulness.py",
    "outputs/metrics/fakeddit_stats.json",
    "outputs/metrics/final_baseline_results.csv",
    "outputs/metrics/final_rl_fusion_metrics.json",
    "outputs/metrics/final_supervised_fusion_matched_metrics.json",
    "outputs/metrics/final_threshold_tuning_matched_summary.csv",
    "outputs/metrics/final_seed_significance_summary.csv",
    "outputs/metrics/final_headline_significance_summary.csv",
    "outputs/metrics/final_rl_controller_seed_comparison.csv",
    "outputs/metrics/final_explainability_faithfulness_n300.csv",
    "outputs/metrics/external_openfake_calibration_holdout.csv",
    "outputs/tables/final_method_comparison.csv",
    "outputs/tables/final_results_summary.json",
    "thesis_latex/main.tex",
    "thesis_latex/main.pdf",
    "thesis_latex/generated/07_evaluation.tex",
    "thesis_latex/generated/08_conclusion.tex",
    "thesis_latex/tables/table_7_8_external_openfake_calibration.csv",
]

FULL_FINAL_ARTIFACTS = GIT_EVIDENCE + [
    "data/raw/fakeddit_v2",
    "data/processed/fakeddit_dataset.csv",
    "data/final_splits/train.csv",
    "data/final_splits/validation.csv",
    "data/final_splits/test.csv",
    "data/final_splits_available/train.csv",
    "data/final_splits_available/validation.csv",
    "data/final_splits_available/test.csv",
    "data/final_modality_outputs/train_outputs.csv",
    "data/final_modality_outputs/validation_outputs.csv",
    "data/final_modality_outputs/test_outputs.csv",
    "data/final_modality_outputs/train_image_outputs.csv",
    "data/final_modality_outputs/validation_image_outputs.csv",
    "data/final_modality_outputs/test_image_outputs.csv",
    "data/final_modality_outputs/train_text_outputs.csv",
    "data/final_modality_outputs/validation_text_outputs.csv",
    "data/final_modality_outputs/test_text_outputs.csv",
    "data/images/fakeddit",
    "outputs/checkpoints/final_image_model/best_image_model.pt",
    "outputs/checkpoints/final_text_model/best_text_model.pt",
    "outputs/checkpoints/final_rl_fusion/best_rl_fusion.pt",
    "outputs/checkpoints/final_supervised_fusion_matched/best_supervised_fusion.pt",
    "outputs/explainability",
    "outputs/metrics/seed_significance",
    "outputs/metrics/controller_baselines",
    "outputs/metrics/final_rl_fusion_test_predictions.csv",
    "outputs/metrics/final_supervised_fusion_matched_test_predictions.csv",
]

EXTERNAL_OPENFAKE_ARTIFACTS = [
    "data/external_openfake_splits/test.csv",
    "data/external_openfake_modality_outputs/test_image_outputs.csv",
    "data/external_openfake_modality_outputs/test_text_outputs.csv",
    "data/external_openfake_modality_outputs/test_outputs.csv",
    "outputs/metrics/external_openfake_calibration_holdout.csv",
    "outputs/metrics/external_openfake_calibration_holdout.json",
    "outputs/metrics/external_openfake_rl_fusion_test_predictions.csv",
]

PROFILES = {
    "git-evidence": GIT_EVIDENCE,
    "final-full": FULL_FINAL_ARTIFACTS,
    "external-openfake": EXTERNAL_OPENFAKE_ARTIFACTS,
}


def check_paths(root: Path, paths: list[str]) -> dict[str, object]:
    present: list[str] = []
    missing: list[str] = []
    for relative_path in paths:
        candidate = root / relative_path
        if candidate.exists():
            present.append(relative_path)
        else:
            missing.append(relative_path)
    return {
        "root": str(root.resolve()),
        "checked": len(paths),
        "present_count": len(present),
        "missing_count": len(missing),
        "complete": not missing,
        "missing": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="git-evidence",
        help="Artifact group to verify.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root or extracted artifact-bundle root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = check_paths(Path(args.root), PROFILES[args.profile])
    result["profile"] = args.profile
    print(json.dumps(result, indent=2))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
