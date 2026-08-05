# Artifact Manifest

This repository contains the source code, configurations, thesis source, notebooks, and lightweight result summaries. It does not store the full raw datasets, downloaded images, or model checkpoints because those artifacts are too large for normal GitHub version control.

## Evidence Included in Git

These files are small enough to keep in the repository and support the reported results:

- `outputs/metrics/fakeddit_stats.json`
- `outputs/metrics/final_image_model_metrics.json`
- `outputs/metrics/final_image_outputs_metrics.json`
- `outputs/metrics/final_text_model_metrics.json`
- `outputs/metrics/final_text_outputs_metrics.json`
- `outputs/metrics/final_baseline_results.csv`
- `outputs/metrics/final_baseline_summary.json`
- `outputs/metrics/final_rl_fusion_metrics.json`
- `outputs/metrics/final_rl_policy_analysis.json`
- `outputs/metrics/final_ablation_summary.csv`
- `outputs/metrics/final_threshold_tuning_summary.csv`
- `outputs/metrics/final_supervised_fusion_matched_metrics.json`
- `outputs/metrics/final_threshold_tuning_matched_summary.csv`
- `outputs/metrics/final_headline_significance_summary.csv`
- `outputs/metrics/final_seed_significance_summary.csv`
- `outputs/metrics/final_rl_controller_seed_comparison.csv`
- `outputs/metrics/final_explainability_faithfulness_n300.csv`
- `outputs/metrics/external_openfake_calibration_holdout.csv`
- `outputs/robustness/final_robustness_summary.csv`
- `outputs/tables/final_method_comparison.csv`
- `outputs/tables/final_results_summary.json`
- `outputs/tables/final_stage_status.csv`

The final thesis source is in:

- `thesis_latex/main.tex`
- `thesis_latex/generated/`
- `thesis_latex/figures/`
- `thesis_latex/tables/`
- `thesis_latex/main.pdf`

## Full Reproducibility Bundle Required Outside Git

To rerun or independently verify the full experiments without downloading everything again, store the following folders in Google Drive or another external archive:

- `data/raw/fakeddit_v2/`
- `data/processed/fakeddit_dataset.csv`
- `data/final_splits/`
- `data/final_splits_available/`
- `data/final_modality_outputs/`
- `data/images/fakeddit/`
- `data/external_openfake_splits/`
- `data/external_openfake_modality_outputs/`
- `data/images/openfake/`
- `outputs/checkpoints/`
- `outputs/explainability/`
- `outputs/metrics/seed_significance/`
- `outputs/metrics/controller_baselines/`
- `outputs/metrics/final_rl_fusion_test_predictions.csv`
- `outputs/metrics/final_supervised_fusion_matched_test_predictions.csv`
- `outputs/metrics/external_openfake_rl_fusion_test_predictions.csv`

## Colab Command to Create the Full Bundle

Run this from Colab after mounting Google Drive:

```bash
%cd /content/drive/MyDrive/fake-news-rl-fusion
!zip -r /content/drive/MyDrive/fake_news_full_reproducibility_bundle.zip \
  configs notebooks scripts src tests docs README.md requirements.txt run_pipeline.py PROJECT_HANDOFF.md ARTIFACT_MANIFEST.md REPRODUCIBILITY.md \
  thesis_latex thesis \
  data/raw/fakeddit_v2 data/processed/fakeddit_dataset.csv data/final_splits data/final_splits_available data/final_modality_outputs data/images/fakeddit \
  data/external_openfake_splits data/external_openfake_modality_outputs data/images/openfake \
  outputs/checkpoints outputs/explainability outputs/metrics outputs/robustness outputs/tables \
  -x "*/__pycache__/*" "*.pyc" ".git/*" "tmp/*" "tmp_pdf_pages/*"
```

If storage is limited, create a smaller evidence bundle without `data/images/fakeddit`, `data/images/openfake`, and `outputs/checkpoints`. That smaller bundle supports report checking, but it is not enough for a full rerun.

## Verification Command

Use the checker before sharing the folder or zip:

```bash
python scripts/check_artifacts.py --profile final-full
```

For the local GitHub clone, use:

```bash
python scripts/check_artifacts.py --profile git-evidence
```
