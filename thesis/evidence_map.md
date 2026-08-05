# Evidence Map

This file maps thesis claims to local evidence files. It must be kept updated when new results are added.

| Claim Area | Evidence File |
|---|---|
| Full Fakeddit prepared dataset statistics | `outputs/metrics/fakeddit_stats.json` |
| Final available image subset | `data/final_splits_available/image_download_stats.json` |
| Final method comparison | `outputs/tables/final_method_comparison.csv` |
| Final results summary | `outputs/tables/final_results_summary.json` |
| Final stage completion status | `outputs/tables/final_stage_status.csv` |
| Image branch metrics | `outputs/metrics/final_image_outputs_metrics.json` |
| Text branch metrics | `outputs/metrics/final_text_outputs_metrics.json` |
| Baseline metrics | `outputs/metrics/final_baseline_results.csv` and `outputs/metrics/final_baseline_summary.json` |
| RL fusion metrics | `outputs/metrics/final_rl_fusion_metrics.json` |
| RL policy analysis | `outputs/metrics/final_rl_policy_analysis.json` |
| Matched supervised MLP fusion | `outputs/metrics/final_supervised_fusion_matched_metrics.json` |
| Threshold tuning | `outputs/metrics/final_threshold_tuning_matched_summary.csv` |
| Ablation study | `outputs/metrics/final_ablation_summary.csv` |
| Seed stability check | `outputs/metrics/final_seed_significance_summary.csv` |
| Robustness analysis | `outputs/robustness/final_robustness_summary.csv` |
| Explainability examples | `outputs/metrics/final_explainability_summary.json` and `outputs/explainability/final/` |
| Reproducible notebooks | `notebooks/01_data_preparation.ipynb` to `notebooks/06_explainability_and_final_results.ipynb` |

## Main Defensible Result Wording

The proposed framework achieved competitive performance against strong deterministic fusion baselines while providing sample-level adaptive and explainable fusion decisions. In the main final run, RL adaptive fusion achieved `0.8676044474818365` test macro F1, compared with `0.8673332160630083` for equal fusion. After validation-selected threshold tuning, RL adaptive fusion achieved `0.8684550955427615` test macro F1. The gain is modest, so the thesis should emphasize adaptive decision-making, interpretability, and reliability-aware behaviour rather than claiming large statistical superiority.
