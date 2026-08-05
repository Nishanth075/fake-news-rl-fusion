# Final Statistical Reporting Change Log

No model training, fine-tuning, prediction regeneration, threshold alteration, reward change, or checkpoint overwrite was performed.

## Files Read
- `data/final_modality_outputs/test_outputs.csv`
- `outputs/metrics/final_explainability_faithfulness_n300.csv`
- `outputs/metrics/final_rl_controller_seed_comparison.csv`
- `outputs/metrics/final_rl_fusion_test_predictions.csv`
- `outputs/metrics/final_seed_significance_summary.csv`
- `outputs/metrics/final_supervised_fusion_matched_test_predictions.csv`

## Files Created
- `outputs/metrics/final_statistical_source_manifest.csv`
- `outputs/metrics/final_mcnemar_results.csv`
- `outputs/metrics/final_mcnemar_results.json`
- `outputs/metrics/final_seed_aggregated_summary.csv`
- `outputs/metrics/final_seed_pairwise_summary.csv`
- `outputs/metrics/final_explainability_faithfulness_summary.csv`
- `outputs/metrics/final_statistical_reporting_change_log.md`

## Existing Files Modified
- None. Original prediction and metric source files were not overwritten.

## Number Checks
| Item | Original thesis-reported number | Recomputed/reported number | Changed? |
|---|---:|---:|---|
| Headline RL vs equal macro-F1 delta | 0.00027123141882823276 | pending paired recomputation if prediction files restored | not changed here |
| RL seed mean macro F1 | 0.8593558512608466 | 0.8593558512608466 | unchanged |
| RL seed sample std macro F1 | not consistently reported | 0.007185796876341226 | new reporting detail |
| Image salient comprehensiveness | 0.12595064888397853 | 0.1259506488839785 | unchanged |
| Image random comprehensiveness | 0.18579854875802992 | 0.1857985487580299 | unchanged |
| Image faithfulness margin | not explicitly reported | -0.0598478998740514 | new negative margin; image test does not pass |
| Text salient vs least margin | not explicitly reported | 0.09305998343663921 | new positive margin |
| Text salient vs random margin | not explicitly reported | 0.0366740846719283 | new positive margin |

## Recommended Thesis Sections Requiring Updates
- Abstract: avoid any statement that RL is statistically superior to fixed fusion.
- Section 7.11: report image Grad-CAM faithfulness as not supported by the deletion test; keep text saliency as supported against least/random controls.
- Section 7.12: include paired McNemar only after prediction files are present and aligned by sample ID.
- Section 7.13: report RL seed mean, sample standard deviation, min, max, median, and win/loss counts.
- Section 8.4: state that the selected RL run is competitive but the fixed-fusion advantage is marginal/unstable.
- Section 8.5: note image explainability faithfulness limitation explicitly.
- Table 7.2: keep threshold protocol columns clear; do not mix default and validation-selected thresholds silently.
- Table 7.4: add faithfulness pass/fail and margins.
- Figure 7.12: annotate that equal fusion and controller scores are fixed references, not independent seeded runs.

## Recommended Replacement Wording
"The selected RL run achieved competitive performance, but its advantage over fixed fusion was marginal and must be interpreted together with paired significance testing and seed-level instability."

## Validation Checks
- Paired prediction validation: blocked locally because one or more prediction/source CSV files are missing.
  - rl_full_state_adaptive_fusion vs equal_fusion: missing file: outputs/metrics/final_rl_fusion_test_predictions.csv; missing file: data/final_modality_outputs/test_outputs.csv
  - rl_full_state_adaptive_fusion vs confidence_weighted_fusion: missing file: outputs/metrics/final_rl_fusion_test_predictions.csv; missing file: data/final_modality_outputs/test_outputs.csv
  - rl_full_state_adaptive_fusion vs matched_supervised_mlp_fusion: missing file: outputs/metrics/final_rl_fusion_test_predictions.csv; missing file: outputs/metrics/final_supervised_fusion_matched_test_predictions.csv
- CSV outputs were written from saved metrics/prediction sources only.
- No training script was called by this audit.
- No original prediction file was overwritten.
