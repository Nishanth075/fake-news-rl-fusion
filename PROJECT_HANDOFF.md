# Project Handoff: Fake News RL Fusion

## Project Summary

This project implements and evaluates an explainable multimodal fake news detection framework using image-text fusion. The system uses separate image and text classifiers, computes reliability-aware state features, and compares multiple fusion strategies including deterministic baselines, supervised learned controllers, and a reinforcement-learning-style adaptive fusion controller.

The current defensible research claim is:

> The proposed framework provides reliability-aware, sample-level adaptive and explainable image-text fusion for fake news detection. It achieves competitive performance against strong fixed-fusion baselines, outperforms simple learned same-state controllers on average across tested seeds, and provides interpretable modality-weight decisions with quantitative text-saliency faithfulness evidence.

Avoid claiming that RL significantly outperforms equal fusion. The headline RL-vs-equal-fusion difference is small and not statistically significant.

## Repository / Environment

- Repository: `fake-news-rl-fusion`
- Main branch: `main`
- Local PC path used in Codex: `C:\Users\acer\Documents\GitHub\fake-news-rl-fusion`
- Colab/Drive path used for experiments: `/content/drive/MyDrive/fake-news-rl-fusion`
- Main dataset: Fakeddit-derived multimodal subset
- Label convention:
  - `0 = Real`
  - `1 = Fake`

## Major Completed Implementation Stages

1. Dataset preparation and Fakeddit label standardization.
2. Image downloading and available-split filtering.
3. Image branch training/inference using ResNet-18.
4. Text branch training/inference using DistilBERT.
5. Reliability feature generation.
6. Deterministic baseline fusion evaluation.
7. RL adaptive fusion training/evaluation.
8. Supervised MLP fusion baseline.
9. Same-state logistic regression and decision tree controller baselines.
10. Threshold tuning.
11. Ablation analysis.
12. Policy analysis.
13. Robustness analysis.
14. Headline McNemar significance test.
15. Multi-seed RL stability against equal fusion.
16. Multi-seed RL comparison against best same-state controller.
17. Explainability generation: Grad-CAM, token saliency, fusion weights.
18. Explainability faithfulness testing with deletion/comprehensiveness metrics.
19. Final result export tables.
20. Reproducible notebooks for viva/demo.

## Key Dataset Results

Final available image-text subset:

| Split | Rows |
|---|---:|
| Train | 18,893 |
| Validation | 3,798 |
| Test | 3,780 |
| Total | 26,471 |

Image download result:

- Requested: 28,000
- Available/downloaded: 26,471
- Failed: 1,529

Evidence files:

- `outputs/metrics/fakeddit_stats.json`
- `data/final_splits_available/image_download_stats.json`

## Key Model / Method Results

| Method | Test Accuracy | Test Macro F1 | Notes |
|---|---:|---:|---|
| Image-only ResNet | 0.7791 | 0.7769 | Unimodal image branch |
| Text-only DistilBERT | 0.8389 | 0.8388 | Unimodal text branch |
| Equal fusion | 0.8675 | 0.8673 | Strong fixed baseline |
| Confidence-weighted fusion | 0.8675 | 0.8673 | Strong fixed baseline |
| Reliability-weighted fusion | 0.8614 | 0.8612 | Deterministic reliability rule |
| Supervised MLP fusion | 0.8500 | 0.8500 | Earlier supervised fusion |
| Matched supervised MLP fusion | 0.8548 | 0.8547 | Fairer matched-budget MLP |
| Logistic regression controller | 0.8471 | 0.8470 | Same 9-state features |
| Decision tree controller depth 3 | 0.8524 | 0.8523 | Same 9-state features |
| Decision tree controller depth 5 | 0.8529 | 0.8529 | Best simple controller |
| Unlimited decision tree controller | 0.8474 | 0.8473 | Same 9-state features |
| RL adaptive fusion, seed 42 | 0.8677 | 0.8676 | Main proposed model |
| RL adaptive fusion, threshold tuned | 0.8685 | 0.8685 | Validation-selected threshold |
| RL adaptive fusion, 3-seed mean | ~0.8594 | 0.8594 | Seeds 42, 7, 13 |

## Statistical Findings

### Headline RL vs Equal Fusion

Evidence:

- `outputs/metrics/final_headline_significance.json`
- `outputs/metrics/final_headline_significance_summary.csv`

Result:

- Equal fusion macro F1: `0.8673332160630083`
- RL macro F1: `0.8676044474818365`
- Delta macro F1: `+0.00027123141882823276`
- McNemar p-value: `1.0`

Interpretation:

- RL is slightly higher in the main run.
- The improvement over equal fusion is not statistically significant.
- Use careful wording: competitive performance with adaptive/interpretable decisions.

### RL vs Same-State Controller Across Seeds

Evidence:

- `outputs/metrics/final_rl_controller_seed_comparison.csv`
- `outputs/metrics/final_rl_controller_seed_comparison.json`

Best same-state controller:

- `state_decision_tree_depth_5`
- Test macro F1: `0.8528505686350071`

RL across three seeds:

- Mean macro F1: `0.8593558512608466`
- Std: `0.007185796876341226`
- Mean delta over controller: `+0.006505282625839419`

Per seed:

| Seed | RL Macro F1 | Delta vs Controller | McNemar p-value |
|---:|---:|---:|---:|
| 42 | 0.8676044474818365 | +0.014753878846829438 | 1.1307691767152769e-05 |
| 7 | 0.8544529826427305 | +0.0016024140077234295 | 0.5323087760278421 |
| 13 | 0.8560101236579725 | +0.003159555022965388 | 0.12634707581392135 |

Interpretation:

- RL is consistently above the best same-state controller across tested seeds.
- The gain is strongest and statistically significant in seed 42.
- Other seeds are positive but not statistically significant.
- Safe claim: RL outperforms simple same-state controllers on average, but the magnitude is seed-sensitive.

## Explainability Findings

Generated explanation types:

- Image Grad-CAM heatmaps.
- Text token saliency CSVs.
- Fusion action, image weight, and text weight explanations.

Evidence:

- `outputs/metrics/final_explainability_summary.json`
- `outputs/explainability/final/`

### Faithfulness Evaluation

Latest faithfulness test uses `300` samples.

Evidence:

- `outputs/metrics/final_explainability_faithfulness_n300.json`
- `outputs/metrics/final_explainability_faithfulness_n300.csv`

Result:

| Explanation Test | Mean Comprehensiveness |
|---|---:|
| Image salient deletion | 0.12595064888397853 |
| Image random deletion | 0.18579854875802992 |
| Text salient token masking | 0.1561762981209904 |
| Text least-salient token masking | 0.06311631468435129 |
| Text random token masking | 0.11950221344906216 |

Interpretation:

- Text saliency has positive quantitative faithfulness support:
  - salient > random > least-salient.
- Image Grad-CAM is useful qualitatively, but deletion-based faithfulness is weak:
  - random image deletion caused a larger drop than Grad-CAM salient deletion.
- Do not claim quantitatively validated image explanation faithfulness.

## Policy / Robustness Findings

RL policy behaviour:

- Average image weight: `0.6594`
- Average text weight: `0.3406`
- Agreement macro F1: `0.9110`
- Disagreement macro F1: `0.7156`

Evidence:

- `outputs/metrics/final_rl_policy_analysis.json`
- `outputs/metrics/final_rl_action_details.csv`

Robustness:

- Image quality degradation reduced image weight in the expected direction.
- Larger image degradation produced larger image-weight reduction.

Evidence:

- `outputs/robustness/final_robustness_summary.csv`
- `outputs/robustness/final_robustness_details.json`

## Pretrained vs Trained Models

Pretrained backbones:

- ResNet-18 with ImageNet weights.
- DistilBERT `distilbert-base-uncased`.

Models trained/fine-tuned in this project:

- Image classifier: ResNet-18 + binary head, fine-tuned on Fakeddit image data.
- Text classifier: DistilBERT + binary head, fine-tuned on Fakeddit text data.
- RL adaptive fusion controller.
- Supervised MLP fusion baseline.
- Logistic regression same-state controller.
- Decision tree same-state controllers.

Computed components, not trained models:

- Equal fusion.
- Confidence-weighted fusion.
- Reliability-weighted fusion.
- Reliability feature extractor.
- Grad-CAM.
- Token saliency.
- Threshold tuning.

## Evaluation Metrics Used

Classification:

- Accuracy.
- Macro F1.
- Weighted F1.
- Precision macro.
- Recall macro.
- Balanced accuracy.
- ROC-AUC.
- Confusion matrix.

Training:

- Train loss.
- Validation loss.
- Validation macro F1.

Statistical / stability:

- McNemar p-value.
- Mean macro F1 across seeds.
- Standard deviation across seeds.
- Delta macro F1.

Policy / robustness:

- Action distribution.
- Average image weight.
- Average text weight.
- Agreement/disagreement macro F1.
- Image/text quality group macro F1.
- Delta quality.
- Delta modality weight.

Explainability:

- Comprehensiveness / deletion drop.
- Salient vs random deletion.
- Salient vs least-salient deletion.

## Important Commands

Run from Colab after mounting Drive:

```bash
%cd /content/drive/MyDrive/fake-news-rl-fusion
```

Pull latest code:

```bash
!git fetch origin
!git merge origin/main --no-edit
```

Controller baselines:

```bash
!python scripts/evaluate_controller_baselines.py --config configs/final_controller_baselines.yaml
```

Headline RL-vs-equal-fusion McNemar test:

```bash
!python scripts/run_headline_significance.py --config configs/final_headline_significance.yaml
```

Explainability faithfulness, N=300:

```bash
!python scripts/run_explainability_faithfulness.py --config configs/final_explainability_faithfulness.yaml
```

RL-vs-controller multi-seed comparison:

```bash
!python scripts/compare_rl_controller_seeds.py --config configs/final_rl_controller_seed_comparison.yaml
```

Seed significance against equal fusion:

```bash
!python scripts/run_seed_significance.py --config configs/final_seed_significance.yaml
```

## Important Output Files

Dataset:

- `outputs/metrics/fakeddit_stats.json`
- `data/final_splits_available/image_download_stats.json`

Model outputs:

- `outputs/metrics/final_image_outputs_metrics.json`
- `outputs/metrics/final_text_outputs_metrics.json`
- `data/final_modality_outputs/test_image_outputs.csv`
- `data/final_modality_outputs/test_text_outputs.csv`
- `data/final_modality_outputs/test_outputs.csv`

Baselines and fusion:

- `outputs/metrics/final_baseline_results.csv`
- `outputs/metrics/final_baseline_summary.json`
- `outputs/metrics/final_rl_fusion_metrics.json`
- `outputs/metrics/final_rl_fusion_test_predictions.csv`
- `outputs/metrics/final_supervised_fusion_matched_metrics.json`
- `outputs/metrics/final_controller_baseline_summary.csv`
- `outputs/metrics/final_controller_baseline_details.json`

Statistical tests:

- `outputs/metrics/final_headline_significance.json`
- `outputs/metrics/final_headline_significance_summary.csv`
- `outputs/metrics/final_seed_significance_summary.csv`
- `outputs/metrics/final_seed_significance_details.json`
- `outputs/metrics/final_rl_controller_seed_comparison.csv`
- `outputs/metrics/final_rl_controller_seed_comparison.json`

Explainability:

- `outputs/metrics/final_explainability_summary.json`
- `outputs/metrics/final_explainability_faithfulness_n300.json`
- `outputs/metrics/final_explainability_faithfulness_n300.csv`
- `outputs/explainability/final/`

Tables:

- `outputs/tables/final_method_comparison.csv`
- `outputs/tables/final_results_summary.json`
- `outputs/tables/final_stage_status.csv`

## Reproducibility Notebooks

- `notebooks/01_data_preparation.ipynb`
- `notebooks/02_image_model_training.ipynb`
- `notebooks/03_text_model_training.ipynb`
- `notebooks/04_baselines_and_reliability.ipynb`
- `notebooks/05_rl_adaptive_fusion.ipynb`
- `notebooks/06_explainability_and_final_results.ipynb`

## Known Caveats / Safe Wording

Do not say:

- "RL significantly outperforms equal fusion."
- "Image Grad-CAM explanations are quantitatively validated."
- "The method is proven generalizable across domains."
- "The RL controller is full long-horizon deep RL."

Safe wording:

- "RL adaptive fusion is competitive with strong fixed fusion baselines."
- "RL outperforms simple same-state learned controllers on average across tested seeds."
- "The controller is a lightweight contextual RL / bandit-style adaptive fusion controller."
- "Text saliency has positive deletion-test faithfulness support."
- "Image Grad-CAM is retained mainly as qualitative visual evidence."
- "Generalizability beyond the Fakeddit-derived setting remains future work."

## Recent Important Commits

- `230f42c` Add controller significance and faithfulness evaluations
- `e3da0ec` Fix image faithfulness binary scoring
- `b5a049f` Improve faithfulness text masking and sample size
- `df56645` Add RL controller seed comparison

## Current Local Git Notes

During thesis drafting, local untracked folders may appear:

- `thesis/`
- `tmp_pdf_pages/`

These were not part of the core implementation commits unless intentionally committed later. Be careful not to mix thesis draft artifacts with code/evaluation commits unless that is the intended task.

## Recommended Next Steps

1. Update Chapter 7 evaluation with:
   - same-state controller comparison,
   - RL-vs-controller seed comparison,
   - headline McNemar result,
   - N=300 explainability faithfulness result.
2. Update conclusion wording to avoid overclaiming RL superiority over equal fusion.
3. Update thesis tables/figures to include the new evidence files.
4. Optionally rerun final result export if the exporter is extended to include the newest artifacts.
5. Prepare viva defense points around:
   - why contextual RL/controller rather than full sequential RL,
   - why equal fusion remains a strong baseline,
   - why RL still contributes adaptive/interpretable decisions,
   - why text saliency has stronger quantitative support than image Grad-CAM.

## Reproducibility Artifact Status

A normal GitHub clone is expected to contain source code, configs, notebooks, thesis source, and lightweight evidence files. It is not expected to contain the full downloaded image corpus, trained checkpoints, or large explanation folders.

Use the artifact checker from the repository root:

```bash
python scripts/check_artifacts.py --profile git-evidence
python scripts/check_artifacts.py --profile final-full
```

Current interpretation:

- `git-evidence` verifies that the repository contains the code, thesis source, and small result summaries needed to inspect the reported findings.
- `final-full` verifies the separate Google Drive artifact bundle needed for full reruns and independent verification.

See `ARTIFACT_MANIFEST.md` and `REPRODUCIBILITY.md` for the exact bundle contents and Colab zip command.
