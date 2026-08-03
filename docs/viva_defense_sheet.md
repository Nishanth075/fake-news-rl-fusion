# Viva Defense Sheet

## One-Minute Project Summary

This research proposes an explainable fake-news detection framework for multimodal posts. The system first trains separate image and text classifiers, then learns an adaptive reinforcement-learning fusion policy that selects how much to trust each modality for each sample. The fusion state includes prediction probabilities, confidence, image/text quality indicators, disagreement, confidence difference, and quality difference. The final output includes classification metrics, policy weights, ablation, robustness checks, and image/text explanations.

## Main Novelty

**Question:** What is new compared with normal multimodal fake-news detection?

**Answer:** The novelty is not only combining image and text. The contribution is a sample-level adaptive fusion policy. Fixed fusion uses the same weighting rule for every post, but this framework uses reliability-aware state features and chooses a fusion action per sample. That means the model can change image/text weights depending on confidence, quality, and disagreement.

**Evidence:** Final RL adaptive fusion test macro F1 was `0.8676`, and threshold-tuned RL reached `0.8685`. The best fixed fusion baseline was around `0.8671` after threshold tuning. The gain is small, so the main claim is competitive/slightly improved performance plus adaptive explainability.

## Why Reinforcement Learning?

**Question:** Why use RL instead of simple concatenation or MLP fusion?

**Answer:** The fusion task can be formulated as an action-selection problem: for each sample, the agent observes reliability features and selects a fusion weight action. The reward is based on whether the fused prediction is correct. This makes the fusion decision explicit and interpretable.

**Evidence and caveat:** The first supervised MLP fusion result reached `0.8510` tuned macro F1, while RL adaptive fusion reached `0.8685` tuned macro F1. However, this comparison used different training budgets, so I should not overclaim from it. A matched-budget supervised MLP config is included for a fair rerun, and the main defended comparison remains RL against deterministic fusion baselines plus the ablation and policy analysis.

## Baseline Comparison

**Question:** Did you compare against strong baselines?

**Answer:** Yes. The evaluation includes image-only, text-only, equal fusion, confidence-weighted fusion, reliability-weighted fusion, supervised MLP fusion, ablations of the RL state, and the full RL adaptive fusion model.

Final test macro F1:

| Method | Macro F1 |
|---|---:|
| Image only | `0.7769` |
| Text only | `0.8388` |
| Reliability weighted fusion | `0.8612` |
| Equal fusion | `0.8673` |
| Confidence weighted fusion | `0.8673` |
| RL adaptive fusion | `0.8676` |
| RL adaptive fusion, threshold tuned | `0.8685` |

**Careful wording:** The improvement is modest. The defensible conclusion is that RL fusion is competitive and slightly better, while giving adaptive policy explanations that fixed baselines do not provide.

## Ablation Defense

**Question:** How do you prove the reliability features matter?

**Answer:** I tested RL fusion with three state settings.

| State | Test Macro F1 |
|---|---:|
| Probabilities only | `0.8523` |
| Prediction + confidence | `0.8590` |
| Full reliability state | `0.8673` |

This shows that adding confidence improves over raw probabilities, and the full reliability state gives the best performance. Therefore, the added reliability features are not decorative; they contribute to the fusion decision.

## Policy Explanation

**Question:** How can you explain the fusion decision?

**Answer:** For each sample, the system records the selected fusion action, image weight, text weight, final probability, and final prediction. At the global level, the final RL policy used average image weight `0.6594` and text weight `0.3406`. This means the policy often trusted the image branch strongly, but still changed weights depending on sample state.

**Useful examples from policy analysis:**

- When image and text predictions agreed, macro F1 was `0.9110`.
- When they disagreed, macro F1 dropped to `0.7156`, and weights became more balanced: image `0.4987`, text `0.5013`.
- This is important because disagreement cases are naturally harder and need adaptive handling.

## Explainability

**Question:** Did you implement explainability?

**Answer:** Yes. The framework includes three explanation levels:

1. Image branch: Grad-CAM heatmaps show visual regions influencing the image prediction.
2. Text branch: token saliency shows influential text tokens.
3. Fusion level: selected action, image/text weights, and final probability show how the modalities were combined.

**Evidence:** The final explainability run generated explanations for `16` samples in `outputs/explainability/final`, with the committed summary in `outputs/metrics/final_explainability_summary.json`.

## Robustness

**Question:** How does the model behave if one modality quality drops?

**Answer:** The robustness test applies controlled corruptions to image/text quality and prediction probabilities. The clearest adaptation is for image degradation: as image quality drops, the model reduces image weight.

Examples:

| Corruption | Level | Delta Target Weight | Macro F1 |
|---|---:|---:|---:|
| Image quality drop | `0.25` | `-0.0164` | `0.8671` |
| Image quality drop | `0.50` | `-0.0632` | `0.8665` |
| Image quality drop | `0.75` | `-0.1283` | `0.8658` |

**Caveat:** Text-quality degradation did not always move in the expected direction. The explanation is that normalized text quality was high for almost all selected samples, and the text branch stayed confident. This should be presented as a limitation, not hidden.

## Dataset and Data Filtering

**Question:** What dataset did you use and how did you handle missing images?

**Answer:** The dataset is Fakeddit multimodal metadata. The full prepared metadata contains `682,661` rows. For the final computable experiment, a balanced subset was selected and images were downloaded. Out of `28,000` requested final samples, `26,471` had available downloaded images.

Final available rows:

- Train: `18,893`
- Validation: `3,798`
- Test: `3,780`

Missing images were not synthetically filled; unavailable images were excluded from the final multimodal subset. The final results therefore evaluate only samples where both modalities are available.

## Label Standardization

**Question:** How are fake/real labels represented?

**Answer:** The original Fakeddit two-way label was converted into the project standard: `0 = Real`, `1 = Fake`. This conversion is recorded in `outputs/metrics/fakeddit_stats.json`.

## Text Branch Overfitting

**Question:** The text branch has high train performance and lower test performance. Is it overfitting?

**Answer:** Yes, the text branch shows an overfitting gap: train macro F1 is around `0.988`, while test macro F1 is around `0.839`. This is not hidden; it is visible in the final metrics. The checkpoint was selected using validation performance, not training performance, so the final test result is still a held-out evaluation. A stronger future version should use more aggressive regularization, fewer epochs, or validation-loss early stopping for the text branch.

## Reward Design

**Question:** Why is the RL reward only correct/incorrect?

**Answer:** I used a simple binary reward to make the fusion policy directly optimize classification correctness and keep the action interpretation clear. A limitation is that it does not distinguish a confident correct prediction from a marginal one. A future improvement is a margin-based or log-loss-based reward that also encourages calibration.

## Quality Heuristic Validation

**Question:** How do you know the quality features are meaningful?

**Answer:** The image quality features are hand-designed proxies using blur, contrast, entropy, and related statistics. They are reasonable but not perfect. The policy analysis gives partial support: low image-quality samples had lower macro F1 (`0.8394`) than high-quality samples (`0.8652`). I should present these as heuristic reliability features, not as ground-truth quality labels.

## Action Space

**Question:** Why only seven fixed fusion actions?

**Answer:** The discrete action space is a simplicity-versus-expressiveness tradeoff. Seven actions make the policy interpretable because each action maps to a clear image/text weight pair. A continuous policy or finer action grid may improve performance, but would be less simple to explain and was left as future work.

## Complexity

**Question:** Did the adaptive fusion make the system too complex?

**Answer:** The computationally expensive parts are the pretrained image and text encoders. The RL fusion network is small and runs on a low-dimensional state vector with only 9 features and a small action space. So the added fusion controller is lightweight compared with ResNet and DistilBERT inference.

**Careful wording:** I did not claim a formal big-O complexity improvement over all baselines. I can defend that the added adaptive fusion overhead is small in practical terms because it operates after unimodal predictions are already computed.

## Generalizability

**Question:** How do you prove it generalizes to the real world?

**Answer:** The final evaluation uses held-out train/validation/test splits and robustness tests, but it is still within one dataset domain. Therefore, the correct claim is within-dataset generalization, not universal real-world generalization.

**Future work:** Test on another multimodal fake-news dataset, cross-domain events, or real social media streams to measure external generalization.

## Why Improvement Is Small

**Question:** If the improvement is small, why is the work useful?

**Answer:** The baseline fusion methods are already strong because image and text predictions are highly complementary. The RL model still performs slightly better, but the bigger contribution is that it gives an adaptive, inspectable decision process. It tells us not only the final label, but also how modality trust changed for each sample.

**Important limitation:** The raw RL edge over equal fusion is very small (`0.8676` vs `0.8673`). Therefore, I should describe it as competitive/slightly improved, and strengthen the claim with multi-seed runs and paired significance testing rather than relying on a single-seed headline.

## Single-Seed Limitation

**Question:** Are the results statistically reliable?

**Answer:** The current committed final run is seed `42`, so it is a single-seed result. That is a limitation. To address this, I added a multi-seed comparison script that reruns RL fusion over multiple seeds and compares it with equal fusion using an exact McNemar test on paired test predictions. If time allows before final defense, I will report mean plus/minus standard deviation and the paired p-values.

## Best Final Claim

Use this wording:

> The proposed RL-based adaptive fusion framework achieved competitive and slightly improved macro F1 compared with fixed fusion baselines, while adding sample-level interpretability through fusion actions, modality weights, Grad-CAM image explanations, and text token saliency. Ablation confirms that reliability-aware state features improve the learned fusion policy.

Avoid this wording:

> The model dramatically outperforms all baselines.

That is too strong for the final numbers.

## Files to Show in Viva

- `notebooks/01_data_preparation.ipynb`
- `notebooks/02_image_model_training.ipynb`
- `notebooks/03_text_model_training.ipynb`
- `notebooks/04_baselines_and_reliability.ipynb`
- `notebooks/05_rl_adaptive_fusion.ipynb`
- `notebooks/06_explainability_and_final_results.ipynb`
- `outputs/tables/final_method_comparison.csv`
- `outputs/metrics/final_threshold_tuning_summary.csv`
- `outputs/metrics/final_rl_policy_analysis.json`
- `outputs/metrics/final_explainability_summary.json`
