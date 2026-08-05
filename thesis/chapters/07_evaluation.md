# Chapter 7: Evaluation of the Adaptive Fusion Framework

## 7.1 Introduction

This chapter evaluates whether the proposed framework achieved the research objectives defined in Chapter 1 and operationalized through the approach in Chapter 4. The evaluation focuses on classification performance, adaptive fusion behaviour, reliability feature contribution, robustness, and explainability. The results are reported cautiously because the final dataset is derived from one benchmark source and the multi-seed evaluation showed variation in reinforcement learning performance.

## 7.2 Evaluation Strategy

The evaluation was designed around five questions. First, the unimodal image and text branches were evaluated separately to determine the strength of each modality. Second, deterministic fusion baselines were compared with the proposed reinforcement learning fusion model. Third, ablation experiments tested whether reliability-aware state features improved the fusion controller. Fourth, policy and robustness analyses examined whether the model changed modality weights in meaningful ways. Fifth, explainability outputs were inspected to verify that image, text, and fusion-level explanations were generated.

The final evaluation evidence-flow diagram in this chapter makes these tests explicit. It is included to avoid presenting the final macro F1 score in isolation and to show how each examiner-facing question is connected to an artifact-backed test.

## 7.3 Experimental Setup

The implementation used Python, PyTorch, Transformers, pandas, scikit-learn, and image-processing libraries. Experiments were executed using Google Colab GPU runtime and Google Drive storage. The image branch used a pretrained ResNet-based model, and the text branch used a DistilBERT-based model. The fusion layer operated on low-dimensional reliability features rather than raw image or text embeddings, which kept the adaptive controller lightweight compared with the unimodal encoders.

## 7.4 Dataset and Splits

The experiments used the Fakeddit multimodal dataset. The full prepared metadata contained 682,661 rows after label standardization. For the final computable experiment, 28,000 balanced image-text samples were requested. Due to unavailable or failed image downloads, 26,471 samples were available for final multimodal evaluation. The final available splits contained 18,893 training rows, 3,798 validation rows, and 3,780 test rows.

The original Fakeddit two-way label was converted into the project label standard, where 0 represents real content and 1 represents fake content. Samples without available images were excluded from the final multimodal subset rather than being synthetically filled.

## 7.5 Evaluation Metrics

The main classification metrics were macro F1, accuracy, balanced accuracy, precision, recall, confusion matrix, and ROC-AUC. Macro F1 was treated as the primary metric because the task involves two classes and the evaluation should not rely only on overall accuracy. ROC-AUC was used to evaluate ranking quality from predicted probabilities. For policy analysis, average image weight, average text weight, action distribution, and group-wise metrics were also reported.

## 7.6 Unimodal Branch Results

The image-only branch achieved a test macro F1 of 0.7769 and test accuracy of 0.7791. The text-only branch achieved a test macro F1 of 0.8388 and test accuracy of 0.8389. These results show that the text branch was stronger than the image branch on the final test split. However, the image branch still contributed useful information because fusion methods improved over the text-only baseline.

The text branch also showed a visible overfitting gap: the training macro F1 was approximately 0.9882, while the test macro F1 was 0.8388. The model checkpoint was selected using validation performance, so the reported test result remains a held-out evaluation, but the gap indicates that stronger regularization or validation-loss-based selection should be considered in future work.

## 7.7 Baseline and Fusion Results

The final method comparison is summarized in Table 7.1 and visualized through the performance dashboard, confusion-matrix panel, and macro-F1 comparison figure. Image-only and text-only baselines were weaker than multimodal fusion. Equal fusion and confidence-weighted fusion both achieved test macro F1 of 0.8673. The proposed RL adaptive fusion model achieved test macro F1 of 0.8676 in the main final run. With validation-selected threshold tuning, RL adaptive fusion achieved test macro F1 of 0.8685.

The improvement over the strongest fixed fusion baseline was modest. Therefore, the result should be interpreted as competitive performance with a small main-run improvement, rather than as a large statistical superiority claim. The additional contribution of the proposed method is that it provides sample-level fusion actions and modality weights, which fixed fusion baselines do not provide.

The matched supervised MLP fusion baseline reached 0.8547 macro F1 at the default threshold and 0.8565 macro F1 after threshold tuning. This fairer comparison used the matched supervised fusion evidence files and shows that the adaptive RL fusion result remained stronger in the final reported run.

## 7.8 Ablation Study

The ablation study tested three reinforcement learning state representations and compared them with simple same-state controller baselines. The probabilities-only state achieved test macro F1 of 0.8523. Adding prediction confidence improved the score to 0.8590. The full reliability state, which included probability, confidence, quality, disagreement, confidence difference, and quality difference, achieved 0.8673. This pattern indicates that the reliability-aware features contributed meaningfully to the fusion controller.

## 7.9 Policy Analysis and Reliability Behaviour

The final policy analysis, including the weight-behaviour and action-distribution diagrams, showed that the RL fusion model used an average image weight of 0.6594 and an average text weight of 0.3406 on the test split. This indicates that the policy often assigned a larger contribution to the image branch in the selected final run. However, the action distribution also showed that the model did not use only one fixed weighting pattern.

When image and text branch predictions agreed, macro F1 was 0.9110. When they disagreed, macro F1 decreased to 0.7156, and the weights became more balanced, with average image weight 0.4987 and average text weight 0.5013. This behaviour is important because disagreement samples are harder and require more careful fusion decisions.

## 7.10 Robustness Analysis

Robustness tests were used to examine whether the fusion policy changed modality weights when modality quality or probabilities were corrupted. Under image quality degradation, the model reduced image weight as the degradation level increased, as shown in Figure 7.4. At degradation levels 0.25, 0.50, and 0.75, the image-weight changes were -0.0164, -0.0632, and -0.1283, respectively. Macro F1 decreased only slightly from 0.8671 to 0.8658 across these image-quality degradation levels.

The text-quality degradation behaviour was less consistent. In the tested configuration, the model did not always adapt in the expected direction for text-quality drops. This limitation is discussed in Chapter 8, and it suggests that text reliability estimation requires stronger validation.

## 7.11 Explainability Evaluation

The explainability component generated qualitative explanation artifacts and a deletion-based faithfulness check. For the image branch, Grad-CAM heatmaps were generated to highlight image regions influencing the prediction. For the text branch, token saliency files were generated to identify influential text tokens. For the fusion decision, the output included selected action, image weight, text weight, final probability, and final prediction.

A larger faithfulness check was conducted on 300 final test samples. Text salient deletion produced a larger probability drop than least-salient and random deletion, which supports the text saliency ranking. Image salient deletion did not outperform random deletion in this metric, so the thesis treats image Grad-CAM as qualitative visual evidence rather than a fully validated faithfulness result.

## 7.12 Seed Stability and Statistical Considerations

A small multi-seed check and a controller-baseline seed comparison were conducted for the RL fusion controller. The main seed 42 run achieved a small improvement over equal fusion. However, additional seeds showed lower RL macro F1 than the equal-fusion baseline. This indicates that the RL controller was sensitive to random seed and training conditions. The thesis therefore avoids claiming statistically proven superiority over equal fusion. The safer conclusion is that the proposed model demonstrates adaptive and explainable fusion with competitive performance, while training stability remains a limitation.

## 7.13 Discussion of Findings

The evaluation supports the usefulness of multimodal fusion because fusion methods outperformed both image-only and text-only baselines. It also supports the value of reliability-aware state features because the full state outperformed reduced ablation settings. The adaptive fusion model provided interpretable modality weights and action records, which directly addressed the explainability aspect of the research problem.

At the same time, the performance difference between RL adaptive fusion and strong fixed fusion was small. This suggests that the final data split contains strong complementarity between image and text predictions that can already be captured by simple fusion. The proposed framework is therefore best defended as an explainable adaptive fusion framework with competitive performance and modest main-run improvement, not as a method that universally dominates fixed fusion.


## 7.14 External Generalization Case Study

An additional external case study was conducted using the OpenFake Reddit test split. This dataset was not used for training or model selection. A balanced external sample of 3,780 image-text posts was exported, matching the scale of the final Fakeddit test split. The purpose of this experiment was to examine whether the Fakeddit-trained models transferred directly to a different multimodal data source.

The zero-shot OpenFake result showed strong domain shift. The image-only branch achieved weak external performance, and the text branch assigned very high fake-class probabilities to almost all samples at the default 0.5 threshold. However, the text branch retained ranking signal: after selecting a threshold on 20 percent of OpenFake and applying it unchanged to the remaining 80 percent holdout subset, text-only performance reached 0.8589 macro F1. In contrast, equal fusion, reliability-weighted fusion, and RL adaptive fusion remained near 0.49 macro F1 after the same calibration protocol.

This finding is important because it separates ranking transfer from calibration transfer. The DistilBERT text representation carried transferable signal, but its probabilities were poorly calibrated under the OpenFake distribution. The image branch and the Fakeddit-trained fusion controller did not transfer well in this zero-shot setting. Therefore, external deployment would require calibration, domain adaptation, or additional external training data.

## 7.15 Summary

This chapter evaluated the proposed framework using final dataset statistics, unimodal results, fusion comparisons, ablation, policy analysis, robustness tests, explainability artifacts, and seed stability checks. The results show that the framework achieved competitive performance and provided sample-level adaptive explanations. The next chapter concludes the thesis by summarizing contributions, objective achievement, limitations, and future work.


