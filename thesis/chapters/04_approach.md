# Chapter 4: RL-Based Adaptive Image-Text Fusion Approach

## Introduction

This chapter presents the conceptual approach proposed in this research. The purpose of the approach is to extend multimodal fake news detection by replacing a fixed image-text fusion rule with an adaptive and explainable fusion policy. The approach treats fusion as an action-selection problem in which a reinforcement learning controller observes reliability-related features and selects how much influence should be assigned to the image and text branches.

## Hypothesis and Inspiration

The hypothesis of this research is that a reliability-aware adaptive fusion policy can provide competitive fake news detection performance while making the modality contribution more explainable at sample level. The approach is inspired by the observation that image and text evidence are not equally reliable for every social media post. A fixed fusion method applies the same rule even when one modality is uncertain, noisy, or in conflict with the other modality. An adaptive policy can instead select a fusion action based on the observed reliability state of the sample.

## Inputs and Outputs of the Proposed Extension

The inputs to the proposed extension are the outputs of the image and text branches, together with reliability indicators derived from the sample. The image branch produces an image probability and an image confidence value. The text branch produces a text probability and a text confidence value. Additional reliability indicators represent image quality, text quality, modality disagreement, confidence difference, and quality difference.

The main output is a binary fake news prediction using the label standard 0 for real and 1 for fake. The framework also outputs the final probability, selected fusion action, image weight, text weight, and explanation artifacts. These outputs make the decision more inspectable than a prediction label alone.

## Process Workflow of the Proposed Extension

The proposed workflow contains six conceptual stages, as illustrated in Figure 4.1. First, the dataset is prepared and labels are standardized. Second, the image and text branches are trained separately. Third, both branches generate prediction probabilities for each split. Fourth, reliability features are computed and merged with modality predictions. Fifth, the reinforcement learning fusion controller selects a fusion action for each sample. Sixth, the framework generates classification metrics, policy analysis, robustness results, and explanation outputs.

The process is designed to separate unimodal representation learning from adaptive decision fusion. This separation allows the fusion controller to remain lightweight, because it uses a compact reliability-aware state instead of processing raw images or raw text directly.

The methodological contribution is the replacement of a fixed post-hoc fusion rule with a sample-level controller that observes reliability evidence before assigning modality weights. The novelty comparison diagram in this chapter makes this distinction explicit by separating simple fixed fusion from the proposed reliability-aware action-selection path.

## Reliability-Aware State Representation

The reliability-aware state is the main input to the fusion controller. It contains:

1. Image prediction probability.
2. Image confidence.
3. Image quality.
4. Text prediction probability.
5. Text confidence.
6. Text quality.
7. Modality disagreement.
8. Confidence difference.
9. Quality difference.

This state representation was selected to capture both branch-level prediction strength and cross-modal conflict. The image and text probabilities represent the direct unimodal evidence. The confidence features represent prediction certainty. The quality features provide engineered reliability cues. Disagreement and difference features help the controller identify samples where the modalities may require more careful balancing.

## Adaptive Fusion Action Space

The reinforcement learning controller selects from a discrete set of fusion actions. Each action maps to a fixed pair of image and text weights. This design makes the policy easy to interpret because each selected action can be directly translated into a modality trust decision. A continuous action space may provide finer control, but the discrete action space was selected for simplicity, stability, and explainability.

## Reward Design

The reward is based on whether the fused prediction is correct. If the selected fusion action leads to a correct prediction, the controller receives a positive reward. If the prediction is incorrect, it receives a negative reward. This binary reward directly connects the policy to classification correctness.

The reward design is intentionally simple. It supports clear interpretation but does not distinguish between highly confident and marginally correct predictions. A margin-based or log-loss-based reward is therefore identified as a possible future improvement.

## Explanation Strategy

The framework provides transparent decision evidence at three levels. At the image level, Grad-CAM heatmaps highlight visual regions associated with the image prediction. At the text level, token saliency identifies influential tokens from the text branch. At the fusion level, the selected action, image weight, text weight, final probability, and final prediction show how the modalities were combined.

This transparency strategy is aligned with the research problem because it does not stop at model performance. It provides inspectable decision evidence about the modality weighting process without claiming a complete causal explanation of the final prediction.

## Positioning within the AI Body of Knowledge

The proposed extension lies at the intersection of multimodal learning, reinforcement learning, and explainable AI. The image and text branches provide unimodal deep learning representations. The fusion controller applies reinforcement learning to select a decision action based on a compact state. The explanation layer supports interpretability by exposing modality weights and branch-level saliency.

The contribution is not the invention of a new image encoder or language model. Instead, the contribution is a reliability-aware contextual-bandit-style reinforcement learning controller that makes image-text fusion adaptive, inspectable, and reproducible for fake news detection.

## Summary

This chapter presented the proposed RL-based adaptive image-text fusion approach. The approach uses separate image and text branches, a reliability-aware state representation, a discrete fusion action space, and explanation outputs. The next chapter converts this conceptual approach into a system-level analysis and design.


