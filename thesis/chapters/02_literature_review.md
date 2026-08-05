# Chapter 2: Developments and Challenges in Multimodal Fake News Detection

## Introduction

This chapter reviews prior work related to fake news detection, with emphasis on multimodal fake news detection and fusion strategies. The purpose of the review is to position the research gap addressed in this thesis: the need for explainable, sample-level adaptive image-text fusion. The chapter first discusses fake news detection as a social media analysis problem, then reviews text-based, image-based, and multimodal approaches. It also examines explainability and the limitations of existing fusion methods.

## Fake News Detection in Social Media

Fake news detection has been studied as a data mining and artificial intelligence problem because online misinformation can be produced and distributed at high speed. Shu et al. described fake news detection on social media as a challenging task because the problem involves news content, user behaviour, propagation patterns, and psychological factors [1]. Their survey showed that fake news cannot be treated only as a conventional text classification problem, because social media content is shaped by both the message and the surrounding context.

In social media environments, fake news may be expressed through short captions, emotionally persuasive headlines, edited or misleading images, and posts designed to exploit existing beliefs. These properties make automated detection difficult. A model may learn superficial features from a particular dataset but fail when the event, topic, or presentation style changes. Therefore, recent research has increasingly focused on multimodal and generalizable detection methods.

## Text-Based Fake News Detection

Text-based fake news detection uses textual features such as titles, captions, article content, writing style, and semantic representations. Earlier text-based systems used manually engineered linguistic features or classical machine learning methods. Later systems used neural language models to learn representations directly from text.

Transformer-based language models improved many natural language processing tasks by learning contextual word representations. BERT introduced deep bidirectional pretraining and showed that a pretrained language model could be fine-tuned for many downstream tasks with limited architectural modification [4]. DistilBERT reduced the computational cost of BERT through knowledge distillation while retaining much of its language understanding ability [5]. These models are relevant to fake news detection because they can represent short and noisy social media text more effectively than simple bag-of-words features.

However, text alone may be insufficient for multimodal posts. A caption may appear harmless while the image carries misleading context, or an image may be neutral while the text contains the false claim. Text models also risk overfitting to dataset-specific wording, topic patterns, or annotation artifacts. This limitation motivates multimodal approaches that use additional visual evidence.

## Image-Based and Multimodal Fake News Detection

Visual evidence can provide important cues in fake news detection. Images may be manipulated, reused from unrelated events, or paired with misleading captions. Deep convolutional networks are commonly used for image representation. ResNet introduced residual learning to make deep visual networks easier to optimize and has become a widely used backbone for image classification [6].

Multimodal fake news detection combines text and image information. Wang et al. proposed Event Adversarial Neural Networks (EANN) for multimodal fake news detection and emphasized the problem of event-specific features that do not transfer well to unseen events [2]. Their work showed that multimodal representations can improve fake news detection, but it also highlighted the difficulty of learning robust and generalizable features.

The Fakeddit dataset was introduced as a large-scale multimodal fake news benchmark with multiple label granularities and image-text samples [3]. It is particularly relevant to this thesis because it provides image and text information suitable for evaluating multimodal fake news detection. The dataset also exposes a practical challenge: not all image URLs remain available, so final multimodal experiments often require careful filtering and documentation of available samples.

## Fusion Strategies in Multimodal Learning

Fusion is the process of combining information from multiple modalities. In multimodal fake news detection, fusion may occur at feature level, decision level, or through attention-based interaction. Feature-level fusion combines learned image and text embeddings before classification. Decision-level fusion combines modality predictions, such as probability scores. Attention-based and similarity-aware methods try to capture relationships between modalities.

SAFE introduced a similarity-aware multimodal fake news detection approach, showing that cross-modal similarity can be useful when analysing image-text consistency [9]. Progressive Fusion Networks later argued that multimodal fake news detection should not only use deep semantic features, because shallow-level visual and textual cues may also contain useful information [10]. TRIMOON focused on inconsistency-aware fusion and reported that image-text inconsistency can introduce fusion noise if it is not handled carefully [11]. These studies support the view that the relationship between modalities matters, not only the presence of multiple modalities.

Recent research has moved further toward adaptive and reliability-aware fusion. SAMPLE introduced similarity-aware multimodal prompt learning and used similarity-aware fusion to reduce noise from unrelated cross-modal representations [12]. MMFND used multi-task learning and cross-modal correlation weighting to guide multimodal fusion [13]. DAMMFND considered domain-aware multimodal decision-making and emphasized that modality contribution can vary across domains [14]. MSAF-Net, developed for short-video fake news detection, also used similarity-guided adaptive fusion to balance unimodal and inter-modal consistency features [15]. These recent methods show that adaptive fusion remains an active research direction.

Despite these advances, many fusion methods still provide limited sample-level explanation. A fixed averaging rule is simple but applies the same weighting behaviour to every sample. Confidence-weighted fusion is more flexible, but it remains deterministic and may not fully capture richer reliability features such as image quality, text quality, and modality disagreement. Deep feature-fusion models may learn complex interactions, but their fusion decisions can be difficult to inspect. The present research therefore focuses on a lightweight reinforcement learning controller that exposes the selected action and modality weights for each sample.

## Explainability in Fake News Detection

Explainability is important in fake news detection because the output can influence user trust and decision-making. A prediction label alone may be insufficient when users need to understand why a post was classified as fake or real. In multimodal systems, explanation should ideally show evidence from image, text, and fusion levels.

Grad-CAM provides visual explanations for convolutional neural networks by highlighting image regions that influence a target prediction [8]. Token saliency methods can show which words or subword tokens influenced a text model. These explanation methods help inspect unimodal branches. However, multimodal fake news detection also needs fusion-level transparency: how much each modality contributed and what fusion decision evidence was recorded.

Recent work has also connected fake news detection with interpretability and trust. Fusion-based approaches have been proposed to improve both detection and interpretability [16], and recent explainable multimodal fake-news studies show that robustness and transparency are becoming central concerns in misinformation detection [17]. The proposed research addresses this direction by exposing the selected fusion action, image weight, text weight, final probability, and branch-level saliency artifacts. This does not fully solve the broader problem of explanation faithfulness, but it improves transparency compared with a black-box final prediction.

## Comparative Analysis of Existing Methods

Table 2.1 compares representative multimodal fake news detection methods against the properties required by this thesis. The comparison is qualitative because the cited systems use different datasets, splits, and evaluation protocols. Its purpose is to locate the methodological gap rather than to make a leaderboard-style performance claim.

**Table 2.1: Feature-level comparison of related multimodal fake news detection methods**

| Method | Adaptive fusion | Sample weights | Reliability features | Explicit action | RL policy | Branch explanation | Fusion transparency |
|---|---|---|---|---|---|---|---|
| EANN [2] | Partial | No | Event adaptation | No | No | Limited | Limited |
| SAFE [9] | Partial | No | Similarity | No | No | Limited | Limited |
| PFN [10] | Partial | No | Shallow/deep cues | No | No | Limited | Limited |
| TRIMOON [11] | Partial | No | Inconsistency | No | No | Limited | Limited |
| SAMPLE [12] | Partial | No | Similarity/prompt signal | No | No | Limited | Limited |
| MMFND [13] | Partial | No | Correlation/task signal | No | No | Limited | Limited |
| DAMMFND [14] | Partial | No | Domain signal | No | No | Limited | Limited |
| MSAF-Net [15] | Partial | No | Similarity guidance | No | No | Limited | Limited |
| This thesis | Yes | Yes | Confidence, quality, disagreement | Yes | Contextual-bandit RL | Grad-CAM/token saliency | Action and modality weights |

The comparison shows that existing methods address important aspects of multimodal fake news detection, including event generalization, similarity-aware fusion, inconsistency handling, and domain-aware decision-making. The contribution of this thesis is different: it treats late image-text fusion as a reliability-aware sample-level action-selection problem and records the selected fusion action and modality weights for every evaluated sample. The value of this contribution is not a leaderboard-style accuracy claim; it is a transparent adaptive-fusion framework that can be inspected, ablated, and statistically compared with deterministic and supervised controller baselines.

## Research Gap

The literature indicates that multimodal fake news detection can benefit from both textual and visual evidence. It also shows that modality similarity, inconsistency, cross-modal noise, and domain variation are current research concerns. However, several gaps remain. First, many systems use fixed or opaque fusion methods that do not clearly adapt modality weights at sample level. Second, reliability information such as modality confidence, quality, and disagreement is not always explicitly represented in the fusion decision. Third, explanation often focuses on branch-level evidence but does not always make the fusion process directly inspectable.

Therefore, the research gap addressed in this thesis is the lack of a transparent, reliability-aware, sample-level adaptive fusion mechanism for image-text fake news detection. This thesis proposes a reinforcement learning-based fusion controller to address this gap by selecting a fusion action for each sample and exposing the resulting modality weights. The contribution is framed as decision transparency and adaptive fusion design, not as a claim of universal predictive superiority.

## Summary

This chapter reviewed fake news detection, text-based modelling, image-based modelling, multimodal fusion, and explainability. The review showed that multimodal approaches are important but still face challenges in adaptive fusion and interpretability. The next chapter studies the theoretical foundations needed to design a reliability-aware reinforcement learning fusion framework.
