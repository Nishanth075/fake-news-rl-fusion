# Chapter 2: Developments and Challenges in Multimodal Fake News Detection

## 2.1 Introduction

This chapter reviews prior work related to fake news detection, with emphasis on multimodal fake news detection and fusion strategies. The purpose of the review is to position the research gap addressed in this thesis: the need for explainable, sample-level adaptive image-text fusion. The chapter first discusses fake news detection as a social media analysis problem, then reviews text-based, image-based, and multimodal approaches. It also examines explainability and the limitations of existing fusion methods.

## 2.2 Fake News Detection in Social Media

Fake news detection has been studied as a data mining and artificial intelligence problem because online misinformation can be produced and distributed at high speed. Shu et al. described fake news detection on social media as a challenging task because the problem involves news content, user behaviour, propagation patterns, and psychological factors [1]. Their survey showed that fake news cannot be treated only as a conventional text classification problem, because social media content is shaped by both the message and the surrounding context.

In social media environments, fake news may be expressed through short captions, emotionally persuasive headlines, edited or misleading images, and posts designed to exploit existing beliefs. These properties make automated detection difficult. A model may learn superficial features from a particular dataset but fail when the event, topic, or presentation style changes. Therefore, recent research has increasingly focused on multimodal and generalizable detection methods.

## 2.3 Text-Based Fake News Detection

Text-based fake news detection uses textual features such as titles, captions, article content, writing style, and semantic representations. Earlier text-based systems used manually engineered linguistic features or classical machine learning methods. Later systems used neural language models to learn representations directly from text.

Transformer-based language models improved many natural language processing tasks by learning contextual word representations. BERT introduced deep bidirectional pretraining and showed that a pretrained language model could be fine-tuned for many downstream tasks with limited architectural modification [4]. DistilBERT reduced the computational cost of BERT through knowledge distillation while retaining much of its language understanding ability [5]. These models are relevant to fake news detection because they can represent short and noisy social media text more effectively than simple bag-of-words features.

However, text alone may be insufficient for multimodal posts. A caption may appear harmless while the image carries misleading context, or an image may be neutral while the text contains the false claim. Text models also risk overfitting to dataset-specific wording, topic patterns, or annotation artifacts. This limitation motivates multimodal approaches that use additional visual evidence.

## 2.4 Image-Based and Multimodal Fake News Detection

Visual evidence can provide important cues in fake news detection. Images may be manipulated, reused from unrelated events, or paired with misleading captions. Deep convolutional networks are commonly used for image representation. ResNet introduced residual learning to make deep visual networks easier to optimize and has become a widely used backbone for image classification [6].

Multimodal fake news detection combines text and image information. Wang et al. proposed Event Adversarial Neural Networks (EANN) for multimodal fake news detection and emphasized the problem of event-specific features that do not transfer well to unseen events [2]. Their work showed that multimodal representations can improve fake news detection, but it also highlighted the difficulty of learning robust and generalizable features.

The Fakeddit dataset was introduced as a large-scale multimodal fake news benchmark with multiple label granularities and image-text samples [3]. It is particularly relevant to this thesis because it provides image and text information suitable for evaluating multimodal fake news detection. The dataset also exposes a practical challenge: not all image URLs remain available, so final multimodal experiments often require careful filtering and documentation of available samples.

## 2.5 Fusion Strategies in Multimodal Learning

Fusion is the process of combining information from multiple modalities. In multimodal fake news detection, fusion may occur at feature level, decision level, or through attention-based interaction. Feature-level fusion combines learned image and text embeddings before classification. Decision-level fusion combines modality predictions, such as probability scores. Attention-based and similarity-aware methods try to capture relationships between modalities.

SAFE introduced a similarity-aware multimodal fake news detection approach, showing that cross-modal similarity can be useful when analysing image-text consistency [9]. Progressive Fusion Networks later argued that multimodal fake news detection should not only use deep semantic features, because shallow-level visual and textual cues may also contain useful information [10]. TRIMOON focused on inconsistency-aware fusion and reported that image-text inconsistency can introduce fusion noise if it is not handled carefully [11]. These studies support the view that the relationship between modalities matters, not only the presence of multiple modalities.

Recent research has moved further toward adaptive and reliability-aware fusion. SAMPLE introduced similarity-aware multimodal prompt learning and used similarity-aware fusion to reduce noise from unrelated cross-modal representations [12]. MMFND used multi-task learning and cross-modal correlation weighting to guide multimodal fusion [13]. DAMMFND considered domain-aware multimodal decision-making and emphasized that modality contribution can vary across domains [14]. MSAF-Net, developed for short-video fake news detection, also used similarity-guided adaptive fusion to balance unimodal and inter-modal consistency features [15]. These recent methods show that adaptive fusion remains an active research direction.

Despite these advances, many fusion methods still provide limited sample-level explanation. A fixed averaging rule is simple but applies the same weighting behaviour to every sample. Confidence-weighted fusion is more flexible, but it remains deterministic and may not fully capture richer reliability features such as image quality, text quality, and modality disagreement. Deep feature-fusion models may learn complex interactions, but their fusion decisions can be difficult to inspect. The present research therefore focuses on a lightweight reinforcement learning controller that exposes the selected action and modality weights for each sample.

## 2.6 Explainability in Fake News Detection

Explainability is important in fake news detection because the output can influence user trust and decision-making. A prediction label alone may be insufficient when users need to understand why a post was classified as fake or real. In multimodal systems, explanation should ideally show evidence from image, text, and fusion levels.

Grad-CAM provides visual explanations for convolutional neural networks by highlighting image regions that influence a target prediction [8]. Token saliency methods can show which words or subword tokens influenced a text model. These explanation methods help inspect unimodal branches. However, multimodal fake news detection also needs fusion-level explanation: how much each modality contributed and why the final decision was formed.

Recent work has also connected fake news detection with interpretability and trust. Fusion-based approaches have been proposed to improve both detection and interpretability [16], and recent explainable multimodal fake-news studies show that robustness and transparency are becoming central concerns in misinformation detection [17]. The proposed research addresses this direction by exposing the selected fusion action, image weight, text weight, final probability, and branch-level saliency artifacts. This does not fully solve the broader problem of explanation faithfulness, but it improves transparency compared with a black-box final prediction.

## 2.7 Comparative Analysis of Existing Methods

Table 2.1 summarizes key categories of related work and their relevance to this research.

**Table 2.1: Comparative analysis of related fake news detection methods**

| Method Category | Example Work | Strength | Limitation Relevant to This Thesis |
|---|---|---|---|
| Social media fake news surveys | Shu et al. [1] | Defines fake news detection challenges and datasets | Does not propose an adaptive multimodal fusion mechanism |
| Event-aware multimodal detection | EANN [2] | Uses text and image features and addresses event transfer | Fusion is not presented as an explicit sample-level action policy |
| Large multimodal benchmark | Fakeddit [3] | Provides large-scale image-text fake news data | Dataset availability and multimodal filtering must be handled carefully |
| Transformer text modelling | BERT [4], DistilBERT [5] | Strong contextual text representation | Text-only evidence can overfit or miss visual misinformation |
| CNN image modelling | ResNet [6] | Strong visual representation backbone | Image-only evidence may be weak without text context |
| Similarity-aware multimodal fusion | SAFE [9], SAMPLE [12] | Considers image-text relation and cross-modal noise | Fusion decisions may still be difficult to inspect at sample level |
| Progressive and inconsistency-aware fusion | PFN [10], TRIMOON [11] | Studies shallow/deep cues and modality inconsistency | Does not formulate fusion as an explicit action-selection policy |
| Recent adaptive/domain-aware fusion | MMFND [13], DAMMFND [14], MSAF-Net [15] | Shows that modality contribution and cross-modal similarity vary across settings | Often uses complex fusion modules that may reduce direct interpretability |
| Visual explainability | Grad-CAM [8] | Highlights image evidence | Does not explain text contribution or fusion action by itself |

## 2.8 Research Gap

The literature indicates that multimodal fake news detection can benefit from both textual and visual evidence. It also shows that modality similarity, inconsistency, cross-modal noise, and domain variation are current research concerns. However, several gaps remain. First, many systems use fixed or opaque fusion methods that do not clearly adapt modality weights at sample level. Second, reliability information such as modality confidence, quality, and disagreement is not always explicitly represented in the fusion decision. Third, explanation often focuses on branch-level evidence but does not fully explain how modalities were combined.

Therefore, the research gap addressed in this thesis is the lack of an explainable, reliability-aware, sample-level adaptive fusion mechanism for image-text fake news detection. This thesis proposes a reinforcement learning-based fusion controller to address this gap by selecting a fusion action for each sample and exposing the resulting modality weights.

## 2.9 Summary

This chapter reviewed fake news detection, text-based modelling, image-based modelling, multimodal fusion, and explainability. The review showed that multimodal approaches are important but still face challenges in adaptive fusion and interpretability. The next chapter studies the theoretical foundations needed to design a reliability-aware reinforcement learning fusion framework.
