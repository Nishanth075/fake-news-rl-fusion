# Chapter 6: Implementation of the Multimodal Fake News Detection Framework

## Introduction

This chapter explains how the adaptive fusion framework was implemented. The implementation used a modular Python repository with separate configuration files, scripts, source modules, tests, notebooks, and output folders. The implementation followed a staged workflow so that data preparation, model training, fusion, analysis, and result export could be verified independently.

## Development Environment

The framework was implemented in Python using PyTorch for deep learning, Transformers for the text branch, scikit-learn for metrics, pandas for tabular processing, and image-processing utilities for visual data handling. Experiments were executed mainly in Google Colab with GPU runtime and Google Drive storage. GitHub was used for version control, and GitHub Desktop was used to manage commits and pushes from the local machine.

The repository was organized into configuration files, scripts, source modules, tests, notebooks, and output folders. YAML configuration files controlled paths, hyperparameters, and experiment variants. This reduced the risk of manually changing code for each experiment.

## Dataset Preparation and Label Standardization

The dataset implementation converted Fakeddit multimodal metadata into a standard project schema. The standard fields were sample identifier, image path, text, and label. Fakeddit two-way labels were converted into the project convention where 0 represents real content and 1 represents fake content.

The preparation stage exported dataset statistics and checked the data for missing required fields, empty text, duplicate identifiers, duplicate text, and duplicate image paths. A final subset was then created for computable experiments. Image downloading was implemented as a separate stage so that unavailable image URLs could be tracked without corrupting the original metadata.

The final image-available subset contained 26,471 samples after image download filtering. This included 18,893 training samples, 3,798 validation samples, and 3,780 test samples.

## Image Branch Implementation

The image branch was implemented using a ResNet-based classifier. Images were resized and transformed before being passed into the model. The model produced binary probabilities for fake news detection. Training used the training split, while the best checkpoint was selected using validation macro F1.

The image branch produced output CSV files for the train, validation, and test splits. These files contained sample identifiers, labels, image probabilities, predictions, and confidence values. The final image-only test macro F1 was 0.7769.

## Text Branch Implementation

The text branch was implemented using a DistilBERT-based classifier. Text samples were tokenized and passed through the pretrained language model with a classification head. The branch produced text probabilities, predictions, and confidence values for all splits.

The final text-only test macro F1 was 0.8388. The text branch was stronger than the image branch, but it also showed a larger train-test gap. This behaviour was recorded as an implementation and evaluation limitation rather than hidden.

## Reliability Feature Extraction

Reliability feature extraction merged image and text outputs and added quality-related features. Image quality was estimated using engineered visual statistics such as blur, contrast, entropy, and related normalized indicators. Text quality was estimated using text-based indicators such as length and availability. The reliability builder also computed confidence difference, quality difference, and modality disagreement.

The output of this stage was a set of reliability-enriched CSV files for train, validation, and test splits. These files were the direct inputs to baseline evaluation, RL fusion training, supervised fusion, robustness analysis, and explainability.

## Reinforcement Learning Fusion Implementation

The adaptive fusion controller was implemented as an offline contextual-bandit style reinforcement learning module. For each sample, the controller received a reliability-aware state vector and selected an action from the fusion action space. Each action corresponded to image and text weights. The final probability was computed using the selected weights and the modality probabilities.

The reward was based on classification correctness. A correct fused prediction received positive reward, and an incorrect prediction received negative reward. The model was trained on the training split, selected using validation macro F1, and evaluated on the test split.

The final main RL fusion run achieved 0.8676 test macro F1. The implementation also stored policy analysis files containing action distribution, average modality weights, group-level behaviour, and oracle upper-bound information for the action space.

## Supervised and Deterministic Baseline Implementation

Several baselines were implemented to evaluate the proposed method. Image-only and text-only baselines used the unimodal branch outputs. Equal fusion averaged image and text probabilities. Confidence-weighted fusion used prediction confidence to weight modalities. Reliability-weighted fusion used reliability indicators in a deterministic fusion rule. A supervised MLP fusion model was also implemented as a learned non-RL comparison.

Threshold tuning was implemented as a validation-based post-processing stage. The threshold was selected using validation macro F1 and then applied unchanged to the test split. This avoided selecting the threshold directly on test data.

## Explainability Implementation

The explainability implementation generated three types of outputs. Grad-CAM heatmaps were produced for image predictions. Token saliency CSV files were produced for text predictions. Fusion-level explanation summaries recorded selected action, image weight, text weight, final probability, and final prediction.

The final explainability run generated image, text, and fusion-level explanation artifacts. The corresponding summary file was exported under the final metrics directory for traceability.

## Testing and Reproducibility

The repository includes automated tests for dataset preparation, image inference, text modelling, reliability baselines, RL fusion, ablation, robustness, explainability, supervised fusion, threshold tuning, seed significance, and result export. These tests were used to verify that the implementation stages produced expected files and followed the configured schema.

The six notebooks in the `notebooks` folder provide a reproducible demonstration workflow. They cover data preparation, image model training, text model training, baselines and reliability, RL adaptive fusion, and explainability with final results.

## Summary

This chapter described the implementation of the proposed framework. The project was implemented as a modular Python pipeline with configurable stages and file-backed outputs. The next chapter evaluates the implemented framework using final experiment results.

