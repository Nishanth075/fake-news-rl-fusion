# Final Academic Review Report

This report documents the final academic-quality edits made after the critical thesis review. No models were retrained, no metrics were changed, and no experiments were invented.

## Chapters Modified

- Front matter abstract: contribution reframed around competitive adaptive fusion and sample-level decision transparency.
- Chapter 1 Introduction: aim, novelty/contribution framing, objectives, and scope were rewritten for consistency.
- Chapter 2 Literature Review: related-work comparison strengthened with SAFE, SAMPLE, TRIMOON, MMFND, DAMMFND, MSAF-Net, and EANN positioning.
- Chapter 3 Theoretical Foundations: RL formulation clarified as contextual-bandit-style reinforcement learning; reliability and explainability limitations made explicit.
- Chapter 4 Approach: novelty language changed to methodological contribution; explanation wording changed to transparent decision evidence.
- Chapter 5 Analysis and Design: section-title numbering formatting cleaned.
- Chapter 6 Implementation: section-title numbering formatting cleaned.
- Chapter 7 Evaluation: discussion expanded with interpretation of RL behaviour, fixed-fusion competitiveness, MLP underperformance, robustness, faithfulness, and OpenFake transfer.
- Chapter 8 Conclusion and Further Work: limitations rewritten and conclusion reframed around transparent adaptive fusion rather than predictive superiority.

## Substantially Rewritten Paragraphs

- Abstract final evaluation paragraph.
- Chapter 1 problem/motivation contribution framing.
- Chapter 1 proposed approach/novelty paragraph.
- Chapter 1 research aim and evaluation objective.
- Chapter 1 scope and delimitations paragraph.
- Chapter 2 comparative-analysis section and table.
- Chapter 2 research-gap paragraph.
- Chapter 3 reinforcement-learning formulation paragraph.
- Chapter 3 reliability-estimation paragraphs.
- Chapter 3 explainability-theory paragraph.
- Chapter 4 workflow contribution paragraph.
- Chapter 4 explanation-strategy paragraph.
- Chapter 4 AI body-of-knowledge positioning paragraph.
- Chapter 7 baseline and fusion interpretation paragraphs.
- Chapter 7 robustness interpretation paragraph.
- Chapter 7 seed-stability/statistical paragraph.
- Chapter 7 discussion section.
- Chapter 7 OpenFake case-study framing.
- Chapter 8 contribution summary.
- Chapter 8 limitations section.
- Chapter 8 future-work and final summary paragraphs.

## Scientific Claims Made More Cautious

- RL fusion is no longer framed as primarily improving classification accuracy.
- The small RL-vs-equal-fusion difference is stated as non-significant under paired McNemar testing.
- The contribution is framed as reliability-aware sample-level adaptive fusion and decision transparency.
- Grad-CAM, token saliency, and fusion weights are described as inspectable evidence, not complete causal explanations.
- OpenFake is described as an exploratory external case study, not proof of broad generalization.
- Reliability features are described as heuristic usability proxies, not truthfulness measures.
- Contextual-bandit-style RL framing is used instead of implying a long-horizon sequential RL problem.

## Limitations Added or Strengthened

- Marginal improvement over strong fixed fusion.
- Seed instability of the RL controller.
- Dataset scope and limited external validation.
- OpenFake performance drop under zero-shot external transfer.
- Image availability filtering and possible availability bias.
- Heuristic reliability features for image and text quality.
- Text-branch overfitting risk.
- Discrete action-space restriction.
- Binary reward lacking calibration or margin information.
- Mixed explainability faithfulness evidence.
- Grouped/subset-based ablations and faithfulness checks.

## Contradictions Fixed

- Removed the outdated claim that no quantitative faithfulness evaluation was performed; the thesis now states that an N=300 deletion-based faithfulness check was conducted.
- Fixed the thesis framing so it no longer implies statistically proven superiority over equal fusion.
- Removed duplicate manual section numbering from generated LaTeX headings, preventing repeated headings such as 1.1 1.1.
- Replaced weak wording such as "novel in this project" with methodological-contribution wording.
- Aligned OpenFake wording with the actual result: useful for diagnosing domain shift, not for claiming generalization success.

## Remaining Weaknesses Requiring New Experiments

- Stronger seed stability would require more RL training runs and hyperparameter tuning.
- Stronger external generalization claims would require training or calibration on additional external datasets.
- Stronger explanation validation would require larger faithfulness protocols, insertion tests, or human studies.
- Better reliability modelling would require implementing uncertainty, calibration, perplexity, Monte Carlo dropout, or learned reliability estimators.
- Stronger statistical power would require repeated full-pipeline experiments rather than only saved-output audits.
