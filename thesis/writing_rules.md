# Thesis Writing Rules

This file records the writing rules used for the thesis draft. The university guideline and template are the main authority. The newer reference thesis is used only for structure and academic style, not for copied wording.

## Grammar and Tense

- Use present tense for established knowledge, chapter descriptions, and definitions.
- Use past tense for completed implementation, training, experiments, and observed results.
- Use cautious present or past tense for findings, for example: "The results indicate" or "The final run achieved".
- Avoid exaggerated claims such as "dramatically outperforms" unless supported by strong statistical evidence.
- Keep terminology consistent: use "adaptive fusion policy", "image branch", "text branch", "reliability-aware state", and "modality weight" consistently.

## Plagiarism and Turnitin Safety

- Do not copy paragraphs from the friend's thesis, papers, websites, or tool outputs.
- Common compulsory university wording, such as the declaration template, may be preserved where required.
- Literature ideas must be cited using IEEE style.
- Project-specific claims must be backed by files in `outputs/`, `notebooks/`, `configs/`, or `src/`.
- Results must be reported with careful wording: competitive performance, modest main-run improvement, explainable adaptive behavior, and identified limitations.

## Evidence Discipline

- Every result in the thesis must be traceable to a local evidence file.
- The main comparison table should be based on `outputs/tables/final_method_comparison.csv`.
- Threshold tuning should be based on `outputs/metrics/final_threshold_tuning_matched_summary.csv`.
- Seed stability should be discussed carefully using `outputs/metrics/final_seed_significance_summary.csv`.
- Explainability should be described as qualitative evidence unless a quantitative faithfulness metric is later added.

## University Guideline Compliance

- Every chapter must include an Introduction and Summary.
- Figures and tables must be numbered, captioned, and cited in text.
- Chapter 5 must avoid implementation-level algorithm details.
- Chapter 6 must describe implementation, algorithms, environment, and testing.
- Chapter 7 must align evaluation directly with the objectives and approach.
- Chapter 8 must include objective achievement, limitations, challenges, and future work.
