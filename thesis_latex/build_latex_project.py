from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thesis"
OUT = ROOT / "thesis_latex"
GENERATED = OUT / "generated"
FIGURES_OUT = OUT / "figures"
TABLES_OUT = OUT / "tables"


FRONT_MATTER = [
    ("declaration", "Declaration", SOURCE / "front_matter" / "declaration.md"),
    ("dedication", "Dedication", SOURCE / "front_matter" / "dedication.md"),
    ("acknowledgement", "Acknowledgement", SOURCE / "front_matter" / "acknowledgement.md"),
    ("abstract", "Abstract", SOURCE / "front_matter" / "abstract_draft.md"),
    ("abbreviations", "List of Abbreviations", SOURCE / "front_matter" / "list_of_abbreviations.md"),
]


CHAPTERS = [
    SOURCE / "chapters" / "01_introduction.md",
    SOURCE / "chapters" / "02_literature_review.md",
    SOURCE / "chapters" / "03_theoretical_foundations.md",
    SOURCE / "chapters" / "04_approach.md",
    SOURCE / "chapters" / "05_analysis_design.md",
    SOURCE / "chapters" / "06_implementation.md",
    SOURCE / "chapters" / "07_evaluation.md",
    SOURCE / "chapters" / "08_conclusion.md",
]


def read_metadata() -> dict[str, str]:
    data: dict[str, str] = {}
    for line in (SOURCE / "metadata.yaml").read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            key, value = line.split(":", 1)
            data[key] = value.strip().strip('"')
    return data


def escape_tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def convert_inline(text: str) -> str:
    code_spans: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code_spans.append(match.group(1))
        return f"@@CODE{len(code_spans) - 1}@@"

    text = re.sub(r"`([^`]+)`", stash_code, text)
    text = escape_tex(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    for index, code in enumerate(code_spans):
        text = text.replace(f"@@CODE{index}@@", r"\texttt{" + escape_tex(code) + "}")
    return text


def normalize_chapter_title(title: str) -> str:
    return re.sub(r"^Chapter\s+\d+\s*:\s*", "", title).strip()


def markdown_to_tex(path: Path, chapter: bool = False) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    in_itemize = False
    table_lines: list[str] = []

    def flush_itemize() -> None:
        nonlocal in_itemize
        if in_itemize:
            output.append(r"\end{itemize}")
            output.append("")
            in_itemize = False

    def flush_table() -> None:
        nonlocal table_lines
        if len(table_lines) < 2:
            table_lines = []
            return
        rows = []
        for raw in table_lines:
            if re.match(r"^\|\s*-", raw):
                continue
            rows.append([convert_inline(cell.strip()) for cell in raw.strip("|").split("|")])
        if rows:
            column_count = len(rows[0])
            output.append(r"\begin{table}[H]")
            output.append(r"\centering")
            output.append(r"\small")
            output.append(r"\begin{tabularx}{\textwidth}{" + "Y" * column_count + "}")
            output.append(r"\toprule")
            output.append(" & ".join(rows[0]) + r" \\")
            output.append(r"\midrule")
            for row in rows[1:]:
                output.append(" & ".join(row) + r" \\")
            output.append(r"\bottomrule")
            output.append(r"\end{tabularx}")
            output.append(r"\end{table}")
            output.append("")
        table_lines = []

    first_heading_done = False
    for raw in lines:
        stripped = raw.strip().lstrip("\ufeff")
        if stripped.startswith("|"):
            flush_itemize()
            table_lines.append(stripped)
            continue
        flush_table()

        if not stripped:
            flush_itemize()
            output.append("")
            continue
        if stripped.startswith("# "):
            flush_itemize()
            title = convert_inline(normalize_chapter_title(stripped[2:]))
            if chapter and not first_heading_done:
                output.append(r"\chapter{" + title + "}")
                first_heading_done = True
            else:
                output.append(r"\section*{" + title + "}")
            output.append("")
        elif stripped.startswith("## "):
            flush_itemize()
            output.append(r"\section{" + convert_inline(stripped[3:]) + "}")
            output.append("")
        elif stripped.startswith("### "):
            flush_itemize()
            output.append(r"\subsection{" + convert_inline(stripped[4:]) + "}")
            output.append("")
        elif stripped.startswith("- "):
            if not in_itemize:
                output.append(r"\begin{itemize}")
                in_itemize = True
            output.append(r"\item " + convert_inline(stripped[2:]))
        else:
            flush_itemize()
            output.append(convert_inline(stripped))
            output.append("")
    flush_table()
    flush_itemize()
    return "\n".join(output).strip() + "\n"



def insert_after_section(content: str, section_title: str, addition: str) -> str:
    marker = r"\section{" + section_title + "}"
    position = content.find(marker)
    if position == -1:
        return content + "\n" + addition + "\n"
    next_section = content.find(r"\section{", position + len(marker))
    if next_section == -1:
        return content + "\n" + addition + "\n"
    return content[:next_section].rstrip() + "\n\n" + addition + "\n\n" + content[next_section:]


def table_escape_tex(text: str) -> str:
    escaped = escape_tex(text)
    return escaped.replace(r"\_", r"\_\allowbreak ").replace("/", r"/\allowbreak ")


def copy_assets() -> None:
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    for fig in (SOURCE / "figures").glob("*.png"):
        shutil.copy2(fig, FIGURES_OUT / fig.name)
    shutil.copy2(ROOT / "docs" / "research_architecture_tikz.tex", FIGURES_OUT / "research_architecture_tikz.tex")
    for table in (SOURCE / "tables").glob("*.csv"):
        shutil.copy2(table, TABLES_OUT / table.name)


def write_front_matter() -> None:
    for filename, _title, source_path in FRONT_MATTER:
        (GENERATED / f"{filename}.tex").write_text(markdown_to_tex(source_path), encoding="utf-8")


def write_chapters() -> None:
    for chapter_path in CHAPTERS:
        stem = chapter_path.stem
        content = markdown_to_tex(chapter_path, chapter=True)
        if chapter_path.name == "04_approach.md":
            content = insert_after_section(content, "Inputs and Outputs of the Proposed Extension", method_equations_tex())
            approach_figures = "\n".join([
                r"\input{figures/research_architecture_tikz.tex}",
                figure_tex("figure_4_2_novelty_architecture_difference.png", "Structural difference between fixed fusion and the proposed reliability-aware adaptive fusion controller.", "fig:novelty_architecture_difference"),
            ])
            content = insert_after_section(content, "Process Workflow of the Proposed Extension", approach_figures)
            content = insert_after_section(content, "Adaptive Fusion Action Space", action_fusion_equations_tex())
            content = insert_after_section(content, "Reward Design", reward_equation_tex())
        if chapter_path.name == "05_analysis_design.md":
            content = insert_after_section(content, "Top-Level Architecture", figure_tex("figure_5_1_system_architecture.png", "System architecture of the adaptive fusion framework.", "fig:system_architecture"))
            content = insert_after_section(content, "Data Flow and Interaction Design", figure_tex("figure_5_2_data_preparation_and_retention.png", "Dataset preparation, image-availability filtering, and final split retention evidence.", "fig:data_preparation_retention"))
            content = insert_after_section(content, "Design of the Adaptive Fusion Controller", reliability_quality_equations_tex())
        if chapter_path.name == "06_implementation.md":
            content = insert_after_section(content, "Summary", figure_tex("figure_7_3_final_evaluation_evidence_flow.png", "Implementation-to-evaluation evidence flow used to connect saved artifacts with final thesis claims.", "fig:implementation_evidence_flow"))
        if chapter_path.name == "07_evaluation.md":
            content = insert_after_section(content, "Evaluation Strategy", figure_tex("figure_7_3_final_evaluation_evidence_flow.png", "Final evaluation evidence flow across unimodal models, fusion methods, robustness, explainability, and statistical checks.", "fig:evaluation_evidence_flow"))
            content = insert_after_section(content, "Dataset and Splits", csv_table_tex(TABLES_OUT / "table_7_3_dataset_retention.csv", "Dataset preparation and retention evidence for the final multimodal experiment.", "tab:dataset_retention"))
            content = insert_after_section(content, "Evaluation Metrics", evaluation_equations_tex())
            content = insert_after_section(content, "Baseline and Fusion Results", "\n".join([
                figure_tex("figure_7_8_final_performance_dashboard.png", "Final test performance dashboard across accuracy, macro F1, and ROC-AUC.", "fig:final_performance_dashboard"),
                figure_tex("figure_7_9_confusion_matrix_panel.png", "Confusion matrix comparison for unimodal, fixed-fusion, and adaptive-fusion methods.", "fig:confusion_matrix_panel"),
                figure_tex("figure_7_1_method_macro_f1.png", "Final test macro F1 comparison across selected methods.", "fig:method_macro_f1"),
                csv_table_tex(TABLES_OUT / "table_7_1_final_method_comparison.csv", "Final test method comparison.", "tab:final_method_comparison"),
                figure_tex("figure_7_7_threshold_tuning_comparison.png", "Comparison of default-threshold and validation-tuned threshold results.", "fig:threshold_tuning_comparison"),
            ]))
            content = insert_after_section(content, "Ablation Study", "\n".join([
                figure_tex("figure_7_2_ablation_macro_f1.png", "Ablation results for different RL fusion state representations.", "fig:ablation_macro_f1"),
                figure_tex("figure_7_5_controller_baseline_comparison.png", "Same-state controller baseline comparison against the RL adaptive fusion controller.", "fig:controller_baseline_comparison"),
                csv_table_tex(TABLES_OUT / "table_7_4_controller_baselines.csv", "Same-state controller baseline results.", "tab:controller_baselines"),
            ]))
            content = insert_after_section(content, "Policy Analysis and Reliability Behaviour", "\n".join([
                figure_tex("figure_7_11_modality_weight_behavior.png", "Average modality weights selected by the RL policy across agreement and image-quality groups.", "fig:modality_weight_behavior"),
                figure_tex("figure_7_4_policy_action_distribution.png", "Distribution of selected RL fusion actions on the final test split.", "fig:policy_action_distribution"),
            ]))
            content = insert_after_section(content, "Robustness Analysis", figure_tex("figure_7_4_robustness_image_quality.png", "Image quality degradation and image-weight adaptation.", "fig:robustness_image_quality"))
            content = insert_after_section(content, "Explainability Evaluation", "\n".join([
                faithfulness_equation_tex(),
                figure_tex("figure_7_6_explainability_faithfulness_n300.png", "Deletion-based explainability faithfulness check on 300 final test samples.", "fig:faithfulness_n300"),
                csv_table_tex(TABLES_OUT / "table_7_6_faithfulness_n300.csv", "Explainability faithfulness summary for image and text evidence deletion.", "tab:faithfulness_n300"),
            ]))
            content = insert_after_section(content, "Seed Stability and Statistical Considerations", "\n".join([
                statistical_equations_tex(),
                figure_tex("figure_7_10_seed_stability.png", "Seed-stability comparison between RL adaptive fusion, equal fusion, and the best same-state controller baseline.", "fig:seed_stability"),
            ]))
            content = insert_after_section(content, "External Generalization Case Study", csv_table_tex(TABLES_OUT / "table_7_8_external_openfake_calibration.csv", "External OpenFake calibration-holdout case study results.", "tab:external_openfake_calibration"))
            content = insert_after_section(content, "Discussion of Findings", csv_table_tex(TABLES_OUT / "table_7_7_research_question_evidence.csv", "Research-question and examiner-question evidence mapping.", "tab:research_question_evidence"))
        content += "\n" + r"\clearpage" + "\n"
        (GENERATED / f"{stem}.tex").write_text(content, encoding="utf-8")


def figure_tex(filename: str, caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.92\textwidth,height=0.42\textheight,keepaspectratio]{{figures/{filename}}}",
            rf"\caption{{{escape_tex(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )



def equation_tex(body: str, label: str | None = None) -> str:
    lines = [r"\begin{equation}", body]
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{equation}")
    return "\n".join(lines)


def method_equations_tex() -> str:
    return "\n\n".join([
        "The image and text branches are represented by functions that output fake-class probabilities:",
        equation_tex(r"p_i = f_i(x_i), \qquad p_t = f_t(x_t)", "eq:unimodal_probabilities"),
        "The confidence of each modality is computed from the stronger binary-class probability:",
        equation_tex(r"c_m = \max(p_m, 1-p_m), \qquad m \in \{i,t\}", "eq:modality_confidence"),
        "The reliability-aware state used by the fusion controller is defined as:",
        equation_tex(r"s = [p_i, c_i, q_i, p_t, c_t, q_t, |p_i-p_t|, c_i-c_t, q_i-q_t]", "eq:reliability_state"),
    ])


def action_fusion_equations_tex() -> str:
    return "\n\n".join([
        "Each discrete action maps to an interpretable image-text weighting pair:",
        equation_tex(r"\begin{aligned}\mathcal{A}=\{&(0.90,0.10),(0.75,0.25),(0.60,0.40),(0.50,0.50),\\ &(0.40,0.60),(0.25,0.75),(0.10,0.90)\}\end{aligned}", "eq:action_space"),
        equation_tex(r"a_k \mapsto (w_i^{(k)}, w_t^{(k)}), \qquad w_i^{(k)} + w_t^{(k)} = 1", "eq:action_weight_mapping"),
        "The final fake-class probability is then produced by weighted late fusion:",
        equation_tex(r"p_f = w_i^{(k)}p_i + w_t^{(k)}p_t", "eq:fused_probability"),
        "The binary decision rule is:",
        equation_tex(r"\hat{y}=\begin{cases}1, & p_f \ge \tau \\ 0, & p_f < \tau\end{cases}", "eq:prediction_rule"),
    ])


def reward_equation_tex() -> str:
    return "\n\n".join([
        "The reward function links the selected fusion action directly to classification correctness:",
        equation_tex(r"r(s,a_k)=\begin{cases}+1, & \hat{y}=y \\ -1, & \hat{y}\ne y\end{cases}", "eq:reward_function"),
    ])


def reliability_quality_equations_tex() -> str:
    return "\n\n".join([
        "The image reliability score is computed as the mean of availability and normalized visual quality cues:",
        equation_tex(r"q_i = \frac{q_{blur}+q_{brightness}+q_{contrast}+q_{entropy}+q_{available}}{5}", "eq:image_quality"),
        "The text reliability score is computed from length adequacy, truncation quality, and text availability:",
        equation_tex(r"q_t = \frac{q_{length}+q_{truncation}+q_{available}}{3}", "eq:text_quality"),
    ])


def evaluation_equations_tex() -> str:
    return "\n\n".join([
        "Macro F1 was used as the primary classification metric because it gives equal importance to both classes:",
        equation_tex(r"F1_{macro}=\frac{1}{C}\sum_{c=1}^{C}F1_c", "eq:macro_f1"),
        "Validation-threshold tuning selected the threshold that maximized validation macro F1 and then applied that threshold unchanged to the test split:",
        equation_tex(r"\tau^* = \arg\max_{\tau \in T} F1_{macro}^{val}(\tau)", "eq:threshold_tuning"),
    ])


def statistical_equations_tex() -> str:
    return "\n\n".join([
        "For paired classifier comparison, McNemar's test uses the discordant correctness counts:",
        equation_tex(r"b = |\{j: \hat{y}^{A}_{j}=y_j,\ \hat{y}^{B}_{j}\ne y_j\}|, \qquad c = |\{j: \hat{y}^{A}_{j}\ne y_j,\ \hat{y}^{B}_{j}=y_j\}|", "eq:mcnemar_counts"),
    ])


def faithfulness_equation_tex() -> str:
    return "\n\n".join([
        "Deletion-based faithfulness was measured using comprehensiveness, defined as the probability drop after removing evidence E:",
        equation_tex(r"Comp(E)=P(y_c\mid x)-P(y_c\mid x_{\setminus E})", "eq:faithfulness_comprehensiveness"),
    ])


def csv_table_tex(path: Path, caption: str, label: str) -> str:
    rows = list(csv.reader(path.open(encoding="utf-8")))
    column_count = len(rows[0])
    output = [
        r"\begin{table}[H]",
        r"\centering",
        r"\tiny",
        rf"\caption{{{escape_tex(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabularx}{\textwidth}{" + "Y" * column_count + "}",
        r"\toprule",
        " & ".join(table_escape_tex(cell) for cell in rows[0]) + r" \\",
        r"\midrule",
    ]
    for row in rows[1:]:
        output.append(" & ".join(table_escape_tex(cell) for cell in row) + r" \\")
    output.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(output)


def write_references() -> None:
    references = markdown_to_tex(SOURCE / "references" / "initial_ieee_references.md")
    (GENERATED / "references.tex").write_text(references, encoding="utf-8")
    appendix = markdown_to_tex(SOURCE / "appendices" / "artifact_manifest.md")
    (GENERATED / "appendix_artifacts.tex").write_text(appendix, encoding="utf-8")


def write_main() -> None:
    meta = read_metadata()
    chapter_inputs = "\n".join(rf"\input{{generated/{path.stem}.tex}}" for path in CHAPTERS)
    main = rf"""\documentclass[12pt,a4paper,oneside]{{report}}

\usepackage[a4paper,left=1.25in,right=1in,top=1in,bottom=1in]{{geometry}}
\usepackage{{times}}
\usepackage{{setspace}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage{{tikz}}
\usetikzlibrary{{positioning,arrows.meta}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{titlesec}}
\usepackage{{tocloft}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{enumitem}}
\usepackage{{ragged2e}}
\usepackage{{placeins}}

\onehalfspacing
\setlength{{\parindent}}{{0.25in}}
\setlength{{\parskip}}{{0.30em}}
\captionsetup{{font=small,labelfont=bf,justification=centering,singlelinecheck=false}}
\renewcommand{{\arraystretch}}{{1.18}}
\newcolumntype{{Y}}{{>{{\RaggedRight\arraybackslash}}X}}

\titleformat{{\chapter}}[display]
  {{\normalfont\bfseries\centering}}
  {{\chaptertitlename\ \thechapter}}
  {{12pt}}
  {{\Large}}

\begin{{document}}

\pagenumbering{{roman}}

\begin{{titlepage}}
\centering
\vspace*{{1.0in}}
{{\Large\bfseries {escape_tex(meta["title"])}\par}}
\vspace{{0.8in}}
{{\large {escape_tex(meta["candidate_name"])}\par}}
{{\large {escape_tex(meta["registration_number"])}\par}}
\vspace{{0.5in}}
{{\large BSc. (Hons) in Artificial Intelligence\par}}
\vfill
{{\large {escape_tex(meta["department"])}\par}}
{{\large {escape_tex(meta["faculty"])}\par}}
{{\large {escape_tex(meta["university"])}\par}}
{{\large {escape_tex(meta["country"])}\par}}
\vspace{{0.4in}}
{{\large {escape_tex(meta["submission_month_year"])}\par}}
\end{{titlepage}}

\begin{{titlepage}}
\centering
\vspace*{{0.8in}}
{{\Large\bfseries {escape_tex(meta["title"])}\par}}
\vspace{{0.5in}}
{{\large {escape_tex(meta["candidate_name"])}\par}}
{{\large {escape_tex(meta["registration_number"])}\par}}
\vspace{{0.5in}}
Thesis submitted in partial fulfillment of the requirements for the Research Project in Artificial Intelligence for the degree of BSc. (Hons) in Artificial Intelligence
\vfill
{{\large {escape_tex(meta["department"])}\par}}
{{\large {escape_tex(meta["faculty"])}\par}}
{{\large {escape_tex(meta["university"])}\par}}
{{\large {escape_tex(meta["country"])}\par}}
\vspace{{0.4in}}
{{\large {escape_tex(meta["submission_month_year"])}\par}}
\end{{titlepage}}

\input{{generated/declaration.tex}}
\clearpage
\input{{generated/dedication.tex}}
\clearpage
\input{{generated/acknowledgement.tex}}
\clearpage
\input{{generated/abstract.tex}}
\clearpage

\tableofcontents
\clearpage
\listoffigures
\clearpage
\listoftables
\clearpage
\input{{generated/abbreviations.tex}}
\clearpage

\pagenumbering{{arabic}}
{chapter_inputs}

\clearpage
\input{{generated/references.tex}}

\appendix
\input{{generated/appendix_artifacts.tex}}

\end{{document}}
"""
    (OUT / "main.tex").write_text(main, encoding="utf-8")


def write_readme() -> None:
    readme = """# LaTeX Thesis Project

This folder is an Overleaf-ready LaTeX version of the thesis draft.

## How to use

1. Upload the entire `thesis_latex` folder to Overleaf, or zip the contents and upload them as a new project.
2. Set `main.tex` as the main file.
3. Compile with pdfLaTeX.

If compiling locally, install a LaTeX distribution such as MiKTeX or TeX Live, then run:

```bash
pdflatex main.tex
pdflatex main.tex
```

The generated chapter files are in `generated/`. Re-run `python build_latex_project.py` from this folder after editing markdown sources under `thesis/`.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    copy_assets()
    write_front_matter()
    write_chapters()
    write_references()
    write_main()
    write_readme()
    print(OUT / "main.tex")


if __name__ == "__main__":
    main()




