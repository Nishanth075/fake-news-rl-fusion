from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "thesis" / "figures"
TABLES = ROOT / "thesis" / "tables"


def box(ax, xy, text, width=1.8, height=0.55, color="#e9f2ff", edge="#26547c", size=9):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=size)
    return patch


def arrow(ax, start, end, color="#444444"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def novelty_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.text(0.2, 5.0, "Fixed or simple fusion", fontsize=12, weight="bold", color="#444")
    ax.text(6.1, 5.0, "Proposed reliability-aware adaptive fusion", fontsize=12, weight="bold", color="#0b3954")

    box(ax, (0.3, 3.8), "Image model\nprobability", color="#e8f4f8")
    box(ax, (0.3, 2.6), "Text model\nprobability", color="#e8f4f8")
    box(ax, (2.7, 3.2), "Fixed rule\n(equal/confidence)", color="#fff3cd", edge="#b38600")
    box(ax, (5.0, 3.2), "Final\nprediction", color="#e7f6e7", edge="#2a7f2e")
    arrow(ax, (2.1, 4.05), (2.7, 3.55))
    arrow(ax, (2.1, 2.85), (2.7, 3.35))
    arrow(ax, (4.5, 3.47), (5.0, 3.47))

    box(ax, (6.0, 4.25), "Image probability\n+ confidence", color="#e8f4f8")
    box(ax, (6.0, 3.25), "Text probability\n+ confidence", color="#e8f4f8")
    box(ax, (6.0, 2.25), "Image/text quality\n+ disagreement", color="#e8f4f8")
    box(ax, (8.1, 3.25), "9-D reliability\nstate", color="#ede7f6", edge="#5e35b1")
    box(ax, (9.8, 3.25), "RL fusion action\nimage/text weights", color="#ffe8d6", edge="#c75c00", width=1.7)
    box(ax, (9.8, 2.05), "Prediction +\nexplanation record", color="#e7f6e7", edge="#2a7f2e", width=1.7)
    arrow(ax, (7.8, 4.5), (8.1, 3.68))
    arrow(ax, (7.8, 3.5), (8.1, 3.5))
    arrow(ax, (7.8, 2.5), (8.1, 3.3))
    arrow(ax, (9.9, 3.52), (9.8, 3.52))
    arrow(ax, (10.65, 3.25), (10.65, 2.6))
    ax.text(6.15, 0.75, "Novelty: fusion is selected per sample from reliability evidence, and the selected action is inspectable.", fontsize=10)
    save(fig, "figure_4_2_novelty_architecture_difference.png")


def data_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    steps = [
        ("Raw Fakeddit\nmetadata", "682,661 rows"),
        ("Label\nstandardization", "0=Real, 1=Fake"),
        ("Balanced final\nsubset request", "28,000 rows"),
        ("Image download\navailability check", "26,471 available"),
        ("Final splits", "18,893 train\n3,798 val\n3,780 test"),
    ]
    xs = [0.2, 2.35, 4.5, 6.65, 8.8]
    for x, (title, subtitle) in zip(xs, steps):
        box(ax, (x, 2.6), title + "\n" + subtitle, width=1.8, height=1.05, color="#eef7ee", edge="#2f7d32", size=8.5)
    for x in xs[:-1]:
        arrow(ax, (x + 1.8, 3.12), (x + 2.15, 3.12), color="#2f7d32")
    ax.text(0.3, 1.55, "Retention evidence: failed image downloads were recorded, not silently filled.", fontsize=10, weight="bold")
    ax.text(0.3, 1.1, "Download summary: requested 28,000; downloaded 26,471; failed 1,529.", fontsize=10)
    ax.text(0.3, 0.7, "Validation checks also recorded duplicate ID, image-path, and text-overlap diagnostics.", fontsize=10)
    save(fig, "figure_5_2_data_preparation_and_retention.png")


def evaluation_flow():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    top = [
        ("Image branch\nResNet18", 0.4, 3.8),
        ("Text branch\nDistilBERT", 0.4, 2.35),
        ("Reliability\nstate builder", 2.8, 3.05),
        ("Fusion methods\nbaselines + RL", 5.0, 3.05),
        ("Evaluation suite", 7.25, 3.05),
        ("Evidence for\nresearch claims", 9.35, 3.05),
    ]
    for label, x, y in top:
        box(ax, (x, y), label, width=1.7, height=0.85, color="#edf2fb", edge="#1f4e79", size=9)
    arrow(ax, (2.1, 4.2), (2.8, 3.55))
    arrow(ax, (2.1, 2.75), (2.8, 3.25))
    for x1, x2 in [(4.5, 5.0), (6.7, 7.25), (8.95, 9.35)]:
        arrow(ax, (x1, 3.47), (x2, 3.47), color="#1f4e79")
    tests = ["Unimodal metrics", "Fixed fusion baselines", "Controller baselines", "Ablation", "Robustness", "Faithfulness", "McNemar / seeds"]
    for i, t in enumerate(tests):
        x = 0.5 + (i % 4) * 2.55
        y = 1.25 - (i // 4) * 0.75
        box(ax, (x, y), t, width=2.1, height=0.45, color="#fff8e1", edge="#a66a00", size=8)
    ax.text(0.35, 0.25, "The thesis should show the full evidence chain, not only the final macro F1 number.", fontsize=10)
    save(fig, "figure_7_3_final_evaluation_evidence_flow.png")


def policy_distribution():
    actions = ["0", "1", "2", "3", "4", "5"]
    counts = [771, 1072, 465, 1279, 187, 6]
    colors = ["#52796f", "#84a98c", "#cad2c5", "#f4a261", "#e76f51", "#8d99ae"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(actions, counts, color=colors)
    ax.set_title("RL Fusion Action Distribution on Final Test Split")
    ax.set_xlabel("Selected action")
    ax.set_ylabel("Number of test samples")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.93, "Average weights: image=0.6594, text=0.3406", transform=ax.transAxes, fontsize=10)
    save(fig, "figure_7_4_policy_action_distribution.png")


def controller_baselines():
    methods = ["Logistic\nregression", "Tree\ndepth 3", "Tree\ndepth 5", "Tree\nunlimited", "RL\nseed 42"]
    f1 = [0.8470, 0.8523, 0.8529, 0.8473, 0.8676]
    colors = ["#8ecae6", "#8ecae6", "#219ebc", "#8ecae6", "#fb8500"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(methods, f1, color=colors)
    ax.set_ylim(0.83, 0.875)
    ax.set_ylabel("Test macro F1")
    ax.set_title("Same-State Controller Baselines versus RL Adaptive Fusion")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    ax.text(0.02, 0.05, "All controller baselines use the same 9 reliability-state features.", transform=ax.transAxes, fontsize=9)
    save(fig, "figure_7_5_controller_baseline_comparison.png")


def faithfulness():
    labels = ["Image\nsalient", "Image\nrandom", "Text\nsalient", "Text\nleast", "Text\nrandom"]
    vals = [0.12595, 0.18580, 0.15618, 0.06312, 0.11950]
    colors = ["#90be6d", "#b7b7a4", "#43aa8b", "#d9ed92", "#b7b7a4"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Comprehensiveness")
    ax.set_title("Explainability Faithfulness Check on 300 Test Samples")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.92, "Higher = larger drop after deleting evidence.", transform=ax.transAxes, fontsize=9)
    save(fig, "figure_7_6_explainability_faithfulness_n300.png")


def threshold_tuning():
    methods = ["Text", "Equal\nfusion", "Confidence\nfusion", "RL\nadaptive", "Matched\nMLP"]
    default = [0.8388, 0.8673, 0.8673, 0.8676, 0.8547]
    tuned = [0.8385, 0.8671, 0.8671, 0.8685, 0.8565]
    x = range(len(methods))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width = 0.34
    ax.bar([i - width / 2 for i in x], default, width, label="0.5 threshold", color="#8ecae6")
    ax.bar([i + width / 2 for i in x], tuned, width, label="Validation-tuned", color="#ffb703")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylim(0.83, 0.872)
    ax.set_ylabel("Test macro F1")
    ax.set_title("Validation Threshold Tuning Applied Unchanged to Test")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "figure_7_7_threshold_tuning_comparison.png")



def final_performance_dashboard():
    methods = [
        "Image",
        "Text",
        "Equal",
        "Confidence",
        "Reliability",
        "MLP\nmatched",
        "RL",
        "RL tuned",
    ]
    macro_f1 = [0.7769, 0.8388, 0.8673, 0.8673, 0.8612, 0.8547, 0.8676, 0.8685]
    accuracy = [0.7791, 0.8389, 0.8675, 0.8675, 0.8614, 0.8548, 0.8677, 0.8685]
    roc_auc = [0.8572, 0.9121, 0.9325, 0.9340, 0.9329, 0.9291, 0.9329, 0.9329]
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.bar(x - width, macro_f1, width, label="Macro F1", color="#1f77b4")
    ax.bar(x, accuracy, width, label="Accuracy", color="#2ca02c")
    ax.bar(x + width, roc_auc, width, label="ROC-AUC", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.74, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("Final Test Performance Dashboard")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.02,
        0.04,
        "Main reading: fusion improves over unimodal branches; RL-tuned gives the best final macro F1.",
        transform=ax.transAxes,
        fontsize=9,
    )
    save(fig, "figure_7_8_final_performance_dashboard.png")


def confusion_matrix_panel():
    matrices = {
        "Image-only": np.array([[1659, 207], [628, 1286]]),
        "Text-only": np.array([[1637, 229], [380, 1534]]),
        "Equal fusion": np.array([[1698, 168], [333, 1581]]),
        "RL adaptive": np.array([[1697, 169], [331, 1583]]),
    }
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.5))
    vmax = max(mat.max() for mat in matrices.values())
    for ax, (title, mat) in zip(axes, matrices.items()):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Real", "Fake"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        for (i, j), val in np.ndenumerate(mat):
            ax.text(j, i, str(val), ha="center", va="center", fontsize=10, color="#111")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72, label="Samples")
    fig.suptitle("Confusion Matrix Comparison on Final Test Split", y=1.04, fontsize=12, weight="bold")
    save(fig, "figure_7_9_confusion_matrix_panel.png")


def seed_stability_chart():
    seeds = ["42", "7", "13"]
    rl = np.array([0.8676044475, 0.8544529826, 0.8560101237])
    equal = np.array([0.8673332161, 0.8673332161, 0.8673332161])
    controller = np.array([0.8528505686, 0.8528505686, 0.8528505686])
    x = np.arange(len(seeds))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(x, equal, marker="o", label="Equal fusion baseline", color="#6c757d", linewidth=2)
    ax.plot(x, controller, marker="s", label="Best same-state controller", color="#219ebc", linewidth=2)
    ax.plot(x, rl, marker="^", label="RL adaptive fusion", color="#fb8500", linewidth=2.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylim(0.848, 0.871)
    ax.set_ylabel("Test macro F1")
    ax.set_title("Seed Stability of RL Fusion Compared with Baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left")
    ax.text(
        0.02,
        0.92,
        "Interpretation: RL is consistently above the same-state controller, but not consistently above equal fusion.",
        transform=ax.transAxes,
        fontsize=9,
    )
    save(fig, "figure_7_10_seed_stability.png")


def modality_weight_behavior():
    groups = ["All test", "Agreement", "Disagreement", "Image high", "Image medium", "Image low"]
    image_w = np.array([0.6594, 0.7132, 0.4987, 0.6659, 0.6579, 0.5628])
    text_w = 1 - image_w
    y = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(y, image_w, color="#457b9d", label="Image weight")
    ax.barh(y, text_w, left=image_w, color="#f4a261", label="Text weight")
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average fusion weight")
    ax.set_title("Policy Behaviour by Agreement and Image Quality Group")
    ax.legend(loc="lower right")
    for i, v in enumerate(image_w):
        ax.text(v / 2, i, f"{v:.2f}", va="center", ha="center", color="white", fontsize=9)
        ax.text(v + text_w[i] / 2, i, f"{text_w[i]:.2f}", va="center", ha="center", color="#222", fontsize=9)
    save(fig, "figure_7_11_modality_weight_behavior.png")

def write_tables():
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = {
        "table_7_3_dataset_retention.csv": [
            ["stage", "requested_or_source", "available", "failed_or_removed", "evidence"],
            ["Full prepared metadata", "682661", "682661", "0", "fakeddit_stats.json"],
            ["Final requested subset", "28000", "26471", "1529 failed image downloads", "final image download summary"],
            ["Train split", "20000 requested", "18893", "1107 failed", "final_splits_available/train.csv"],
            ["Validation split", "4000 requested", "3798", "202 failed", "final_splits_available/validation.csv"],
            ["Test split", "4000 requested", "3780", "220 failed", "final_splits_available/test.csv"],
        ],
        "table_7_4_controller_baselines.csv": [
            ["method", "state_dim", "validation_macro_f1", "test_macro_f1", "test_accuracy"],
            ["state_logistic_regression", "9", "0.8541", "0.8470", "0.8471"],
            ["state_decision_tree_depth_3", "9", "0.8576", "0.8523", "0.8524"],
            ["state_decision_tree_depth_5", "9", "0.8604", "0.8529", "0.8529"],
            ["state_decision_tree_unlimited", "9", "0.8541", "0.8473", "0.8474"],
            ["rl_adaptive_fusion_seed_42", "9", "0.8678", "0.8676", "0.8677"],
        ],
        "table_7_6_faithfulness_n300.csv": [
            ["explanation_test", "samples", "salient_or_selected", "control", "interpretation"],
            ["image deletion", "300", "0.1260", "random 0.1858", "Grad-CAM evidence weak in this metric"],
            ["text deletion", "300", "0.1562", "least 0.0631; random 0.1195", "token saliency shows positive faithfulness"],
        ],
        "table_7_7_research_question_evidence.csv": [
            ["examiner_question", "test_or_evidence", "main_finding"],
            ["Novelty and architecture difference", "adaptive fusion architecture; same-state controller baselines", "sample-level RL action and weights provide inspectable fusion"],
            ["Baseline comparison", "image/text/equal/confidence/reliability/MLP/RL comparisons", "RL competitive; threshold-tuned RL macro F1 0.8685"],
            ["Why reliability features", "state ablation", "full reliability state outperformed probability-only and confidence states"],
            ["Dataset filtering", "download and validation statistics", "26,471 final multimodal rows retained; failures recorded"],
            ["Generalizability", "threats-to-validity analysis", "claims restricted to Fakeddit; cross-dataset work remains future work"],
            ["Explainability", "Grad-CAM, token saliency, fusion weights, N=300 faithfulness", "text saliency positive; image faithfulness remains limited"],
        ],
    }
    for name, rows in tables.items():
        with (TABLES / name).open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)


def main():
    novelty_architecture()
    data_pipeline()
    evaluation_flow()
    policy_distribution()
    controller_baselines()
    faithfulness()
    threshold_tuning()
    final_performance_dashboard()
    confusion_matrix_panel()
    seed_stability_chart()
    modality_weight_behavior()
    write_tables()


if __name__ == "__main__":
    main()

