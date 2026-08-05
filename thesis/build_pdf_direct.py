from __future__ import annotations

import csv
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "thesis"
BUILD = THESIS / "build"
FIGURES = THESIS / "figures"
TABLES = THESIS / "tables"


def read_metadata() -> dict[str, str]:
    data = {}
    for line in (THESIS / "metadata.yaml").read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            key, value = line.split(":", 1)
            data[key] = value.strip().strip('"')
    return data


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCustom", parent=base["Title"], fontName="Times-Bold", fontSize=18, leading=24, alignment=TA_CENTER, spaceAfter=18),
        "center": ParagraphStyle("Center", parent=base["Normal"], fontName="Times-Roman", fontSize=12, leading=18, alignment=TA_CENTER, spaceAfter=8),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Times-Bold", fontSize=16, leading=20, spaceBefore=12, spaceAfter=10),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Times-Bold", fontSize=14, leading=18, spaceBefore=10, spaceAfter=8),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontName="Times-Bold", fontSize=12, leading=16, spaceBefore=8, spaceAfter=6),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Times-Roman", fontSize=11.5, leading=17, alignment=TA_JUSTIFY, spaceAfter=7),
        "caption": ParagraphStyle("Caption", parent=base["Normal"], fontName="Times-Italic", fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=10),
        "ref": ParagraphStyle("Reference", parent=base["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=13, alignment=TA_LEFT, spaceAfter=5),
    }


def clean_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def add_markdown(story, path: Path, st: dict, page_break: bool = True):
    if page_break:
        story.append(PageBreak())
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = []

    def flush_table():
        nonlocal table_lines
        if len(table_lines) >= 2:
            rows = []
            for line in table_lines:
                if re.match(r"^\|\s*-", line):
                    continue
                rows.append([clean_inline(c.strip()) for c in line.strip("|").split("|")])
            if rows:
                tbl = Table(rows, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 10))
        table_lines = []

    for line in lines:
        if line.strip().startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            story.append(Paragraph(clean_inline(stripped[2:]), st["h1"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(clean_inline(stripped[3:]), st["h2"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(clean_inline(stripped[4:]), st["h3"]))
        elif stripped.startswith("- "):
            story.append(Paragraph("&#8226; " + clean_inline(stripped[2:]), st["body"]))
        else:
            style = st["ref"] if re.match(r"^\[\d+\]", stripped) else st["body"]
            story.append(Paragraph(clean_inline(stripped), style))
    flush_table()


def add_csv_table(story, csv_path: Path, caption: str, st: dict):
    rows = list(csv.reader(csv_path.open(encoding="utf-8")))
    story.append(Paragraph(caption, st["caption"]))
    tbl = Table(rows, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def add_figure(story, fig: Path, caption: str, st: dict):
    story.append(Image(str(fig), width=6.2 * inch, height=3.6 * inch, kind="proportional"))
    story.append(Paragraph(caption, st["caption"]))


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(A4[0] / 2, 0.45 * inch, str(doc.page))
    canvas.restoreState()


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    st = styles()
    meta = read_metadata()
    out = BUILD / "Nishanth_Thesis_Final_Draft.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        rightMargin=0.9 * inch,
        leftMargin=0.9 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title=meta["title"],
        author=meta["candidate_name"],
    )
    story = []

    story.extend([Spacer(1, 1.0 * inch), Paragraph(meta["title"], st["title"]), Spacer(1, 0.25 * inch)])
    for item in [
        meta["candidate_name"],
        meta["registration_number"],
        "Thesis submitted in partial fulfillment of the requirements for the degree Research Project in Artificial Intelligence for the degree of BSc Hons in Artificial Intelligence",
        meta["department"],
        meta["faculty"],
        meta["university"],
        meta["country"],
        meta["submission_month_year"],
    ]:
        story.append(Paragraph(clean_inline(item), st["center"]))

    for fm in [
        "declaration.md",
        "dedication.md",
        "acknowledgement.md",
        "abstract_draft.md",
        "table_of_contents_draft.md",
        "list_of_figures.md",
        "list_of_tables.md",
        "list_of_abbreviations.md",
        "list_of_appendices.md",
    ]:
        add_markdown(story, THESIS / "front_matter" / fm, st)

    for chapter in sorted((THESIS / "chapters").glob("*.md")):
        add_markdown(story, chapter, st)
        if chapter.name == "04_approach.md":
            add_figure(story, FIGURES / "figure_4_1_research_workflow.png", "Figure 4.1: Research workflow for the proposed adaptive fusion framework.", st)
        elif chapter.name == "05_analysis_design.md":
            add_figure(story, FIGURES / "figure_5_1_system_architecture.png", "Figure 5.1: System architecture of the adaptive fusion framework.", st)
        elif chapter.name == "07_evaluation.md":
            add_csv_table(story, TABLES / "table_7_1_final_method_comparison.csv", "Table 7.1: Final test method comparison.", st)
            add_figure(story, FIGURES / "figure_7_1_method_macro_f1.png", "Figure 7.1: Final test macro F1 comparison across selected methods.", st)
            add_csv_table(story, TABLES / "table_7_2_threshold_tuning.csv", "Table 7.2: Validation-selected threshold tuning results.", st)
            add_figure(story, FIGURES / "figure_7_2_ablation_macro_f1.png", "Figure 7.2: Ablation results for different RL fusion state representations.", st)
            add_figure(story, FIGURES / "figure_7_4_robustness_image_quality.png", "Figure 7.4: Image quality degradation and image-weight adaptation.", st)

    add_markdown(story, THESIS / "references" / "initial_ieee_references.md", st)
    add_markdown(story, THESIS / "appendices" / "artifact_manifest.md", st)
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(out)


if __name__ == "__main__":
    build()
