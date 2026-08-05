# LaTeX Thesis Project

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
