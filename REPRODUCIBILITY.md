# Reproducibility Notes

This project uses two reproducibility layers.

## 1. GitHub Repository

The GitHub repository is intended to contain:

- source code,
- YAML configurations,
- tests,
- notebooks,
- thesis source,
- result summaries and comparison tables.

It is not intended to contain the full image corpus, raw dataset archive, trained checkpoints, or full explanation image folders.

## 2. External Artifact Bundle

The full experiment rerun requires a separate artifact bundle stored outside GitHub. This is expected for this project because the downloaded images and checkpoints are several gigabytes.

The external bundle should contain:

- raw and processed Fakeddit metadata,
- final split CSVs,
- downloaded image folders,
- modality-output CSVs,
- trained checkpoints,
- Grad-CAM/token-saliency explanation outputs,
- final metrics and exported tables.

See `ARTIFACT_MANIFEST.md` for the exact list.

## Checking Completeness

From the repository root:

```bash
python scripts/check_artifacts.py --profile git-evidence
python scripts/check_artifacts.py --profile final-full
```

Expected interpretation:

- `git-evidence` should pass in a normal cloned repository.
- `final-full` passes only after the external Google Drive artifact bundle is restored.

## Main Colab Path

The experiments were run under:

```text
/content/drive/MyDrive/fake-news-rl-fusion
```

If Colab starts in `/content`, mount Drive and change directory before running scripts:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/fake-news-rl-fusion
```

## Important Reporting Rule

Only report a result as completed when its evidence file exists. If a metric is not present in `outputs/metrics`, `outputs/robustness`, or `outputs/tables`, rerun the matching script or mark it as missing.
