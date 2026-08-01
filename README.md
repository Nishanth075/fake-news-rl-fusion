# fake-news-rl-fusion

Explainable fake-news detection using reinforcement-learning-based adaptive image-text fusion.

## Research Direction

This project implements the thesis framework:

**Explainable Deep Learning Framework for Fake News Detection using Reinforcement Learning-Based Adaptive Image-Text Fusion**

The proposed contribution is an offline contextual-bandit fusion controller that learns sample-specific image/text weights from modality probabilities, confidence, quality and disagreement. Results must be generated from real experiments only.

## Current Stage

Stage 1 is implemented:

- CSV-based paired image-text dataset preparation
- Standard schema: `sample_id,image_path,text,label`
- Label format: `0 = Real`, `1 = Fake`
- Validation for missing fields, empty text, duplicate IDs, duplicate text and duplicate image paths
- Reproducible stratified train/validation/test splitting
- Dataset statistics export
- Unit tests for the Stage 1 data pipeline

Model training is intentionally not included yet. We will add ResNet18 and DistilBERT after the dataset pipeline is verified in Colab.

## Repository Layout

```text
configs/              YAML experiment configuration
data/raw/             Original metadata files
data/processed/       Standardized dataset CSV
data/splits/          Train, validation and test CSVs
scripts/              Command-line entry points
src/                  Research source code
tests/                Unit tests
outputs/metrics/      Dataset statistics and later evaluation metrics
```

## Stage 1 Input Format

Place your raw metadata CSV at:

```text
data/raw/dataset.csv
```

By default it must contain these columns:

```text
sample_id,image_path,text,label
```

If the raw dataset uses different column names, edit:

```text
configs/data.yaml
```

For example, map Fakeddit metadata columns to the standard schema under `data.columns`.

## Run Stage 1

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset:

```bash
python scripts/prepare_dataset.py --config configs/data.yaml
```

Or through the pipeline runner:

```bash
python run_pipeline.py --prepare-data
```

Outputs:

```text
data/processed/dataset.csv
data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv
outputs/metrics/dataset_stats.json
```


## Fakeddit Metadata Conversion

For the official Fakeddit v2 multimodal-only TSV files, download the metadata folder into:

```text
data/raw/fakeddit_v2/
```

Then run:

```bash
python scripts/prepare_fakeddit.py --config configs/fakeddit.yaml
```

Or:

```bash
python run_pipeline.py --prepare-fakeddit
```

This reads:

```text
data/raw/fakeddit_v2/multimodal_only_samples/multimodal_train.tsv
data/raw/fakeddit_v2/multimodal_only_samples/multimodal_validate.tsv
data/raw/fakeddit_v2/multimodal_only_samples/multimodal_test_public.tsv
```

It writes:

```text
data/processed/fakeddit_dataset.csv
data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv
outputs/metrics/fakeddit_stats.json
```

Fakeddit's `2_way_label` is converted to this project's standard:

```text
Fakeddit 0 = False/Fake  -> project 1 = Fake
Fakeddit 1 = True/Real   -> project 0 = Real
```

## Debug Subset and Images

Create a small balanced subset for Colab smoke tests:

```bash
python scripts/create_debug_subset.py --config configs/debug_subset.yaml
```

This writes:

```text
data/debug_splits/train.csv
data/debug_splits/validation.csv
data/debug_splits/test.csv
```

Download only the debug subset images and create image-available splits:

```bash
python scripts/download_images.py --config configs/image_download.yaml
```

This also writes `data/debug_splits_available/` with only samples whose images downloaded successfully. Large image archives are intentionally avoided at the beginning. The first training check should use a small subset before scaling up.
## Colab Workflow

After pushing changes to GitHub, run this in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/fake-news-rl-fusion
!git pull
!pip install -r requirements.txt
!python scripts/prepare_dataset.py --config configs/data.yaml
```


## Stage 2 Image Model Smoke Test

After creating `data/debug_splits_available/`, verify the ResNet18 image branch with one train/evaluation batch:

```bash
python scripts/train_image_model.py --config configs/image_model.yaml --smoke-test
```

If the smoke test succeeds, run the small debug training configuration:

```bash
python scripts/train_image_model.py --config configs/image_model.yaml
```

The image model uses only the train split for training and selects the best checkpoint using validation macro F1.
Generate image-branch outputs for fusion experiments:

```bash
python scripts/infer_image_model.py --config configs/image_model.yaml --splits train validation test
```

This writes image probabilities, predictions and confidence values under `data/modality_outputs/`.

## Stage 3 Text Model Smoke Test

Verify the DistilBERT text branch with one train/evaluation batch:

```bash
python scripts/train_text_model.py --config configs/text_model.yaml --smoke-test
```

If the smoke test succeeds, run debug text training:

```bash
python scripts/train_text_model.py --config configs/text_model.yaml
```

Generate text-branch outputs for fusion experiments:

```bash
python scripts/infer_text_model.py --config configs/text_model.yaml --splits train validation test
```

This writes text probabilities, predictions and confidence values under `data/modality_outputs/`.

## Stage 4 Reliability and Baselines

Merge image/text predictions with image and text quality features:

```bash
python scripts/build_reliability_outputs.py --config configs/reliability.yaml
```

This writes:

```text
data/modality_outputs/train_outputs.csv
data/modality_outputs/validation_outputs.csv
data/modality_outputs/test_outputs.csv
```

Evaluate deterministic baselines:

```bash
python scripts/evaluate_baselines.py --config configs/baselines.yaml
```

Baseline metrics are saved under `outputs/metrics/`.
## Tests

Run:

```bash
pytest
```

## Research Rules

- Use paired image-text samples only.
- Do not fabricate accuracy, F1, reward, latency or explanation outputs.
- Do not train on the test set.
- Use validation data for model selection.
- Use the untouched test set only for final evaluation.
- Report negative or mixed findings honestly.







