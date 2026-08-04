from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.explainability.explain import (
    _build_tokenizer,
    _load_image_model,
    _load_text_model,
    _select_rows,
    compute_gradcam,
)
from src.image_model.transforms import build_eval_transforms
from src.utils.device import get_device
from src.utils.file_io import write_json
from src.utils.seed import set_seed


def run_explainability_faithfulness(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate image and text explanations with deletion-style faithfulness tests.

    For each selected sample, the most salient image region or text tokens are removed.
    A useful explanation should identify evidence whose removal decreases the model's
    probability for its originally predicted class.
    """
    seed = int(config.get("seed", 42))
    set_seed(seed)
    faithfulness_config = config["explainability_faithfulness"]
    device = get_device()
    rows = _select_rows(config)

    image_model = _load_image_model(config, device)
    text_model = _load_text_model(config, device)
    tokenizer = _build_tokenizer(config)

    image_fraction = float(faithfulness_config.get("image_mask_fraction", 0.15))
    text_fraction = float(faithfulness_config.get("text_mask_fraction", 0.25))
    rng = np.random.default_rng(seed)

    summary_rows: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        image_result = image_deletion_test(config, image_model, row, device, image_fraction, rng)
        text_result = text_deletion_test(config, text_model, tokenizer, row, device, text_fraction, rng)
        summary_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "label": int(row["label"]),
                **{f"image_{key}": value for key, value in image_result.items()},
                **{f"text_{key}": value for key, value in text_result.items()},
            }
        )

    output_path = Path(faithfulness_config["summary_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    aggregate = _aggregate(summary_rows)
    result = {
        "device": str(device),
        "rows": len(summary_rows),
        "summary_csv": str(output_path),
        "aggregate": aggregate,
        "metric_note": (
            "comprehensiveness = original predicted-class probability minus probability after deleting "
            "salient evidence; larger positive values indicate more faithful explanations."
        ),
    }
    write_json(result, faithfulness_config["summary_json"])
    return result


def image_deletion_test(
    config: dict[str, Any],
    model: torch.nn.Module,
    row: dict[str, Any],
    device: torch.device,
    mask_fraction: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    image = _load_raw_image(config, row)
    cam, predicted_label, original_probability = _image_cam_and_probability(config, model, image, device)
    salient_image = _mask_image_by_cam(image, cam, mask_fraction)
    random_image = _mask_image_random(image, mask_fraction, rng)
    salient_probability = _score_image_probability(config, model, salient_image, device, predicted_label)
    random_probability = _score_image_probability(config, model, random_image, device, predicted_label)
    return {
        "predicted_label": predicted_label,
        "original_target_probability": original_probability,
        "salient_deleted_probability": salient_probability,
        "random_deleted_probability": random_probability,
        "salient_comprehensiveness": original_probability - salient_probability,
        "random_comprehensiveness": original_probability - random_probability,
    }


def text_deletion_test(
    config: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    device: torch.device,
    mask_fraction: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    text = str(row["text"])
    saliency = _text_token_saliency(config, model, tokenizer, text, device)
    predicted_label = int(saliency["predicted_label"])
    original_probability = float(saliency["target_probability"])
    token_count = len(saliency["tokens"])
    delete_count = max(1, int(round(token_count * mask_fraction))) if token_count else 0
    salient_indices = np.argsort(saliency["scores"])[-delete_count:] if delete_count else np.array([], dtype=int)
    random_indices = rng.choice(token_count, size=delete_count, replace=False) if delete_count and token_count else []
    salient_text = _delete_token_indices(saliency["tokens"], set(int(i) for i in salient_indices))
    random_text = _delete_token_indices(saliency["tokens"], set(int(i) for i in random_indices))
    salient_probability = _score_text_probability(config, model, tokenizer, salient_text, device, predicted_label)
    random_probability = _score_text_probability(config, model, tokenizer, random_text, device, predicted_label)
    return {
        "predicted_label": predicted_label,
        "original_target_probability": original_probability,
        "salient_deleted_probability": salient_probability,
        "random_deleted_probability": random_probability,
        "salient_comprehensiveness": original_probability - salient_probability,
        "random_comprehensiveness": original_probability - random_probability,
        "deleted_token_count": int(delete_count),
    }


def _load_raw_image(config: dict[str, Any], row: dict[str, Any]) -> Image.Image:
    image_path = Path(str(row["image_path"]))
    if not image_path.is_absolute():
        image_path = Path(config["paths"].get("image_root", ".")) / image_path
    return Image.open(image_path).convert("RGB")


def _image_cam_and_probability(
    config: dict[str, Any],
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> tuple[torch.Tensor, int, float]:
    image_config = config["image_model"]
    transform = build_eval_transforms(int(image_config["resize_size"]), int(image_config["image_size"]))
    image_tensor = transform(image).unsqueeze(0).to(device)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.backbone.layer4[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    model.zero_grad(set_to_none=True)
    outputs = model(image_tensor)
    fake_probability = torch.sigmoid(outputs["logits"])[0]
    predicted_label = int((fake_probability >= 0.5).detach().cpu().item())
    target_probability = float(
        fake_probability.detach().cpu().item()
        if predicted_label == 1
        else (1.0 - fake_probability).detach().cpu().item()
    )
    target_score = outputs["logits"][0] if predicted_label == 1 else -outputs["logits"][0]
    target_score.backward()
    forward_handle.remove()
    backward_handle.remove()
    return compute_gradcam(activations[0], gradients[0]), predicted_label, target_probability


def _score_image_probability(
    config: dict[str, Any],
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    target_label: int,
) -> float:
    image_config = config["image_model"]
    transform = build_eval_transforms(int(image_config["resize_size"]), int(image_config["image_size"]))
    with torch.no_grad():
        outputs = model(transform(image).unsqueeze(0).to(device))
        fake_probability = torch.sigmoid(outputs["logits"])[0]
    value = fake_probability if target_label == 1 else 1.0 - fake_probability
    return float(value.detach().cpu().item())


def _mask_image_by_cam(image: Image.Image, cam: torch.Tensor, fraction: float) -> Image.Image:
    cam_array = np.asarray(Image.fromarray((cam.numpy() * 255).astype("uint8")).resize(image.size), dtype=float)
    threshold = np.quantile(cam_array, max(0.0, min(1.0, 1.0 - fraction)))
    mask = cam_array >= threshold
    array = np.asarray(image).copy()
    mean_color = array.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    array[mask] = mean_color
    return Image.fromarray(array)


def _mask_image_random(image: Image.Image, fraction: float, rng: np.random.Generator) -> Image.Image:
    array = np.asarray(image).copy()
    mask = rng.random(array.shape[:2]) < fraction
    mean_color = array.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    array[mask] = mean_color
    return Image.fromarray(array)


def _text_token_saliency(
    config: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    device: torch.device,
) -> dict[str, Any]:
    text_config = config["text_model"]
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=int(text_config["max_length"]),
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    embeddings = model.backbone.embeddings.word_embeddings(input_ids)
    embeddings.retain_grad()
    model.zero_grad(set_to_none=True)
    outputs = model.backbone(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = model.classifier(outputs.last_hidden_state[:, 0, :])
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = int(torch.argmax(probabilities, dim=1)[0].detach().cpu().item())
    target_probability = float(probabilities[0, predicted_label].detach().cpu().item())
    logits[0, predicted_label].backward()
    saliency = (embeddings.grad[0] * embeddings.detach()[0]).abs().sum(dim=1).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    keep_mask = attention_mask[0].detach().cpu().numpy().astype(bool)
    valid_tokens: list[str] = []
    valid_scores: list[float] = []
    for token, score, keep in zip(tokens, saliency, keep_mask):
        if keep and token not in {"[CLS]", "[SEP]", "[PAD]"}:
            valid_tokens.append(token)
            valid_scores.append(float(score))
    return {
        "tokens": valid_tokens,
        "scores": np.asarray(valid_scores, dtype=float),
        "predicted_label": predicted_label,
        "target_probability": target_probability,
    }


def _score_text_probability(
    config: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    device: torch.device,
    target_label: int,
) -> float:
    text_config = config["text_model"]
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=int(text_config["max_length"]),
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
        probabilities = torch.softmax(outputs["logits"], dim=1)
    return float(probabilities[0, target_label].detach().cpu().item())


def _delete_token_indices(tokens: list[str], indices: set[int]) -> str:
    kept = [token for index, token in enumerate(tokens) if index not in indices]
    text = " ".join(token.replace("##", "") for token in kept)
    return text if text.strip() else "[UNK]"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {
        "image_salient_comprehensiveness_mean": float(df["image_salient_comprehensiveness"].mean()),
        "image_random_comprehensiveness_mean": float(df["image_random_comprehensiveness"].mean()),
        "text_salient_comprehensiveness_mean": float(df["text_salient_comprehensiveness"].mean()),
        "text_random_comprehensiveness_mean": float(df["text_random_comprehensiveness"].mean()),
    }

