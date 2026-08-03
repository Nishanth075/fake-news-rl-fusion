from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps

from src.image_model.model import build_image_model
from src.image_model.transforms import build_eval_transforms
from src.utils.device import get_device
from src.utils.file_io import write_json
from src.utils.seed import set_seed


def run_explainability(config: dict[str, Any]) -> dict[str, Any]:
    """Generate sample-level image Grad-CAM and text saliency explanations."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    explain_config = config["explainability"]
    output_dir = Path(explain_config["output_dir"])
    image_dir = output_dir / "image_gradcam"
    text_dir = output_dir / "text_saliency"
    image_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    rows = _select_rows(config)
    image_model = _load_image_model(config, device)
    text_model = _load_text_model(config, device)
    tokenizer = _build_tokenizer(config)

    summary_rows = []
    for row in rows.to_dict("records"):
        image_explanation = explain_image_sample(config, image_model, row, device, image_dir)
        text_explanation = explain_text_sample(config, text_model, tokenizer, row, device, text_dir)
        summary_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "label": int(row["label"]),
                "image_probability": _optional_float(row, "image_probability"),
                "text_probability": _optional_float(row, "text_probability"),
                "final_probability": _optional_float(row, "final_probability"),
                "final_prediction": _optional_int(row, "final_prediction"),
                "selected_action": _optional_int(row, "selected_action"),
                "image_weight": _optional_float(row, "image_weight"),
                "text_weight": _optional_float(row, "text_weight"),
                "image_gradcam_path": image_explanation["overlay_path"],
                "text_saliency_path": text_explanation["token_scores_path"],
                "top_text_tokens": text_explanation["top_tokens"],
            }
        )

    summary = {
        "device": str(device),
        "rows": len(summary_rows),
        "output_dir": str(output_dir),
        "summary_csv": str(output_dir / "explanation_summary.csv"),
        "samples": summary_rows,
    }
    pd.DataFrame(summary_rows).to_csv(output_dir / "explanation_summary.csv", index=False)
    write_json(summary, explain_config["summary_path"])
    return summary


def explain_image_sample(
    config: dict[str, Any],
    model: torch.nn.Module,
    row: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, str]:
    """Generate a Grad-CAM overlay for one image sample."""
    image_config = config["image_model"]
    paths = config["paths"]
    image_path = Path(str(row["image_path"]))
    if not image_path.is_absolute():
        image_path = Path(paths.get("image_root", ".")) / image_path

    raw_image = Image.open(image_path).convert("RGB")
    transform = build_eval_transforms(int(image_config["resize_size"]), int(image_config["image_size"]))
    image_tensor = transform(raw_image).unsqueeze(0).to(device)

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
    logit = outputs["logits"][0]
    predicted_label = int(outputs["predicted_label"][0].detach().cpu().item())
    target_score = logit if predicted_label == 1 else -logit
    target_score.backward()

    forward_handle.remove()
    backward_handle.remove()

    cam = compute_gradcam(activations[0], gradients[0])
    overlay = overlay_heatmap(raw_image, cam)
    overlay_path = output_dir / f"{row['sample_id']}_gradcam.png"
    overlay.save(overlay_path)
    return {"overlay_path": str(overlay_path)}


def compute_gradcam(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    """Compute a normalized Grad-CAM map from layer activations and gradients."""
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=False))[0]
    cam = cam.detach().cpu()
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max <= cam_min:
        return torch.zeros_like(cam)
    return (cam - cam_min) / (cam_max - cam_min)


def overlay_heatmap(image: Image.Image, cam: torch.Tensor) -> Image.Image:
    """Blend a normalized Grad-CAM tensor over the original image."""
    cam_image = Image.fromarray((cam.numpy() * 255).astype("uint8")).resize(image.size)
    heatmap = ImageOps.colorize(cam_image, black="#00004c", white="#ff2a00").convert("RGB")
    return Image.blend(image.convert("RGB"), heatmap, alpha=0.42)


def explain_text_sample(
    config: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate gradient-based token saliency for one text sample."""
    text_config = config["text_model"]
    encoded = tokenizer(
        str(row["text"]),
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
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    logits = model.classifier(cls_embedding)
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = int(torch.argmax(probabilities, dim=1)[0].detach().cpu().item())
    logits[0, predicted_label].backward()

    saliency = (embeddings.grad[0] * embeddings.detach()[0]).abs().sum(dim=1)
    mask = attention_mask[0].detach().bool()
    saliency = saliency.detach().cpu()
    token_ids = input_ids[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    rows = []
    valid_scores = saliency[mask.cpu()]
    score_max = float(valid_scores.max()) if len(valid_scores) else 0.0
    for token, score, keep in zip(tokens, saliency.tolist(), mask.cpu().tolist()):
        if not keep or token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        normalized = float(score / score_max) if score_max > 0 else 0.0
        rows.append({"token": token, "importance": normalized})

    token_df = pd.DataFrame(rows)
    token_scores_path = output_dir / f"{row['sample_id']}_text_saliency.csv"
    token_df.to_csv(token_scores_path, index=False)
    top_tokens = token_df.sort_values("importance", ascending=False).head(8)["token"].tolist()
    return {"token_scores_path": str(token_scores_path), "top_tokens": top_tokens}


def _select_rows(config: dict[str, Any]) -> pd.DataFrame:
    explain_config = config["explainability"]
    split_df = pd.read_csv(explain_config["split_csv"])
    sample_count = int(explain_config.get("sample_count", 12))
    seed = int(config.get("seed", 42))

    outputs_path = explain_config.get("fusion_predictions_csv")
    if outputs_path and Path(outputs_path).exists():
        fusion_df = pd.read_csv(outputs_path)
        split_df = split_df.merge(fusion_df, on=["sample_id", "label"], how="left")

    image_outputs = explain_config.get("image_outputs_csv")
    if image_outputs and Path(image_outputs).exists() and "image_probability" not in split_df.columns:
        split_df = split_df.merge(pd.read_csv(image_outputs), on=["sample_id", "label"], how="left")

    text_outputs = explain_config.get("text_outputs_csv")
    if text_outputs and Path(text_outputs).exists() and "text_probability" not in split_df.columns:
        split_df = split_df.merge(pd.read_csv(text_outputs), on=["sample_id", "label"], how="left")

    stratified = (
        split_df.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(min(len(group), max(1, sample_count // 2)), random_state=seed))
        .reset_index(drop=True)
    )
    if len(stratified) < sample_count:
        remaining = split_df[~split_df["sample_id"].isin(stratified["sample_id"])]
        extra = remaining.sample(min(len(remaining), sample_count - len(stratified)), random_state=seed)
        stratified = pd.concat([stratified, extra], ignore_index=True)
    return stratified.head(sample_count)


def _load_image_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(config["paths"]["image_checkpoint_dir"]) / "best_image_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Image checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_image_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_text_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(config["paths"]["text_checkpoint_dir"]) / "best_text_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Text checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _build_text_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _build_text_model(config: dict[str, Any]) -> torch.nn.Module:
    from src.text_model.model import build_text_model

    return build_text_model(config)


def _build_tokenizer(config: dict[str, Any]) -> Any:
    from src.text_model.model import build_tokenizer

    return build_tokenizer(config)


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if pd.isna(value):
        return None
    return float(value)


def _optional_int(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if pd.isna(value):
        return None
    return int(value)


