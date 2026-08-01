from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.evaluation.metrics import binary_classification_metrics
from src.image_model.dataset import ImageNewsDataset
from src.image_model.model import build_image_model
from src.image_model.transforms import build_eval_transforms, build_train_transforms
from src.utils.device import get_device
from src.utils.file_io import write_json
from src.utils.seed import set_seed


def train_image_model(config: dict[str, Any], smoke_test: bool = False) -> dict[str, Any]:
    """Train the ResNet18 image branch and save best validation checkpoint."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    image_config = config["image_model"]
    paths = config["paths"]
    device = get_device()

    train_loader = _build_loader(config, paths["train_csv"], train=True)
    validation_loader = _build_loader(config, paths["validation_csv"], train=False)

    model = build_image_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(image_config["learning_rate"]),
        weight_decay=float(image_config["weight_decay"]),
    )

    checkpoint_dir = Path(paths["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_image_model.pt"

    epochs = 1 if smoke_test else int(image_config["epochs"])
    patience = int(image_config.get("early_stopping_patience", 3))
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, smoke_test=smoke_test
        )
        validation_loss, labels, probabilities = _evaluate(
            model, validation_loader, criterion, device, smoke_test=smoke_test
        )
        metrics = binary_classification_metrics(labels, probabilities)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_metrics": metrics,
        }
        history.append(epoch_record)

        macro_f1 = float(metrics["macro_f1"])
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "validation_macro_f1": best_f1,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if not smoke_test and epochs_without_improvement >= patience:
            break

    result = {
        "device": str(device),
        "smoke_test": smoke_test,
        "best_validation_macro_f1": best_f1,
        "best_checkpoint_path": str(best_checkpoint_path),
        "history": history,
    }
    write_json(result, paths["metrics_path"])
    return result


def _build_loader(config: dict[str, Any], csv_path: str, train: bool) -> DataLoader:
    image_config = config["image_model"]
    paths = config["paths"]
    df = pd.read_csv(csv_path)
    transform = (
        build_train_transforms(int(image_config["resize_size"]), int(image_config["image_size"]))
        if train
        else build_eval_transforms(int(image_config["resize_size"]), int(image_config["image_size"]))
    )
    dataset = ImageNewsDataset(df, image_root=paths.get("image_root", "."), transform=transform)
    return DataLoader(
        dataset,
        batch_size=int(image_config["batch_size"]),
        shuffle=train,
        num_workers=int(image_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    smoke_test: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        if smoke_test and batch_index >= 0:
            break
    return total_loss / max(total_examples, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    smoke_test: bool,
) -> tuple[float, list[int], list[float]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[int] = []
    all_probabilities: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="validation", leave=False)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs["logits"], labels)
            probabilities = outputs["fake_probability"].detach().cpu().numpy().tolist()
            batch_labels = labels.detach().cpu().numpy().astype(int).tolist()
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            all_labels.extend(batch_labels)
            all_probabilities.extend(probabilities)
            if smoke_test and batch_index >= 0:
                break
    return total_loss / max(total_examples, 1), all_labels, all_probabilities
