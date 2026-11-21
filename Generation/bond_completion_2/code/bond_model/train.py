from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from bond_model.dataset import NODE_FEATURE_DIM, BondCompletionDataset
from bond_model.model import BondCompletionModel

try:
    from device_policy import resolve_device
except ImportError:  # pragma: no cover - device_policy is part of core/code
    resolve_device = None


LOSS_POLICY_VERSION = "balanced_bce_molecule_split_v2"


def _split_dataset(dataset: BondCompletionDataset, val_fraction: float, seed: int) -> tuple[list[int], set[int]]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if val_fraction == 0:
        return list(range(len(dataset))), set()
    if not all(dataset.molecule_groups):
        raise ValueError("validation requires canonical_smiles for every row; re-export from the full molecular samples CSV")
    groups = sorted(set(dataset.molecule_groups))
    if len(groups) < 2:
        raise ValueError("molecular validation requires at least two distinct molecules")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, int(len(groups) * val_fraction)))
    validation_groups = set(groups[:count])
    val = {i for i,g in enumerate(dataset.molecule_groups) if g in validation_groups}
    train = [i for i in range(len(dataset)) if i not in val]
    return train, val


@dataclass(frozen=True)
class TrainingResult:
    train_loss: float
    val_loss: float | None
    val_accuracy: float | None
    epochs: int
    checkpoint_path: str | None
    start_epoch: int = 0
    device: str = "cpu"
    training_rows: int = 0
    validation_rows: int = 0
    history_path: str | None = None
    val_precision: float | None = None
    val_recall: float | None = None
    val_f1: float | None = None
    recommended_threshold: float | None = None
    val_best_f1: float | None = None
    positive_label_fraction: float = 0.0
    positive_class_weight: float = 1.0


def train_bond_model(
    csv_path: str | Path,
    checkpoint_path: str | Path | None = None,
    epochs: int = 20,
    batch_size: int = 16,
    hidden_dim: int = 64,
    num_layers: int = 3,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
    resume: bool = False,
    checkpoint_every: int = 1,
    num_workers: int = 0,
    use_amp: bool = True,
    model_role: str = "half_graph_completion",
) -> TrainingResult:
    random.seed(seed)
    torch.manual_seed(seed)
    if model_role not in ("base_to_semi_base", "half_graph_completion"):
        raise ValueError(f"unsupported bond model role: {model_role}")

    dataset = BondCompletionDataset(csv_path)
    if len(dataset) == 0:
        raise ValueError("training CSV contains no rows")

    train_indexes, val_indexes = _split_dataset(dataset, val_fraction, seed)
    if checkpoint_path is not None:
        split_path = Path(checkpoint_path).with_suffix(".split.json")
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps({"policy": LOSS_POLICY_VERSION, "seed":seed,
            "validation_fraction":val_fraction, "training_indices":train_indexes,
            "validation_indices":sorted(val_indexes), "source_csv":str(csv_path)}, indent=2), encoding="utf-8")

    device = resolve_device() if resolve_device else torch.device("cpu")
    pin_memory = device.type == "cuda"
    loader_options = {
        "batch_size": batch_size,
        "num_workers": max(int(num_workers), 0),
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(
        Subset(dataset, train_indexes),
        shuffle=True,
        **loader_options,
    )
    val_loader = (
        DataLoader(
            Subset(dataset, sorted(val_indexes)),
            shuffle=False,
            **loader_options,
        )
        if val_indexes
        else None
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = BondCompletionModel(
        node_feature_dim=NODE_FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    training_positive = sum(dataset.label_counts[i][0] for i in train_indexes)
    training_negative = sum(dataset.label_counts[i][1] for i in train_indexes)
    total_edge_labels = training_positive + training_negative
    positive_fraction = training_positive / max(total_edge_labels, 1)
    positive_class_weight = (
        training_negative / training_positive
        if training_positive > 0
        else 1.0
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_class_weight],
            dtype=torch.float32,
            device=device,
        )
    )
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    history: list[dict[str, float | int | None]] = []
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    history_path = (
        checkpoint_file.with_suffix(checkpoint_file.suffix + ".history.json")
        if checkpoint_file is not None
        else None
    )
    if resume and checkpoint_file is not None and checkpoint_file.exists():
        saved = torch.load(checkpoint_file, map_location=device, weights_only=False)
        if int(saved.get("dataset_size", len(dataset))) != len(dataset):
            raise ValueError("checkpoint dataset size does not match training CSV")
        if saved.get("loss_policy_version") == LOSS_POLICY_VERSION:
            model.load_state_dict(
                saved.get("resume_model_state_dict", saved["model_state_dict"])
            )
            if "optimizer_state_dict" in saved:
                optimizer.load_state_dict(saved["optimizer_state_dict"])
            if "scaler_state_dict" in saved and amp_enabled:
                scaler.load_state_dict(saved["scaler_state_dict"])
            start_epoch = int(saved.get("completed_epochs", saved.get("epochs", 0)))
            history = list(saved.get("history") or [])
        else:
            print(
                "bond checkpoint uses an older loss policy; restarting at epoch 1",
                flush=True,
            )

    train_loss = 0.0
    val_loss: float | None = None
    val_accuracy: float | None = None
    val_precision: float | None = None
    val_recall: float | None = None
    val_f1: float | None = None
    recommended_threshold: float | None = None
    val_best_f1: float | None = None
    selected_state_dict: dict[str, torch.Tensor] | None = None
    selected_epoch: int | None = None
    selected_metrics: dict[str, float | None] | None = None
    if resume and checkpoint_file is not None and checkpoint_file.exists():
        saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if saved.get("loss_policy_version") == LOSS_POLICY_VERSION:
            selected_state_dict = _state_dict_to_cpu(saved["model_state_dict"])
            selected_epoch = int(saved.get("selected_epoch") or saved.get("completed_epochs", 0))
            selected_metrics = {
                "val_loss": saved.get("selected_val_loss", saved.get("val_loss")),
                "val_accuracy": saved.get("selected_val_accuracy", saved.get("val_accuracy")),
                "val_precision": saved.get("selected_val_precision", saved.get("val_precision")),
                "val_recall": saved.get("selected_val_recall", saved.get("val_recall")),
                "val_f1": saved.get("selected_val_f1", saved.get("val_f1")),
                "recommended_threshold": saved.get("recommended_threshold"),
                "val_best_f1": saved.get("val_best_f1"),
            }
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_labels = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(batch)
                loss = criterion(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * int(batch.y.numel())
            total_labels += int(batch.y.numel())
        train_loss = total_loss / max(total_labels, 1)
        if val_loader is not None:
            (
                val_loss,
                val_accuracy,
                val_precision,
                val_recall,
                val_f1,
                recommended_threshold,
                val_best_f1,
            ) = _evaluate(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled,
                non_blocking=pin_memory,
            )
            if (
                selected_metrics is None
                or val_best_f1 > float(selected_metrics.get("val_best_f1") or -1.0)
            ):
                selected_state_dict = _state_dict_to_cpu(model.state_dict())
                selected_epoch = epoch + 1
                selected_metrics = {
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "recommended_threshold": recommended_threshold,
                    "val_best_f1": val_best_f1,
                }
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "recommended_threshold": recommended_threshold,
                "val_best_f1": val_best_f1,
            }
        )
        print(
            (
                f"bond training epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.6f}, "
                f"val_loss={val_loss if val_loss is not None else 'n/a'}, "
                f"val_accuracy={val_accuracy if val_accuracy is not None else 'n/a'}, "
                f"val_precision={val_precision if val_precision is not None else 'n/a'}, "
                f"val_recall={val_recall if val_recall is not None else 'n/a'}, "
                f"val_f1={val_f1 if val_f1 is not None else 'n/a'}"
            ),
            flush=True,
        )
        if (
            checkpoint_file is not None
            and checkpoint_every > 0
            and ((epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs)
        ):
            _save_checkpoint(
                checkpoint_file,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                completed_epochs=epoch + 1,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dataset_size=len(dataset),
                history=history,
                positive_label_fraction=positive_fraction,
                positive_class_weight=positive_class_weight,
                recommended_threshold=recommended_threshold,
                val_best_f1=val_best_f1,
                model_role=model_role,
                selected_state_dict=selected_state_dict,
                selected_epoch=selected_epoch,
                selected_metrics=selected_metrics,
            )
            if history_path is not None:
                history_path.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    saved_path: str | None = None
    if checkpoint_file is not None:
        if start_epoch >= epochs:
            if val_loader is not None:
                (
                    val_loss,
                    val_accuracy,
                    val_precision,
                    val_recall,
                    val_f1,
                    recommended_threshold,
                    val_best_f1,
                ) = _evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    amp_enabled=amp_enabled,
                    non_blocking=pin_memory,
                )
            train_loss = float(history[-1]["train_loss"]) if history else 0.0
        _save_checkpoint(
            checkpoint_file,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            completed_epochs=max(epochs, start_epoch),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dataset_size=len(dataset),
            history=history,
            positive_label_fraction=positive_fraction,
            positive_class_weight=positive_class_weight,
            recommended_threshold=recommended_threshold,
            val_best_f1=val_best_f1,
            model_role=model_role,
            selected_state_dict=selected_state_dict,
            selected_epoch=selected_epoch,
            selected_metrics=selected_metrics,
        )
        saved_path = str(checkpoint_file)

    if selected_metrics is not None:
        val_loss = selected_metrics["val_loss"]
        val_accuracy = selected_metrics["val_accuracy"]
        val_precision = selected_metrics["val_precision"]
        val_recall = selected_metrics["val_recall"]
        val_f1 = selected_metrics["val_f1"]
        recommended_threshold = selected_metrics["recommended_threshold"]
        val_best_f1 = selected_metrics["val_best_f1"]

    return TrainingResult(
        train_loss=train_loss,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        epochs=epochs,
        checkpoint_path=saved_path,
        start_epoch=start_epoch,
        device=str(device),
        training_rows=len(train_indexes),
        validation_rows=len(val_indexes),
        history_path=None if history_path is None else str(history_path),
        val_precision=val_precision,
        val_recall=val_recall,
        val_f1=val_f1,
        recommended_threshold=recommended_threshold,
        val_best_f1=val_best_f1,
        positive_label_fraction=positive_fraction,
        positive_class_weight=positive_class_weight,
    )


def result_to_json(result: TrainingResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False)


def _evaluate(
    model: BondCompletionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
    non_blocking: bool = False,
) -> tuple[float, float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_labels = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    probability_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=non_blocking)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(batch)
                loss = criterion(logits, batch.y)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            probability_chunks.append(torch.sigmoid(logits).detach().cpu())
            label_chunks.append(batch.y.detach().cpu())
            correct += int((predictions == batch.y).sum().item())
            true_positive += int(((predictions == 1) & (batch.y == 1)).sum().item())
            false_positive += int(((predictions == 1) & (batch.y == 0)).sum().item())
            false_negative += int(((predictions == 0) & (batch.y == 1)).sum().item())
            total_loss += float(loss.item()) * int(batch.y.numel())
            total_labels += int(batch.y.numel())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    recommended_threshold, best_f1 = _best_f1_threshold(
        torch.cat(probability_chunks),
        torch.cat(label_chunks),
    )
    return (
        total_loss / max(total_labels, 1),
        correct / max(total_labels, 1),
        precision,
        recall,
        f1,
        recommended_threshold,
        best_f1,
    )


def _best_f1_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = 0.0
    positive = labels >= 0.5
    for threshold_index in range(5, 96):
        threshold = threshold_index / 100.0
        predicted = probabilities >= threshold
        true_positive = int((predicted & positive).sum().item())
        false_positive = int((predicted & ~positive).sum().item())
        false_negative = int((~predicted & positive).sum().item())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
    return best_threshold, best_f1


def _save_checkpoint(
    path: Path,
    *,
    model: BondCompletionModel,
    optimizer: torch.optim.Optimizer,
    scaler,
    completed_epochs: int,
    hidden_dim: int,
    num_layers: int,
    dataset_size: int,
    history: list[dict[str, float | int | None]],
    positive_label_fraction: float,
    positive_class_weight: float,
    recommended_threshold: float | None,
    val_best_f1: float | None,
    model_role: str,
    selected_state_dict: dict[str, torch.Tensor] | None,
    selected_epoch: int | None,
    selected_metrics: dict[str, float | None] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            # The primary state is the deployable model selected by calibrated
            # validation F1. The last-epoch state is kept separately so a
            # resumed run follows the exact optimizer trajectory.
            "model_state_dict": selected_state_dict or _state_dict_to_cpu(model.state_dict()),
            "resume_model_state_dict": _state_dict_to_cpu(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "node_feature_dim": NODE_FEATURE_DIM,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "epochs": completed_epochs,
            "completed_epochs": completed_epochs,
            "dataset_size": dataset_size,
            "history": history,
            "loss_policy_version": LOSS_POLICY_VERSION,
            "positive_label_fraction": positive_label_fraction,
            "positive_class_weight": positive_class_weight,
            "recommended_threshold": (
                selected_metrics.get("recommended_threshold")
                if selected_metrics is not None
                else recommended_threshold
            ),
            "val_best_f1": (
                selected_metrics.get("val_best_f1")
                if selected_metrics is not None
                else val_best_f1
            ),
            "selected_epoch": selected_epoch or completed_epochs,
            "checkpoint_selection": "max_validation_calibrated_f1",
            "selected_val_loss": None if selected_metrics is None else selected_metrics.get("val_loss"),
            "selected_val_accuracy": None if selected_metrics is None else selected_metrics.get("val_accuracy"),
            "selected_val_precision": None if selected_metrics is None else selected_metrics.get("val_precision"),
            "selected_val_recall": None if selected_metrics is None else selected_metrics.get("val_recall"),
            "selected_val_f1": None if selected_metrics is None else selected_metrics.get("val_f1"),
            "model_role": model_role,
        },
        temporary,
    )
    temporary.replace(path)


def _state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach and clone a state dict so later training cannot mutate it."""

    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}
