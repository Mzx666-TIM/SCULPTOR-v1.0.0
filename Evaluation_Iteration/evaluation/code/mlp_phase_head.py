# -*- coding: utf-8 -*-
"""Deployable wrapper: deep MLP ensemble with a sklearn-like predict() API.

Emits the same joblib artifact contract as the tree phase-fusion heads
({"model", "feature_columns", "feature_schema_version", "metrics"}), so the
existing pipeline can load it without code changes. No predicted property is
used as an input feature.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from device_policy import resolve_device
except ImportError:  # pragma: no cover - device_policy is part of core/code
    resolve_device = None


class TorchMLPEnsemble:
    """Average of k identically-architected MLPs trained with different seeds."""

    def __init__(
        self,
        state_dicts: list[dict],
        input_mean: np.ndarray,
        input_std: np.ndarray,
        target_mean: float,
        target_std: float,
        hidden: tuple[int, ...] = (512, 512, 256),
        dropout: float = 0.1,
        q90: float | None = None,
        meta: dict | None = None,
    ) -> None:
        self.state_dicts = state_dicts
        self.input_mean = np.asarray(input_mean, dtype=np.float32)
        self.input_std = np.asarray(input_std, dtype=np.float32)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.hidden = tuple(hidden)
        self.dropout = float(dropout)
        self.q90 = None if q90 is None else float(q90)
        self.meta = dict(meta or {})

    def _build(self, in_dim: int) -> torch.nn.Module:
        layers, prev = [], in_dim
        for h in self.hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.GELU(),
                       torch.nn.Dropout(self.dropout)]
            prev = h
        layers += [torch.nn.Linear(prev, 1)]
        return torch.nn.Sequential(*layers)

    def predict(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        norm = (arr - self.input_mean) / self.input_std
        out = np.zeros((norm.shape[0],), dtype=np.float64)
        with torch.no_grad():
            t = torch.from_numpy(norm)
            for sd in self.state_dicts:
                net = self._build(arr.shape[1])
                net.load_state_dict(sd)
                net.eval()
                out += net(t).squeeze(-1).numpy().astype(np.float64)
        out /= max(len(self.state_dicts), 1)
        return out * self.target_std + self.target_mean


def fit_mlp_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seeds: tuple[int, ...] = (42, 43),
    hidden: tuple[int, ...] = (512, 512, 256),
    dropout: float = 0.1,
    epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    patience: int = 12,
    device: str | None = None,
    target: str = "",
    feature_set: str = "",
) -> tuple[TorchMLPEnsemble, dict[str, object]]:
    """Train the deployable phase-property ensemble used by the runtime.

    Each member uses the same standardized data and architecture but a
    different seed. The saved state for every member is the epoch with the
    lowest validation MSE; early stopping only limits compute.
    """

    train_x = np.asarray(x_train, dtype=np.float32)
    val_x = np.asarray(x_validation, dtype=np.float32)
    train_y = np.asarray(y_train, dtype=np.float32).reshape(-1)
    val_y = np.asarray(y_validation, dtype=np.float32).reshape(-1)
    if train_x.ndim != 2 or val_x.ndim != 2 or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("train and validation features must be 2D with equal width")
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("feature and target row counts do not match")
    if not seeds:
        raise ValueError("at least one ensemble seed is required")

    input_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    input_std[~np.isfinite(input_std) | (input_std < 1e-8)] = 1.0
    target_mean = float(train_y.mean(dtype=np.float64))
    target_std = float(train_y.std(dtype=np.float64))
    if not np.isfinite(target_std) or target_std < 1e-8:
        target_std = 1.0
    train_x_norm = (train_x - input_mean) / input_std
    val_x_norm = (val_x - input_mean) / input_std
    train_y_norm = (train_y - target_mean) / target_std
    val_y_norm = (val_y - target_mean) / target_std

    selected_device = (torch.device(device) if device
                       else (resolve_device() if resolve_device
                             else torch.device("cpu")))
    training_set = TensorDataset(
        torch.from_numpy(train_x_norm),
        torch.from_numpy(train_y_norm),
    )
    val_x_tensor = torch.from_numpy(val_x_norm).to(selected_device)
    val_y_tensor = torch.from_numpy(val_y_norm).to(selected_device)
    state_dicts: list[dict[str, torch.Tensor]] = []
    member_summaries: list[dict[str, object]] = []
    architecture = TorchMLPEnsemble(
        [], input_mean, input_std, target_mean, target_std,
        hidden=hidden, dropout=dropout,
    )

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if selected_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            training_set,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=selected_device.type == "cuda",
        )
        network = architecture._build(train_x.shape[1]).to(selected_device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
        criterion = torch.nn.SmoothL1Loss()
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            network.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(selected_device, non_blocking=True)
                batch_y = batch_y.to(selected_device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(network(batch_x).squeeze(-1), batch_y)
                loss.backward()
                optimizer.step()
            network.eval()
            with torch.no_grad():
                val_loss = float(
                    torch.mean(
                        (network(val_x_tensor).squeeze(-1) - val_y_tensor) ** 2
                    ).item()
                )
            if val_loss < best_loss - 1e-10:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        if best_state is None:
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in network.state_dict().items()
            }
        state_dicts.append(copy.deepcopy(best_state))
        member_summaries.append(
            {
                "seed": seed,
                "selected_epoch": best_epoch or epochs,
                "validation_mse_standardized": best_loss,
            }
        )

    ensemble = TorchMLPEnsemble(
        state_dicts=state_dicts,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        hidden=hidden,
        dropout=dropout,
        meta={
            "target": target,
            "feature_set": feature_set,
            "encoder": "raw descriptors (no STA, no property)",
            "selection": "lowest validation MSE per member",
        },
    )
    validation_predictions = ensemble.predict(val_x)
    q90 = float(np.quantile(np.abs(validation_predictions - val_y), 0.90))
    ensemble.q90 = q90
    return ensemble, {
        "device": str(selected_device),
        "members": member_summaries,
        "conformal90_halfwidth": q90,
    }
