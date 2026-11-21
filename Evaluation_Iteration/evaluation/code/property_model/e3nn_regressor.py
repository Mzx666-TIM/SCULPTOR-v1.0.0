from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from property_model.data import GeometryRecord, PropertyRecord, geometry_to_tensors, records_from_csv


def require_e3nn():
    try:
        from e3nn import o3
    except ImportError as exc:
        raise RuntimeError(
            "e3nn is required for the 3D property model. Install it with "
            "`pip install e3nn` in the project environment before training."
        ) from exc
    return o3


@dataclass(frozen=True)
class PropertyTrainingResult:
    train_loss: float
    val_loss: float | None
    epochs: int
    checkpoint_path: str | None
    rows: int
    train_mae: float
    val_mae: float | None
    train_baseline_mae: float
    val_baseline_mae: float | None
    target_mean: float
    target_std: float
    baseline_mode: str
    baseline_coefficients: tuple[float, ...]


class E3NNInvariantRegressor(nn.Module):
    """A lightweight e3nn-based invariant regressor for early property screening."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@lru_cache(maxsize=8192)
def molecular_e3nn_features(
    geometry: GeometryRecord,
    lmax: int = 3,
    radial_bins: int = 8,
    cutoff: float = 5.0,
) -> torch.Tensor:
    """Build invariant molecular features using e3nn spherical harmonics."""

    o3 = require_e3nn()
    atomic_numbers, positions = geometry_to_tensors(geometry)
    if positions.numel() == 0:
        raise ValueError("geometry has no atoms")

    centered = positions - positions.mean(dim=0, keepdim=True)
    n_atoms = float(positions.shape[0])
    carbon_count = float((atomic_numbers == 6).sum().item())
    hydrogen_count = float((atomic_numbers == 1).sum().item())

    # Vectorized atom-pair computation: all pair distances and spherical
    # harmonics at once, then per-l block norms. The original per-pair loop's
    # by_l norms are direction-invariant (|Y_lm(-v)| = |Y_lm(v)|) and include
    # both (i,j) and (j,i), so mean/std match the original exactly.
    diffs = centered[:, None, :] - centered[None, :, :]
    distances = torch.linalg.norm(diffs, dim=-1)
    valid = (distances > 1e-6) & (distances <= cutoff)
    pair_vectors = diffs[valid]
    pair_distances = distances[valid]
    if pair_vectors.numel():
        sh = o3.spherical_harmonics(
            list(range(lmax + 1)),
            pair_vectors,
            normalize=True,
            normalization="component",
        )
        by_l = []
        cursor = 0
        for degree in range(lmax + 1):
            width = 2 * degree + 1
            block = sh[:, cursor : cursor + width]
            by_l.append(block.norm(dim=-1, keepdim=True))
            cursor += width
        radial = _radial_features(pair_distances, bins=radial_bins, cutoff=cutoff)
        edge_tensor = torch.cat(
            [pair_distances.unsqueeze(-1), *by_l, radial],
            dim=-1,
        )
        edge_summary = torch.cat([edge_tensor.mean(dim=0), edge_tensor.std(dim=0)])
    else:
        width = 1 + (lmax + 1) + radial_bins
        edge_summary = torch.zeros(width * 2, dtype=torch.float32)

    atom_summary = torch.tensor(
        [
            n_atoms,
            carbon_count,
            hydrogen_count,
            hydrogen_count / max(carbon_count, 1.0),
        ],
        dtype=torch.float32,
    )
    return torch.cat([atom_summary, edge_summary]).to(torch.float32)


def train_property_model(
    csv_path: str | Path,
    smiles_column: str = "smiles",
    target_column: str = "target",
    checkpoint_path: str | Path | None = None,
    max_rows: int | None = None,
    epochs: int = 50,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
    baseline_mode: str = "none",
) -> PropertyTrainingResult:
    if baseline_mode not in ("none", "composition", "composition-ring"):
        raise ValueError(
            "baseline_mode must be 'none', 'composition', or 'composition-ring'"
        )
    require_e3nn()
    records = records_from_csv(
        csv_path,
        smiles_column=smiles_column,
        target_column=target_column,
        max_rows=max_rows,
        seed=seed,
    )
    if not records:
        raise ValueError("no property records loaded")

    torch.manual_seed(seed)
    features = torch.stack([molecular_e3nn_features(record.geometry) for record in records])
    targets = torch.tensor([record.target for record in records], dtype=torch.float32)

    permutation = torch.randperm(len(records))
    val_size = int(round(len(records) * val_fraction))
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    if train_indices.numel() == 0:
        train_indices = permutation
        val_indices = torch.empty(0, dtype=torch.long)

    baseline_features = torch.stack(
        [_baseline_features(record.geometry, baseline_mode) for record in records]
    )
    baseline_coefficients = _fit_baseline_coefficients(
        baseline_features[train_indices],
        targets[train_indices],
        baseline_mode=baseline_mode,
    )
    baseline_prediction = baseline_features @ baseline_coefficients
    residual_targets = targets - baseline_prediction
    target_mean = residual_targets[train_indices].mean()
    target_std = residual_targets[train_indices].std(unbiased=False)
    if float(target_std.cpu().item()) <= 1e-12:
        target_std = torch.tensor(1.0, dtype=torch.float32)
    scaled_targets = (residual_targets - target_mean) / target_std

    feature_dim = int(features.shape[1])
    model = E3NNInvariantRegressor(feature_dim=feature_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    last_train_loss = 0.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(features[train_indices])
        loss = loss_fn(prediction, scaled_targets[train_indices])
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.detach().cpu().item())

    val_loss: float | None = None
    train_mae: float = 0.0
    val_mae: float | None = None
    model.eval()
    with torch.no_grad():
        train_residual_prediction = _restore_target_scale(
            model(features[train_indices]),
            target_mean,
            target_std,
        )
        train_prediction = train_residual_prediction + baseline_prediction[train_indices]
        train_mae = float(
            torch.mean(torch.abs(train_prediction - targets[train_indices])).cpu().item()
        )
        train_baseline_mae = float(
            torch.mean(
                torch.abs(baseline_prediction[train_indices] - targets[train_indices])
            )
            .cpu()
            .item()
        )
    if val_indices.numel():
        with torch.no_grad():
            val_prediction = model(features[val_indices])
            val_loss = float(
                loss_fn(val_prediction, scaled_targets[val_indices]).cpu().item()
            )
            val_residual_prediction = _restore_target_scale(
                val_prediction,
                target_mean,
                target_std,
            )
            val_prediction_unscaled = (
                val_residual_prediction + baseline_prediction[val_indices]
            )
            val_mae = float(
                torch.mean(torch.abs(val_prediction_unscaled - targets[val_indices]))
                .cpu()
                .item()
            )
            val_baseline_mae = float(
                torch.mean(
                    torch.abs(baseline_prediction[val_indices] - targets[val_indices])
                )
                .cpu()
                .item()
            )
    else:
        val_baseline_mae = None

    saved_path: str | None = None
    if checkpoint_path is not None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "smiles_column": smiles_column,
            "target_column": target_column,
            "target_mean": float(target_mean.cpu().item()),
            "target_std": float(target_std.cpu().item()),
            "baseline_mode": baseline_mode,
            "baseline_coefficients": [
                float(value) for value in baseline_coefficients.cpu().tolist()
            ],
            "baseline_feature_names": _baseline_feature_names(baseline_mode),
        }
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        saved_path = str(checkpoint_path)

    return PropertyTrainingResult(
        train_loss=last_train_loss,
        val_loss=val_loss,
        epochs=epochs,
        checkpoint_path=saved_path,
        rows=len(records),
        train_mae=train_mae,
        val_mae=val_mae,
        train_baseline_mae=train_baseline_mae,
        val_baseline_mae=val_baseline_mae,
        target_mean=float(target_mean.cpu().item()),
        target_std=float(target_std.cpu().item()),
        baseline_mode=baseline_mode,
        baseline_coefficients=tuple(
            float(value) for value in baseline_coefficients.cpu().tolist()
        ),
    )


def predict_property(
    checkpoint_path: str | Path,
    geometry: GeometryRecord,
) -> float:
    checkpoint, model = _load_e3nn_property_model(checkpoint_path)
    features = molecular_e3nn_features(geometry).unsqueeze(0)
    with torch.no_grad():
        prediction = model(features)
        target_mean = float(checkpoint.get("target_mean", 0.0))
        target_std = float(checkpoint.get("target_std", 1.0))
        restored = prediction * target_std + target_mean
        restored = restored + _predict_baseline_from_checkpoint(checkpoint, geometry)
        return float(restored.cpu().item())


def _load_e3nn_property_model(
    checkpoint_path: str | Path,
) -> tuple[dict[str, object], E3NNInvariantRegressor]:
    """Load (and cache) the e3nn property-model checkpoint."""

    return _load_e3nn_property_model_cached(str(checkpoint_path))


@lru_cache(maxsize=8)
def _load_e3nn_property_model_cached(
    checkpoint_path: str,
) -> tuple[dict[str, object], E3NNInvariantRegressor]:
    """Load (and cache) the e3nn property-model checkpoint.

    The model is used to predict a property for many molecules against the same
    checkpoint, so caching the loaded model / checkpoint per path avoids repeated
    disk reads and model construction. Both are only read during inference.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = E3NNInvariantRegressor(
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def _restore_target_scale(
    value: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    return value * target_std + target_mean


def _fit_baseline_coefficients(
    features: torch.Tensor,
    targets: torch.Tensor,
    baseline_mode: str,
) -> torch.Tensor:
    if baseline_mode == "none":
        return torch.zeros(features.shape[1], dtype=torch.float32)
    solution = torch.linalg.lstsq(features, targets.unsqueeze(1)).solution.squeeze(1)
    return solution.to(torch.float32)


def _baseline_features(geometry: GeometryRecord, baseline_mode: str) -> torch.Tensor:
    if baseline_mode == "composition-ring":
        return _composition_ring_baseline_features(geometry)
    return _composition_baseline_features(geometry)


def _baseline_feature_names(baseline_mode: str) -> tuple[str, ...]:
    if baseline_mode == "composition-ring":
        return (
            "carbon_count",
            "hydrogen_count",
            "ring_count",
            "three_member_rings",
            "four_member_rings",
            "five_member_rings",
            "six_member_rings",
            "large_rings",
            "bridgehead_atoms",
            "spiro_atoms",
            "multi_ring_atoms",
            "intercept",
        )
    return ("carbon_count", "hydrogen_count", "intercept")


def _composition_baseline_features(geometry: GeometryRecord) -> torch.Tensor:
    atomic_numbers, _ = geometry_to_tensors(geometry)
    carbon_count = float((atomic_numbers == 6).sum().item())
    hydrogen_count = float((atomic_numbers == 1).sum().item())
    return torch.tensor([carbon_count, hydrogen_count, 1.0], dtype=torch.float32)


def _composition_ring_baseline_features(geometry: GeometryRecord) -> torch.Tensor:
    base = _composition_baseline_features(geometry)
    ring_features = _ring_baseline_values(geometry.canonical_smiles)
    return torch.tensor(
        [
            float(base[0].item()),
            float(base[1].item()),
            *ring_features,
            1.0,
        ],
        dtype=torch.float32,
    )


def _ring_baseline_values(smiles: str) -> tuple[float, ...]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required for ring-aware baseline features.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES for ring baseline: {smiles}")
    ring_info = mol.GetRingInfo()
    atom_rings = tuple(ring_info.AtomRings())
    ring_sizes = [len(ring) for ring in atom_rings]
    ring_count = float(len(ring_sizes))
    size_counts = {
        size: float(sum(1 for ring_size in ring_sizes if ring_size == size))
        for size in range(3, 7)
    }
    large_rings = float(sum(1 for ring_size in ring_sizes if ring_size >= 7))
    atom_memberships: dict[int, int] = {}
    for ring in atom_rings:
        for atom_index in ring:
            atom_memberships[atom_index] = atom_memberships.get(atom_index, 0) + 1
    multi_ring_atoms = float(sum(1 for count in atom_memberships.values() if count > 1))
    bridgehead_atoms = float(_safe_rdkit_count(rdMolDescriptors.CalcNumBridgeheadAtoms, mol))
    spiro_atoms = float(_safe_rdkit_count(rdMolDescriptors.CalcNumSpiroAtoms, mol))
    return (
        ring_count,
        size_counts[3],
        size_counts[4],
        size_counts[5],
        size_counts[6],
        large_rings,
        bridgehead_atoms,
        spiro_atoms,
        multi_ring_atoms,
    )


def _safe_rdkit_count(function: object, mol: object) -> int:
    try:
        return int(function(mol))
    except Exception:
        return 0


def _predict_baseline_from_checkpoint(
    checkpoint: dict[str, object],
    geometry: GeometryRecord,
) -> torch.Tensor:
    coefficients = checkpoint.get("baseline_coefficients")
    baseline_mode = str(checkpoint.get("baseline_mode", "none"))
    if baseline_mode == "none" or coefficients is None:
        return torch.tensor(0.0, dtype=torch.float32)
    coeff_tensor = torch.tensor(coefficients, dtype=torch.float32)
    return _baseline_features(geometry, baseline_mode) @ coeff_tensor


def _radial_features(distance: torch.Tensor, bins: int, cutoff: float) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, bins, dtype=torch.float32).reshape(1, -1)
    width = cutoff / max(bins - 1, 1)
    distances = distance.reshape(-1, 1)
    return torch.exp(-((distances - centers) ** 2) / max(width**2, 1e-6))
