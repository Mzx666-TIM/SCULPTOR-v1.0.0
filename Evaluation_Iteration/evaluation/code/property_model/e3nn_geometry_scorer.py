from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from geometry_analysis import analyze_mmff_geometry
from property_model.data import GeometryRecord, geometry_from_smiles
from property_model.e3nn_regressor import (
    E3NNInvariantRegressor,
    molecular_e3nn_features,
    require_e3nn,
)

# ``rdMolAlign.GetBestRMS`` enumerates every symmetry-equivalent atom mapping of
# the molecule onto itself, which dominates the runtime of the reproducibility
# term for our highly symmetric cage hydrocarbons. RDKit can spread that search
# over threads; the returned minimum is unchanged (only the reduction order of
# the search changes), so this is a pure speed-up.
_ALIGN_THREADS: int = max(1, min(8, (os.cpu_count() or 1) - 2))


def set_align_threads(threads: int) -> None:
    """Limit the thread count used inside ``GetBestRMS`` for this process."""

    global _ALIGN_THREADS
    _ALIGN_THREADS = max(1, int(threads))


# The reproducibility term (two extra embeddings + force-field optimisation +
# symmetry-aware RMSD) dominates the cost of the programmatic geometry score
# (~460 ms per molecule). It is a confirmatory diagnostic for the candidates
# that enter the compute queue, so it can be switched off for the rest of the
# batch. The composite score is unaffected: production uses the e3nn geometry
# *model*, whose input is the 30-dimensional 3D feature vector only.
_REPRODUCIBILITY_ENABLED: bool = True
_REPRODUCIBILITY_NEUTRAL: float = 0.5


def set_reproducibility_enabled(enabled: bool) -> None:
    global _REPRODUCIBILITY_ENABLED
    _REPRODUCIBILITY_ENABLED = bool(enabled)


def reproducibility_enabled() -> bool:
    return _REPRODUCIBILITY_ENABLED


@dataclass(frozen=True)
class GeometryScore:
    smiles: str
    canonical_smiles: str
    e3nn_geometry_score: float
    symmetry_consistency: float
    mmff_stability: float
    compactness: float
    strain_reasonableness: float
    conformer_reproducibility: float
    mmff_energy_per_heavy_atom: float | None
    mmff_converged: bool
    label_source: str = "programmatic_geometry_label_v1"

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["geometry_label_components_json"] = json.dumps(
            {
                "symmetry_consistency": self.symmetry_consistency,
                "mmff_stability": self.mmff_stability,
                "compactness": self.compactness,
                "strain_reasonableness": self.strain_reasonableness,
                "conformer_reproducibility": self.conformer_reproducibility,
            },
            ensure_ascii=False,
        )
        return payload


@dataclass(frozen=True)
class GeometryScorerTrainingResult:
    train_loss: float
    val_loss: float | None
    epochs: int
    checkpoint_path: str | None
    rows: int
    skipped_rows: int
    target_column: str


def geometry_score_from_smiles(
    smiles: str,
    seed: int = 0,
    geometry: GeometryRecord | None = None,
) -> GeometryScore:
    geometry = geometry if geometry is not None else geometry_from_smiles(smiles, seed=seed)
    mmff = analyze_mmff_geometry(smiles, seed=seed)
    symmetry = _inertia_symmetry_score(geometry)
    compactness = _compactness_score(geometry)
    reproducibility = (
        _conformer_reproducibility_score(smiles, seed=seed)
        if _REPRODUCIBILITY_ENABLED
        else _REPRODUCIBILITY_NEUTRAL
    )
    # ``strain_reasonableness`` is retained in historical CSV schemas as an
    # alias of MMFF stability. Combine its old 0.10 share here so the score is
    # numerically unchanged without presenting one signal as two independent
    # supervision targets.
    strain = mmff.mmff_stability_score
    score = (
        0.35 * symmetry
        + 0.35 * mmff.mmff_stability_score
        + 0.20 * compactness
        + 0.10 * reproducibility
    )
    return GeometryScore(
        smiles=smiles,
        canonical_smiles=geometry.canonical_smiles,
        e3nn_geometry_score=round(float(_clamp01(score)), 6),
        symmetry_consistency=round(float(symmetry), 6),
        mmff_stability=round(float(mmff.mmff_stability_score), 6),
        compactness=round(float(compactness), 6),
        strain_reasonableness=round(float(strain), 6),
        conformer_reproducibility=round(float(reproducibility), 6),
        mmff_energy_per_heavy_atom=mmff.mmff_energy_per_heavy_atom,
        mmff_converged=mmff.mmff_converged,
    )


def write_geometry_scores_csv(
    input_path: str | Path,
    output_path: str | Path,
    smiles_column: str = "canonical_smiles",
    seed: int = 0,
) -> int:
    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    rows = []
    for index, record in enumerate(records):
        smiles = str(record.get(smiles_column, "")).strip()
        if not smiles:
            continue
        try:
            row = geometry_score_from_smiles(smiles, seed=seed + index).as_dict()
            row["geometry_status"] = "ok"
            row["geometry_failure_reason"] = ""
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "smiles": smiles,
                    "canonical_smiles": smiles,
                    "e3nn_geometry_score": "",
                    "symmetry_consistency": "",
                    "mmff_stability": "",
                    "compactness": "",
                    "strain_reasonableness": "",
                    "conformer_reproducibility": "",
                    "mmff_energy_per_heavy_atom": "",
                    "mmff_converged": "",
                    "label_source": "geometry_label_failed",
                    "geometry_label_components_json": "{}",
                    "geometry_status": "failed",
                    "geometry_failure_reason": repr(exc),
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return 0
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def train_geometry_scorer(
    csv_path: str | Path,
    smiles_column: str = "canonical_smiles",
    target_column: str = "e3nn_geometry_score",
    checkpoint_path: str | Path | None = None,
    max_rows: int | None = None,
    epochs: int = 50,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> GeometryScorerTrainingResult:
    require_e3nn()
    geometries: list[GeometryRecord] = []
    targets: list[float] = []
    skipped_rows = 0
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if max_rows is not None and len(geometries) >= max_rows:
                break
            smiles = str(row.get(smiles_column, "")).strip()
            if not smiles:
                continue
            target_text = str(row.get(target_column, "")).strip()
            try:
                target = (
                    float(target_text)
                    if target_text
                    else geometry_score_from_smiles(smiles, seed=seed + len(geometries)).e3nn_geometry_score
                )
                geometries.append(geometry_from_smiles(smiles, seed=seed + len(geometries)))
                targets.append(target)
            except Exception:
                skipped_rows += 1
                continue
    if not geometries:
        raise ValueError("no geometry scoring records loaded")

    torch.manual_seed(seed)
    features = torch.stack([molecular_e3nn_features(geometry) for geometry in geometries])
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    model = E3NNInvariantRegressor(
        feature_dim=int(features.shape[1]),
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    permutation = torch.randperm(len(geometries))
    val_size = int(round(len(geometries) * val_fraction))
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    if train_indices.numel() == 0:
        train_indices = permutation
        val_indices = torch.empty(0, dtype=torch.long)

    train_loss = 0.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(features[train_indices])
        loss = loss_fn(prediction, target_tensor[train_indices])
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu().item())

    val_loss: float | None = None
    if val_indices.numel():
        model.eval()
        with torch.no_grad():
            val_prediction = model(features[val_indices])
            val_loss = float(loss_fn(val_prediction, target_tensor[val_indices]).cpu().item())

    saved_path: str | None = None
    if checkpoint_path is not None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "feature_dim": int(features.shape[1]),
            "hidden_dim": hidden_dim,
            "smiles_column": smiles_column,
            "target_column": target_column,
            "model_role": "e3nn_geometry_scorer",
        }
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        saved_path = str(checkpoint_path)

    return GeometryScorerTrainingResult(
        train_loss=train_loss,
        val_loss=val_loss,
        epochs=epochs,
        checkpoint_path=saved_path,
        rows=len(geometries),
        skipped_rows=skipped_rows,
        target_column=target_column,
    )


def _load_geometry_model(
    checkpoint_path: str | Path,
) -> tuple[E3NNInvariantRegressor, dict[str, object]]:
    """Load (and cache) the e3nn geometry scorer checkpoint."""

    return _load_geometry_model_cached(str(checkpoint_path))


@lru_cache(maxsize=8)
def _load_geometry_model_cached(
    checkpoint_path: str,
) -> tuple[E3NNInvariantRegressor, dict[str, object]]:
    """Load (and cache) the e3nn geometry scorer checkpoint.

    The scorer is used for every molecule in a run with the same checkpoint, so
    caching the loaded model / checkpoint per path avoids repeated disk reads and
    model construction. The model is only ever used for inference and the
    checkpoint dict is read-only, so sharing the cached objects is safe.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = E3NNInvariantRegressor(
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_geometry_score(
    checkpoint_path: str | Path,
    smiles: str,
    seed: int = 0,
) -> dict[str, object]:
    model, checkpoint = _load_geometry_model(checkpoint_path)
    geometry = geometry_from_smiles(smiles, seed=seed)
    features = molecular_e3nn_features(geometry).unsqueeze(0)
    with torch.no_grad():
        prediction = float(model(features).cpu().item())
    programmatic = geometry_score_from_smiles(smiles, seed=seed, geometry=geometry)
    payload = programmatic.as_dict()
    payload["e3nn_geometry_model_score"] = round(float(_clamp01(prediction)), 6)
    payload["e3nn_geometry_model_role"] = checkpoint.get(
        "model_role",
        "e3nn_geometry_scorer",
    )
    return payload


def predict_geometry_score_batch(
    checkpoint_path: str | Path,
    smiles_list: list[str],
    seed: int = 0,
    seeds: list[int] | None = None,
) -> dict[str, tuple[GeometryScore, float | None]]:
    """Score many molecules with one checkpoint load and one batched forward pass.

    Each molecule's 3D geometry is generated once and reused for both the e3nn
    invariant features and the programmatic geometry score. Returns a mapping
    from input SMILES to ``(GeometryScore, model_score_or_None)``; molecules
    whose geometry cannot be embedded are omitted.
    """
    require_e3nn()
    model, _checkpoint = _load_geometry_model(checkpoint_path)
    results: dict[str, tuple[GeometryScore, float | None]] = {}
    pending: list[tuple[str, int, GeometryRecord]] = []
    for index, smiles in enumerate(smiles_list):
        molecule_seed = seeds[index] if seeds is not None else seed + index
        try:
            geometry = geometry_from_smiles(smiles, seed=molecule_seed)
        except Exception:
            continue
        pending.append((smiles, molecule_seed, geometry))
    if pending:
        features = torch.stack([molecular_e3nn_features(g) for _, _, g in pending])
        with torch.no_grad():
            predictions = model(features)
        for (smiles, molecule_seed, geometry), prediction in zip(pending, predictions):
            programmatic = geometry_score_from_smiles(
                smiles,
                seed=molecule_seed,
                geometry=geometry,
            )
            results[smiles] = (
                programmatic,
                round(float(_clamp01(prediction.item())), 6),
            )
    return results


def _inertia_symmetry_score(geometry: GeometryRecord) -> float:
    _, positions = _geometry_tensors(geometry)
    centered = positions - positions.mean(dim=0, keepdim=True)
    inertia = centered.T @ centered
    eigenvalues = torch.linalg.eigvalsh(inertia)
    mean = float(eigenvalues.mean().item())
    if mean <= 1e-9:
        return 0.0
    spread = float(eigenvalues.std(unbiased=False).item()) / mean
    return _clamp01(1.0 / (1.0 + spread))


def _compactness_score(geometry: GeometryRecord) -> float:
    atomic_numbers, positions = _geometry_tensors(geometry)
    heavy = positions[atomic_numbers > 1]
    if heavy.numel() == 0:
        return 0.0
    centered = heavy - heavy.mean(dim=0, keepdim=True)
    radius = float(torch.sqrt((centered * centered).sum(dim=1).mean()).item())
    target = max(float(heavy.shape[0]) ** (1.0 / 3.0), 1.0)
    return _clamp01(target / max(radius, 1e-9))


@lru_cache(maxsize=8192)
def _conformer_reproducibility_score(smiles: str, seed: int) -> float:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolAlign

        mol_a = Chem.AddHs(Chem.MolFromSmiles(smiles))
        mol_b = Chem.AddHs(Chem.MolFromSmiles(smiles))
        for mol, offset in ((mol_a, 0), (mol_b, 7919)):
            params = AllChem.ETKDGv3()
            params.randomSeed = int(seed) + offset
            params.useRandomCoords = offset > 0
            status = AllChem.EmbedMolecule(mol, params)
            if status != 0:
                return 0.0
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
            except Exception:
                pass
        rmsd = float(rdMolAlign.GetBestRMS(mol_a, mol_b, numThreads=_ALIGN_THREADS))
        return _clamp01(1.0 / (1.0 + rmsd))
    except Exception:
        return 0.0


def _geometry_tensors(geometry: GeometryRecord) -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor(geometry.atomic_numbers, dtype=torch.long)
    positions = torch.tensor(geometry.positions, dtype=torch.float32)
    return atomic_numbers, positions


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))
