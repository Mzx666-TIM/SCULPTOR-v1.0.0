from __future__ import annotations

import sys
import csv
import math
import random
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "PyTorch is required for the STA property model. Install torch in the "
        "project environment before training or predicting with STA."
    ) from exc

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - rdkit is part of the normal project env
    Chem = None

from fuel_design.paths import EVAL_DATA_DIR, STA_820K_MODEL_DIR, STA_MODEL_DIR

try:
    from device_policy import resolve_device
except ImportError:  # pragma: no cover - device_policy is part of core/code
    resolve_device = None

DEFAULT_STA_DATA_PATH = EVAL_DATA_DIR / "gdb_sta_multi_property.csv"
DEFAULT_STA_GDB13_DATA_PATH = EVAL_DATA_DIR / "gdb13_sta_multi_property.csv"
DEFAULT_STA_MODEL_PATH = STA_820K_MODEL_DIR / "sta_gdb_density_paper.pt"


def torch_load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint across torch versions.

    torch >= 2.6 defaults ``weights_only=True``, which cannot restore the
    pickled dictionaries used here; torch < 1.13 does not accept the keyword
    at all.  Centralising the call keeps the checkpoints loadable on both.
    """
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only
        return torch.load(Path(path), map_location=map_location)

TARGET_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "flash_point_k": ("flash_point/K", "flash_point_k", "flash_point"),
    "density_g_cm3": ("density/(g/cm3)", "density_g_cm3", "density"),
    "formation_enthalpy_kj_mol": (
        "delta_Hf/(KJ/mol)",
        "formation_enthalpy_kj_mol",
        "delta_hf",
    ),
    "molar_volume_cm3_mol": (
        "molar_volume/(cm3/mol)(default298K)",
        "molar_volume_cm3_mol",
        "molar_volume",
    ),
    "isp": ("ISP", "isp"),
    "melting_point_k": ("Tm/K", "melting_point_k", "tm_k"),
    "boiling_point_k": ("Tb/K", "boiling_point_k", "tb_k"),
    "combustion_enthalpy_kj_mol": (
        "delta_Hc/(KJ/mol)",
        "combustion_enthalpy_kj_mol",
        "delta_hc",
    ),
    "mass_calorific_value_mj_kg": (
        "mass_calorific_value_h/(MJ/kg)",
        "mass_calorific_value_mj_kg",
        "heating_value_mj_kg",
    ),
}

# The GDB source header says cm3/mol, but its stored values are in L/mol
# (for example decane is 0.196 L/mol = 196 cm3/mol). Keep the raw training
# target name for checkpoint compatibility and convert at the public boundary.
STA_PUBLIC_OUTPUT_SCALES: dict[str, float] = {
    "molar_volume_cm3_mol": 1000.0,
}
STA_SOURCE_UNITS: dict[str, str] = {
    "molar_volume_cm3_mol": "L/mol",
}
STA_PUBLIC_UNITS: dict[str, str] = {
    "molar_volume_cm3_mol": "cm3/mol",
}

DEFAULT_TARGETS = (
    "flash_point_k",
    "density_g_cm3",
    "formation_enthalpy_kj_mol",
    "molar_volume_cm3_mol",
    "isp",
    "melting_point_k",
    "boiling_point_k",
)
DEFAULT_SPECIALIST_GROUPS: dict[str, tuple[str, ...]] = {
    "density_molar_volume": ("density_g_cm3", "molar_volume_cm3_mol"),
    "formation_enthalpy": ("formation_enthalpy_kj_mol",),
    "phase_points": ("melting_point_k", "boiling_point_k"),
    "flash_point": ("flash_point_k",),
    "isp": ("isp",),
}
DEFAULT_SPECIALIST_MODEL_PATHS: dict[str, Path] = {
    target: STA_MODEL_DIR / f"sta_gdb_{group}.pt"
    for group, targets in DEFAULT_SPECIALIST_GROUPS.items()
    for target in targets
}
SPECIAL_TOKENS = (
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "Si",
    "Cl",
    "Ca",
    "Sc",
    "Ti",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Rb",
    "Sr",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "He",
    "Ne",
    "Ar",
    "Kr",
    "Xe",
)
@dataclass(frozen=True)
class StaTrainingSummary:
    input_paths: tuple[str, ...]
    output_path: str
    rows_read: int
    rows_written: int
    target_columns: tuple[str, ...]

@dataclass(frozen=True)
class StaModelMetrics:
    model_path: str
    train_rows: int
    validation_rows: int
    target_columns: tuple[str, ...]
    validation_mae: dict[str, float]
    validation_rmse: dict[str, float]
    validation_r2: dict[str, float]

@dataclass(frozen=True)
class StaSpecialistTrainingSummary:
    data_path: str
    output_dir: str
    models: dict[str, StaModelMetrics]

@dataclass(frozen=True)
class StaTargetTrainingConfig:
    target: str
    output_name: str
    d_model: int
    num_heads: int
    num_layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    dropout: float = 0.08
    patience: int = 8
    min_carbons: int | None = None
    max_carbons: int | None = None
    target_min: float | None = None
    target_max: float | None = None
    enumerations_per_molecule: int = 1

@dataclass(frozen=True)
class StaOptimizedTrainingSummary:
    data_path: str
    output_dir: str
    models: dict[str, StaModelMetrics]

DEFAULT_PAPER_TARGET_CONFIGS: tuple[StaTargetTrainingConfig, ...] = (
    StaTargetTrainingConfig(
        target="density_g_cm3",
        output_name="sta_gdb_density_paper.pt",
        d_model=512,
        num_heads=8,
        num_layers=12,
        epochs=100,
        batch_size=10,
        learning_rate=1e-5,
        patience=5,
    ),
    StaTargetTrainingConfig(
        target="molar_volume_cm3_mol",
        output_name="sta_gdb_molar_volume_paper.pt",
        d_model=512,
        num_heads=8,
        num_layers=12,
        epochs=100,
        batch_size=10,
        learning_rate=1e-5,
        patience=5,
    ),
    StaTargetTrainingConfig(
        target="formation_enthalpy_kj_mol",
        output_name="sta_gdb_formation_enthalpy_paper.pt",
        d_model=512,
        num_heads=8,
        num_layers=12,
        epochs=100,
        batch_size=10,
        learning_rate=1e-5,
        patience=5,
    ),
    StaTargetTrainingConfig(
        target="melting_point_k",
        output_name="sta_gdb_melting_point_paper.pt",
        d_model=512,
        num_heads=8,
        num_layers=12,
        epochs=100,
        batch_size=10,
        learning_rate=1e-5,
        patience=5,
    ),
    StaTargetTrainingConfig(
        target="boiling_point_k",
        output_name="sta_gdb_boiling_point_paper.pt",
        d_model=512,
        num_heads=8,
        num_layers=12,
        epochs=100,
        batch_size=10,
        learning_rate=1e-5,
        patience=5,
    ),
    StaTargetTrainingConfig(
        target="flash_point_k",
        output_name="sta_gdb_flash_point_paper.pt",
        d_model=512,
        num_heads=8,
        num_layers=12,
        epochs=100,
        batch_size=10,
        learning_rate=1e-5,
        patience=5,
    ),
    StaTargetTrainingConfig(
        target="isp",
        output_name="sta_gdb_isp_paper.pt",
        d_model=512,
        num_heads=8,
        num_layers=12,
        epochs=100,
        batch_size=10,
        learning_rate=1e-5,
        patience=5,
    ),
)

DEFAULT_PAPER_MODEL_PATHS: dict[str, Path] = {
    config.target: STA_820K_MODEL_DIR / config.output_name
    for config in DEFAULT_PAPER_TARGET_CONFIGS
}
    # Note: molar_volume / ISP are not part of the 820k full training
    # (molar volume is derived from density; ISP is no longer predicted).
    # Their paths are placeholders and no longer fall back to old root models.
DEFAULT_OPTIMIZED_TARGET_CONFIGS = DEFAULT_PAPER_TARGET_CONFIGS
DEFAULT_OPTIMIZED_MODEL_PATHS = DEFAULT_PAPER_MODEL_PATHS
DEFAULT_SPECIALIST_MODEL_PATHS.update(DEFAULT_PAPER_MODEL_PATHS)

def prepare_sta_gdb_dataset(
    input_paths: Iterable[str | Path],
    output_path: str | Path = DEFAULT_STA_DATA_PATH,
    target_columns: Iterable[str] = DEFAULT_TARGETS,
) -> StaTrainingSummary:
    targets = tuple(target_columns)
    output = Path(output_path)
    rows_out: list[dict[str, object]] = []
    rows_read = 0
    seen: set[str] = set()
    for input_path in input_paths:
        with Path(input_path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            column_map = _target_column_map(fieldnames, targets)
            for row in reader:
                rows_read += 1
                smiles = str(row.get("smiles") or row.get("SMILES") or "").strip()
                canonical = canonicalize_smiles(smiles)
                if not canonical or canonical in seen:
                    continue
                values = {
                    target: _optional_float(row.get(column_map[target]))
                    for target in targets
                    if target in column_map
                }
                if not any(value is not None for value in values.values()):
                    continue
                out = {"canonical_smiles": canonical}
                out.update(
                    {
                        target: "" if values.get(target) is None else values[target]
                        for target in targets
                    }
                )
                rows_out.append(out)
                seen.add(canonical)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["canonical_smiles", *targets])
        writer.writeheader()
        writer.writerows(rows_out)
    return StaTrainingSummary(
        input_paths=tuple(str(Path(path)) for path in input_paths),
        output_path=str(output),
        rows_read=rows_read,
        rows_written=len(rows_out),
        target_columns=targets,
    )

def train_sta_property_model(
    data_path: str | Path = DEFAULT_STA_DATA_PATH,
    model_output_path: str | Path = DEFAULT_STA_MODEL_PATH,
    target_columns: Iterable[str] | None = None,
    feature_csv: str | Path | None = None,
    epochs: int = 100,
    batch_size: int = 10,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 12,
    learning_rate: float = 1e-5,
    dropout: float = 0.0,
    pooling: str = "additive",
    loss: str = "mae",
    patience: int = 5,
    enumerations_per_molecule: int = 1,
    min_carbons: int | None = None,
    max_carbons: int | None = None,
    target_ranges: dict[str, tuple[float | None, float | None]] | None = None,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> StaModelMetrics:
    records = _read_sta_records(
        data_path,
        target_columns,
        min_carbons=min_carbons,
        max_carbons=max_carbons,
        target_ranges=target_ranges,
    )
    if len(records) < 10:
        raise ValueError("STA training needs at least 10 molecules")
    feature_map: dict[str, np.ndarray] = {}
    if feature_csv is not None:
        import csv as _csv
        with Path(feature_csv).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = _csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            smiles_col = (
                "canonical_smiles" if "canonical_smiles" in fieldnames
                else ("smiles" if "smiles" in fieldnames else None)
            )
            if smiles_col is None:
                raise ValueError("feature_csv must contain a canonical_smiles column")
            feature_names_all = [
                c for c in fieldnames
                if c != smiles_col and c.startswith("e3nn_3d_feature_")
            ]
            for row in reader:
                smi = str(row.get(smiles_col) or "").strip()
                values = [
                    float(row[c]) for c in feature_names_all
                    if str(row.get(c) or "").strip() not in ("", "nan")
                ]
                if not smi or len(values) != len(feature_names_all):
                    continue
                feature_map[smi] = np.asarray(values, dtype=np.float32)
        if not feature_map:
            raise ValueError("feature_csv provided no usable e3nn features")
        before = len(records)
        records = [
            r for r in records if r.get("canonical_smiles") in feature_map
        ]
        if len(records) < 10:
            raise ValueError("too few records overlap with feature_csv")
        print(
            f"STA feature filter: {before} -> {len(records)} records with e3nn features",
            flush=True,
        )
    rng = random.Random(seed)
    rng.shuffle(records)
    targets = tuple(key for key in records[0] if key != "canonical_smiles")
    y, mask = _target_arrays(records, targets)
    means = np.zeros((len(targets),), dtype=np.float32)
    stds = np.ones((len(targets),), dtype=np.float32)
    split = max(1, int(round(len(records) * (1.0 - validation_fraction))))
    if split >= len(records):
        split = len(records) - 1
    train_sources, train_smiles = _expanded_smiles_for_records(
        records[:split],
        enumerations_per_molecule,
        seed=seed,
    )
    val_sources, val_smiles = _expanded_smiles_for_records(
        records[split:],
        enumerations_per_molecule,
        seed=seed + 1009,
    )
    vocab = _build_vocab([*train_smiles, *val_smiles])
    max_length = max(len(smiles) for smiles in [*train_smiles, *val_smiles])
    train_token_arrays = np.array(
        [_smiles_to_indices(smiles, vocab, max_length) for smiles in train_smiles],
        dtype=np.int64,
    )
    val_token_arrays = np.array(
        [_smiles_to_indices(smiles, vocab, max_length) for smiles in val_smiles],
        dtype=np.int64,
    )
    if feature_map:
        feature_names = tuple(
            c for c in feature_names_all
        )
        train_raw = np.stack(
            [feature_map[canonicalize_smiles(s)] for s in train_smiles],
            axis=0,
        ).astype(np.float32)
        val_raw = np.stack(
            [feature_map[canonicalize_smiles(s)] for s in val_smiles],
            axis=0,
        ).astype(np.float32)
        feature_means = train_raw.mean(axis=0).astype(np.float32)
        feature_stds = train_raw.std(axis=0).astype(np.float32)
        feature_stds[feature_stds < 1e-8] = 1.0
        train_features = (train_raw - feature_means) / feature_stds
        val_features = (val_raw - feature_means) / feature_stds
    else:
        feature_names = ()
        feature_means = np.zeros((0,), dtype=np.float32)
        feature_stds = np.ones((0,), dtype=np.float32)
        train_features = np.zeros((len(train_token_arrays), 0), dtype=np.float32)
        val_features = np.zeros((len(val_token_arrays), 0), dtype=np.float32)
    train_dataset = _StaDataset(
        train_token_arrays,
        train_features,
        y[:split][train_sources],
        mask[:split][train_sources],
    )
    val_dataset = _StaDataset(
        val_token_arrays,
        val_features,
        y[split:][val_sources],
        mask[split:][val_sources],
    )
    device = resolve_device() if resolve_device else torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = _StaTransformer(
        vocab_size=len(vocab),
        length=max_length,
        output_dim=len(targets),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        pooling=pooling,
        feature_dim=len(feature_names),
    ).to(device)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, patience // 3),
    )
    print(
        (
            f"STA training rows={len(train_dataset)} validation_rows={len(val_dataset)} "
            f"targets={len(targets)} device={device} batch_size={batch_size}"
        ),
        flush=True,
    )
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for inputs, features_batch, target, target_mask in loader:
            inputs = inputs.to(device, non_blocking=True)
            features_batch = features_batch.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_mask = target_mask.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(inputs, features_batch)
                loss_value = _masked_loss(prediction, target, target_mask, loss)
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss_value.detach().cpu())
            batches += 1
        val_loss = _evaluate_loss(model, val_dataset, batch_size, device, loss=loss)
        scheduler.step(val_loss)
        print(
            (
                f"STA epoch {epoch + 1}/{epochs}: "
                f"train_loss={running_loss / max(batches, 1):.6f} "
                f"val_loss={val_loss:.6f}"
            ),
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = _evaluate_metrics(model, val_dataset, targets, means, stds, batch_size, device)
    artifact = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "max_length": max_length,
        "target_columns": targets,
        "target_means": means.tolist(),
        "target_stds": stds.tolist(),
        "feature_names": list(feature_names),
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "metrics": asdict(metrics),
        "target_source_units": {
            target: STA_SOURCE_UNITS.get(target, "")
            for target in targets
        },
        "target_output_units": {
            target: STA_PUBLIC_UNITS.get(target, "")
            for target in targets
        },
        "model_config": {
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "pooling": pooling,
            "loss": loss,
        },
        "training_filters": {
            "min_carbons": min_carbons,
            "max_carbons": max_carbons,
            "target_ranges": target_ranges or {},
        },
    }
    output = Path(model_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    return StaModelMetrics(
        model_path=str(output),
        train_rows=len(train_dataset),
        validation_rows=len(val_dataset),
        target_columns=targets,
        validation_mae=_scale_sta_metric_values(metrics.validation_mae),
        validation_rmse=_scale_sta_metric_values(metrics.validation_rmse),
        validation_r2=metrics.validation_r2,
    )

def train_sta_specialist_property_models(
    data_path: str | Path = DEFAULT_STA_DATA_PATH,
    output_dir: str | Path = STA_MODEL_DIR,
    groups: dict[str, tuple[str, ...]] = DEFAULT_SPECIALIST_GROUPS,
    epochs: int = 30,
    batch_size: int = 4096,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 3,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> StaSpecialistTrainingSummary:
    output_root = Path(output_dir)
    metrics_by_group: dict[str, StaModelMetrics] = {}
    for offset, (group_name, targets) in enumerate(groups.items()):
        model_path = output_root / f"sta_gdb_{group_name}.pt"
        print(
            f"===== STA specialist group {group_name}: {', '.join(targets)} =====",
            flush=True,
        )
        metrics_by_group[group_name] = train_sta_property_model(
            data_path=data_path,
            model_output_path=model_path,
            target_columns=targets,
            epochs=epochs,
            batch_size=batch_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            seed=seed + offset,
        )
    return StaSpecialistTrainingSummary(
        data_path=str(Path(data_path)),
        output_dir=str(output_root),
        models=metrics_by_group,
    )

def train_sta_optimized_property_models(
    data_path: str | Path = DEFAULT_STA_DATA_PATH,
    output_dir: str | Path = STA_MODEL_DIR,
    configs: Iterable[StaTargetTrainingConfig] = DEFAULT_OPTIMIZED_TARGET_CONFIGS,
    validation_fraction: float = 0.1,
    seed: int = 42,
    batch_size: int | None = None,
    epochs: int | None = None,
) -> StaOptimizedTrainingSummary:
    output_root = Path(output_dir)
    metrics_by_target: dict[str, StaModelMetrics] = {}
    effective_configs = tuple(
        replace(
            config,
            batch_size=batch_size if batch_size is not None else config.batch_size,
            epochs=epochs if epochs is not None else config.epochs,
        )
        for config in configs
    )
    for offset, config in enumerate(effective_configs):
        model_path = output_root / config.output_name
        target_ranges = None
        if config.target_min is not None or config.target_max is not None:
            target_ranges = {config.target: (config.target_min, config.target_max)}
        print(
            (
                f"===== STA paper target {config.target}: "
                f"d_model={config.d_model} heads={config.num_heads} "
                f"layers={config.num_layers} pooling=additive "
                f"loss=mae enumerations={config.enumerations_per_molecule} ====="
            ),
            flush=True,
        )
        metrics_by_target[config.target] = train_sta_property_model(
            data_path=data_path,
            model_output_path=model_path,
            target_columns=(config.target,),
            epochs=config.epochs,
            batch_size=config.batch_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            learning_rate=config.learning_rate,
            dropout=config.dropout,
            pooling="additive",
            loss="mae",
            patience=config.patience,
            enumerations_per_molecule=config.enumerations_per_molecule,
            min_carbons=config.min_carbons,
            max_carbons=config.max_carbons,
            target_ranges=target_ranges,
            validation_fraction=validation_fraction,
            seed=seed + offset,
        )
    return StaOptimizedTrainingSummary(
        data_path=str(Path(data_path)),
        output_dir=str(output_root),
        models=metrics_by_target,
    )


def _e3nn_features_vector(smiles: str) -> dict[str, float]:
    """Deterministic 30-dim e3nn 3D features, same recipe as the training tables."""
    import hashlib

    from property_model.data import geometry_from_smiles
    from property_model.e3nn_regressor import molecular_e3nn_features

    canonical = canonicalize_smiles(smiles)
    if not canonical:
        raise ValueError("invalid SMILES for STA e3nn features")
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little") & 0x7FFFFFFF
    geometry = geometry_from_smiles(canonical, seed=seed)
    values = molecular_e3nn_features(geometry).detach().cpu().numpy().astype(float)
    return {
        "e3nn_3d_feature_%02d" % index: float(value)
        for index, value in enumerate(values)
    }


def _predict_feature_tensor(
    smiles: str,
    feature_names: tuple[str, ...],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if not feature_names:
        return torch.zeros((1, 0), dtype=torch.float32, device=device)
    feats = _e3nn_features_vector(smiles)
    row = np.asarray(
        [feats.get(name, 0.0) for name in feature_names], dtype=np.float32
    )
    row = (row - feature_means) / feature_stds
    return torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)


def _predict_features_tensor_batch(
    canonicals: list[str],
    feature_names: tuple[str, ...],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if not feature_names:
        return torch.zeros((len(canonicals), 0), dtype=torch.float32, device=device)
    rows = []
    for canonical in canonicals:
        feats = _e3nn_features_vector(canonical)
        row = np.asarray(
            [feats.get(name, 0.0) for name in feature_names], dtype=np.float32
        )
        rows.append((row - feature_means) / feature_stds)
    return torch.tensor(np.stack(rows), dtype=torch.float32, device=device)


def predict_sta_properties(
    smiles: str,
    model_path: str | Path = DEFAULT_STA_MODEL_PATH,
) -> dict[str, float | str]:
    model_path_text = str(Path(model_path))
    (
        artifact,
        targets,
        model,
        means,
        stds,
        mae,
        feature_names,
        feature_means,
        feature_stds,
    ) = _load_sta_predictor(model_path_text)
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        raise ValueError("invalid SMILES for STA prediction")
    device = next(model.parameters()).device
    indices = torch.tensor(
        np.array(
            [_smiles_to_indices(canonical, artifact["vocab"], int(artifact["max_length"]))],
            dtype=np.int64,
        ),
        dtype=torch.long,
        device=device,
    )
    feature_tensor = _predict_feature_tensor(
        canonical, feature_names, feature_means, feature_stds, device
    )
    with torch.no_grad():
        normalized = model(indices, feature_tensor).squeeze(0).cpu().numpy()
    values = normalized * stds + means
    result: dict[str, float | str] = {
        "sta_model": model_path_text,
        "sta_status": "predicted",
    }
    for target, value in zip(targets, values):
        result[f"sta_{target}"] = round(
            sta_value_to_public_units(target, float(value)),
            6,
        )
        if target in mae:
            result[f"sta_{target}_uncertainty"] = round(
                sta_value_to_public_units(target, float(mae[target])),
                6,
            )
    return result

def predict_sta_properties_batch(
    smiles_values: list[str] | tuple[str, ...],
    model_path: str | Path = DEFAULT_STA_MODEL_PATH,
    batch_size: int = 256,
    device: str | None = None,
) -> list[dict[str, float | str]]:
    """Predict one STA checkpoint for many molecules with batched CUDA inference."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_path_text = str(Path(model_path))
    selected_device = (torch.device(device) if device
                       else (resolve_device() if resolve_device
                             else torch.device("cpu")))
    artifact, targets, model, means, stds, mae, feature_names, feature_means, feature_stds = (
        _load_sta_predictor(model_path_text, str(selected_device))
    )
    canonicals = [canonicalize_smiles(smiles) for smiles in smiles_values]
    results: list[dict[str, float | str]] = []
    for start in range(0, len(canonicals), batch_size):
        chunk = canonicals[start : start + batch_size]
        indices = torch.tensor(
            np.asarray(
                [
                    _smiles_to_indices(
                        canonical, artifact["vocab"], int(artifact["max_length"])
                    )
                    for canonical in chunk
                ],
                dtype=np.int64,
            ),
            dtype=torch.long,
            device=selected_device,
        )
        features = _predict_features_tensor_batch(
            chunk, feature_names, feature_means, feature_stds, selected_device
        )
        with torch.inference_mode():
            normalized = model(indices, features).cpu().numpy()
        values = normalized * stds + means
        for prediction in values:
            result: dict[str, float | str] = {
                "sta_model": model_path_text,
                "sta_status": "predicted",
                "sta_inference_device": str(selected_device),
                "sta_batch_size": batch_size,
            }
            for target, value in zip(targets, prediction):
                result[f"sta_{target}"] = round(
                    sta_value_to_public_units(target, float(value)), 6
                )
                if target in mae:
                    result[f"sta_{target}_uncertainty"] = round(
                        sta_value_to_public_units(target, float(mae[target])), 6
                    )
            results.append(result)
    return results

def predict_sta_hidden_batch(
    smiles_values: list[str] | tuple[str, ...],
    model_path: str | Path = DEFAULT_STA_MODEL_PATH,
    batch_size: int = 512,
    device: str | None = None,
) -> list[np.ndarray]:
    """Extract mean-pooled final-layer token representations (B, d_model).

    No prediction head is applied, so the returned vectors are
    representation descriptors, not predicted property values.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_path_text = str(Path(model_path))
    selected_device = (torch.device(device) if device
                       else (resolve_device() if resolve_device
                             else torch.device("cpu")))
    artifact, targets, model, means, stds, mae, feature_names, _, _ = (
        _load_sta_predictor(model_path_text, str(selected_device))
    )
    canonicals = [canonicalize_smiles(smiles) for smiles in smiles_values]
    vectors: list[np.ndarray] = []
    for start in range(0, len(canonicals), batch_size):
        chunk = canonicals[start : start + batch_size]
        indices = torch.tensor(
            np.asarray(
                [
                    _smiles_to_indices(
                        canonical, artifact["vocab"], int(artifact["max_length"])
                    )
                    for canonical in chunk
                ],
                dtype=np.int64,
            ),
            dtype=torch.long,
            device=selected_device,
        )
        with torch.inference_mode():
            pooled = model.encode(indices).cpu().numpy()
        vectors.extend(v for v in pooled)
    return vectors


def sta_value_to_public_units(target: str, value: float) -> float:
    return float(value) * STA_PUBLIC_OUTPUT_SCALES.get(target, 1.0)

def _scale_sta_metric_values(values: dict[str, float]) -> dict[str, float]:
    return {
        target: round(
            sta_value_to_public_units(target, float(value)),
            6,
        )
        for target, value in values.items()
    }

def _load_sta_predictor(
    model_path: str,
    device: str = "cpu",
) -> tuple[
    dict[str, object],
    tuple[str, ...],
    torch.nn.Module,
    np.ndarray,
    np.ndarray,
    dict[str, float],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
]:
    """Load (and cache) one STA predictor checkpoint.

    Caching is keyed by ``(model_path, device)`` so a checkpoint is only loaded
    and rebuilt once per process per device. The model is moved to ``device``
    before it is returned, mirroring the prior per-call behaviour exactly, so
    inference output is bit-for-bit identical while repeated disk reads and
    model construction are eliminated.
    """

    return _load_sta_predictor_cached(str(model_path), str(device))


@lru_cache(maxsize=8)
def _load_sta_predictor_cached(
    model_path: str,
    device: str,
) -> tuple[
    dict[str, object],
    tuple[str, ...],
    torch.nn.Module,
    np.ndarray,
    np.ndarray,
    dict[str, float],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
]:
    artifact = torch_load_checkpoint(model_path, map_location="cpu")
    targets = tuple(str(target) for target in artifact["target_columns"])
    config = artifact["model_config"]
    feature_names = tuple(str(name) for name in artifact.get("feature_names", ()))
    model = _StaTransformer(
        vocab_size=len(artifact["vocab"]),
        length=int(artifact["max_length"]),
        output_dim=len(targets),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config.get("dropout", 0.1)),
        pooling=str(config.get("pooling", "mean")),
        feature_dim=len(feature_names),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    model = model.to(torch.device(device))
    means = np.asarray(artifact["target_means"], dtype=np.float32)
    stds = np.asarray(artifact["target_stds"], dtype=np.float32)
    metrics = artifact.get("metrics", {})
    raw_mae = metrics.get("validation_mae", {}) if isinstance(metrics, dict) else {}
    mae = {str(key): float(value) for key, value in raw_mae.items()}
    feature_means = np.asarray(
        artifact.get("feature_means", [0.0] * len(feature_names)),
        dtype=np.float32,
    )
    feature_stds = np.asarray(
        artifact.get("feature_stds", [1.0] * len(feature_names)),
        dtype=np.float32,
    )
    feature_stds = np.maximum(feature_stds, 1e-6).astype(np.float32)
    return (
        artifact,
        targets,
        model,
        means,
        stds,
        mae,
        feature_names,
        feature_means,
        feature_stds,
    )

def predict_sta_property_csv(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path = DEFAULT_STA_MODEL_PATH,
    smiles_column: str = "canonical_smiles",
) -> int:
    with Path(input_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    written: list[dict[str, object]] = []
    extra_fields: list[str] = []
    for row in rows:
        smiles = str(row.get(smiles_column) or row.get("smiles") or row.get("SMILES") or "")
        enriched = dict(row)
        try:
            prediction = predict_sta_properties(smiles, model_path=model_path)
        except Exception as exc:
            prediction = {"sta_status": "failed", "sta_failure_reason": str(exc)}
        for key in prediction:
            if key not in extra_fields and key not in fieldnames:
                extra_fields.append(key)
        enriched.update(prediction)
        written.append(enriched)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra_fields])
        writer.writeheader()
        writer.writerows(written)
    return len(written)

def canonicalize_smiles(smiles: str) -> str:
    text = smiles.replace("\xa0", "").replace(" ", "").strip()
    if not text:
        return ""
    if Chem is None:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)

def _target_column_map(fieldnames: list[str], targets: Iterable[str]) -> dict[str, str]:
    normalized = {_normalize_column(name): name for name in fieldnames}
    mapping: dict[str, str] = {}
    for target in targets:
        for alias in TARGET_COLUMN_ALIASES.get(target, (target,)):
            if _normalize_column(alias) in normalized:
                mapping[target] = normalized[_normalize_column(alias)]
                break
    return mapping

def _normalize_column(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("_", "")

def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result

def _read_sta_records(
    data_path: str | Path,
    target_columns: Iterable[str] | None,
    min_carbons: int | None = None,
    max_carbons: int | None = None,
    target_ranges: dict[str, tuple[float | None, float | None]] | None = None,
) -> list[dict[str, object]]:
    with Path(data_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        targets = tuple(target_columns) if target_columns is not None else tuple(
            field for field in fieldnames if field != "canonical_smiles"
        )
        records: list[dict[str, object]] = []
        for row in reader:
            smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
            if not smiles:
                continue
            carbon_count = _carbon_count_from_smiles(smiles)
            if min_carbons is not None and carbon_count is not None and carbon_count < min_carbons:
                continue
            if max_carbons is not None and carbon_count is not None and carbon_count > max_carbons:
                continue
            record: dict[str, object] = {"canonical_smiles": smiles}
            has_target = False
            for target in targets:
                value = _optional_float(row.get(target))
                value_range = (target_ranges or {}).get(target)
                if value is not None and value_range is not None:
                    lower, upper = value_range
                    if lower is not None and value < lower:
                        value = None
                    if upper is not None and value > upper:
                        value = None
                record[target] = value
                has_target = has_target or value is not None
            if has_target:
                records.append(record)
    return records

@lru_cache(maxsize=None)
def _carbon_count_from_smiles(smiles: str) -> int | None:
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

def _build_vocab(smiles_values: Iterable[str]) -> dict[str, int]:
    tokens = {"PAD"}
    for smiles in smiles_values:
        tokens.update(_tokenize(smiles))
    vocab = {"PAD": 0}
    for token in sorted(tokens - {"PAD"}):
        vocab[token] = len(vocab)
    return vocab

def _tokenize(smiles: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(smiles):
        if index < len(smiles) - 3 and smiles[index : index + 4] == "CIS-":
            tokens.append("CIS-")
            index += 4
            continue
        two = smiles[index : index + 2]
        if len(two) == 2 and two in SPECIAL_TOKENS:
            tokens.append(two)
            index += 2
        else:
            tokens.append(smiles[index])
            index += 1
    return tokens

def _smiles_to_indices(smiles: str, vocab: dict[str, int], length: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.int64)
    for index, token in enumerate(_tokenize(smiles)[:length]):
        result[index] = int(vocab.get(token, 0))
    return result

def _expanded_smiles_for_records(
    records: list[dict[str, object]],
    enumerations_per_molecule: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    rng = random.Random(seed)
    sources: list[int] = []
    smiles_values: list[str] = []
    for index, record in enumerate(records):
        canonical = str(record["canonical_smiles"])
        variants = _enumerate_smiles(canonical, enumerations_per_molecule, rng)
        for smiles in variants:
            sources.append(index)
            smiles_values.append(smiles)
    return np.array(sources, dtype=np.int64), smiles_values

def _enumerate_smiles(
    smiles: str,
    count: int,
    rng: random.Random,
) -> tuple[str, ...]:
    canonical = canonicalize_smiles(smiles) or smiles
    if count <= 1 or Chem is None:
        return (canonical,)
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return (canonical,)
    variants = {canonical}
    atom_indices = list(range(mol.GetNumAtoms()))
    attempts = max(10, count * 8)
    for _ in range(attempts):
        shuffled = atom_indices[:]
        rng.shuffle(shuffled)
        try:
            renumbered = Chem.RenumberAtoms(mol, shuffled)
            variant = Chem.MolToSmiles(renumbered, canonical=False)
        except Exception:
            continue
        if variant:
            variants.add(variant)
        if len(variants) >= count:
            break
    return tuple(sorted(variants))

def _target_arrays(
    records: list[dict[str, object]],
    targets: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros((len(records), len(targets)), dtype=np.float32)
    mask = np.zeros_like(y)
    for row_index, record in enumerate(records):
        for target_index, target in enumerate(targets):
            value = record.get(target)
            if value is not None:
                y[row_index, target_index] = float(value)
                mask[row_index, target_index] = 1.0
    return y, mask

def _target_stats(y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = np.maximum(mask.sum(axis=0), 1.0)
    means = (y * mask).sum(axis=0) / count
    variance = (((y - means) * mask) ** 2).sum(axis=0) / count
    stds = np.sqrt(np.maximum(variance, 1e-6))
    return means.astype(np.float32), stds.astype(np.float32)

class _StaDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
    ):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int):
        return (
            self.inputs[index],
            self.features[index],
            self.targets[index],
            self.masks[index],
        )

class _PositionalEncoding(nn.Module):
    def __init__(self, length: int, d_model: int):
        super().__init__()
        position = torch.arange(0, length).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        table = torch.zeros(length, d_model)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("table", table.unsqueeze(0))

    def forward(self, values):
        return values + self.table[:, : values.shape[1], :]

class _StaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        length: int,
        output_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        pooling: str = "mean",
        feature_dim: int = 0,
    ):
        super().__init__()
        if pooling != "additive":
            raise ValueError("paper-style STA uses additive token contribution pooling")
        self.d_model = d_model
        self.pooling = pooling
        self.feature_dim = feature_dim
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = _PositionalEncoding(length, d_model)
        self.attention_layers = nn.ModuleList(
            [_StaEncoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )
        self.prediction_head = nn.Linear(d_model, output_dim)
        self.feature_head = (
            nn.Linear(feature_dim, output_dim) if feature_dim > 0 else None
        )

    def forward(self, indices, features=None):
        mask = indices == 0
        embedded = self.embedding(indices) * math.sqrt(self.d_model)
        encoded = self.positional(embedded)
        for layer in self.attention_layers:
            encoded = layer(encoded, key_padding_mask=mask)
        token_values = self.prediction_head(encoded)
        valid = (~mask).unsqueeze(-1).float()
        result = (token_values * valid).sum(dim=1)
        if self.feature_dim > 0 and features is not None:
            result = result + self.feature_head(features)
        return result

    def encode(self, indices):
        """Generalized token extraction: mean-pool final-layer hidden states
        over valid tokens. Returns (B, d_model); does not use any prediction
        head, so no predicted property value enters downstream features."""
        mask = indices == 0
        embedded = self.embedding(indices) * math.sqrt(self.d_model)
        encoded = self.positional(embedded)
        for layer in self.attention_layers:
            encoded = layer(encoded, key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).float()
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return pooled

class _StaEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.ff_layers = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, attn_input, key_padding_mask):
        attn_output, _ = self.attention(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            key_padding_mask=key_padding_mask,
        )
        output_1 = self.layer_norm(attn_input + attn_output)
        return self.layer_norm(output_1 + self.ff_layers(output_1))

def _masked_loss(prediction, target, mask, loss: str = "mse"):
    diff = (prediction - target) * mask
    if loss in {"mae", "l1"}:
        return diff.abs().sum() / mask.sum().clamp_min(1.0)
    if loss == "mse":
        return (diff * diff).sum() / mask.sum().clamp_min(1.0)
    if loss == "smooth_l1":
        abs_diff = diff.abs()
        value = torch.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
        return value.sum() / mask.sum().clamp_min(1.0)
    raise ValueError(f"unknown STA loss: {loss}")

def _evaluate_loss(model, dataset, batch_size: int, device, loss: str = "mse") -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=device.type == "cuda")
    total = 0.0
    batches = 0
    with torch.no_grad():
        for inputs, features, target, mask in loader:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss_value = _masked_loss(
                    model(
                        inputs.to(device, non_blocking=True),
                        features.to(device, non_blocking=True),
                    ),
                    target.to(device, non_blocking=True),
                    mask.to(device, non_blocking=True),
                    loss=loss,
                )
            total += float(loss_value.detach().cpu())
            batches += 1
    return total / max(batches, 1)

@dataclass(frozen=True)
class _Metrics:
    validation_mae: dict[str, float]
    validation_rmse: dict[str, float]
    validation_r2: dict[str, float]

def _evaluate_metrics(
    model,
    dataset,
    targets: tuple[str, ...],
    means: np.ndarray,
    stds: np.ndarray,
    batch_size: int,
    device,
) -> _Metrics:
    model.eval()
    predictions = []
    actuals = []
    masks = []
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=device.type == "cuda")
    with torch.no_grad():
        for inputs, features, target, mask in loader:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred_tensor = model(
                    inputs.to(device, non_blocking=True),
                    features.to(device, non_blocking=True),
                )
            pred = pred_tensor.float().cpu().numpy()
            predictions.append(pred)
            actuals.append(target.numpy())
            masks.append(mask.numpy())
    pred_norm = np.vstack(predictions)
    true_norm = np.vstack(actuals)
    mask = np.vstack(masks)
    pred = pred_norm * stds + means
    true = true_norm * stds + means
    mae: dict[str, float] = {}
    rmse: dict[str, float] = {}
    r2: dict[str, float] = {}
    for index, target in enumerate(targets):
        active = mask[:, index] > 0
        if not np.any(active):
            continue
        pred_col = pred[active, index]
        true_col = true[active, index]
        errors = pred_col - true_col
        mae[target] = round(float(np.mean(np.abs(errors))), 6)
        rmse[target] = round(float(np.sqrt(np.mean(errors * errors))), 6)
        denom = float(np.sum((true_col - np.mean(true_col)) ** 2))
        if denom <= 1e-12:
            r2[target] = 0.0
        else:
            r2[target] = round(float(1.0 - np.sum(errors * errors) / denom), 6)
    return _Metrics(validation_mae=mae, validation_rmse=rmse, validation_r2=r2)
