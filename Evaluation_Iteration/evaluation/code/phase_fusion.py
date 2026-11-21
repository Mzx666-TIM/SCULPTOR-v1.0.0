from __future__ import annotations

import sys
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from chemistry import molar_mass_from_formula
from fuel_design.paths import EVAL_MODEL_DIR, RESULT_ROOT

DEFAULT_PHASE_TARGETS = ("melting_point_k", "boiling_point_k", "flash_point_k")
FEATURE_SCHEMA_VERSION = 4
LEGACY_FEATURE_SCHEMA_VERSIONS = (3,)
E3NN_FEATURE_PREFIX = "e3nn_3d_"
E3NN_FEATURE_DIM = 30
# Production melting/boiling/flash fusion heads are the MLP ensembles trained on
# the total deduplication library; they differ only in whether the e3nn 3D descriptors are
# part of the input (2D heads: 45 descriptors, 3D heads: 45+30), and neither
# takes a predicted property as an input:
#   schema 2/phase_fusion_gdb820k_mlp_ens2d      MLP ensemble, 2D only
#   schema 3/phase_fusion_gdb820k_mlp_ens3d      MLP ensemble, 2D + e3nn
# The reported strict run used the 2D head, so it is also the reproducibility
# default. The 3D head remains an explicitly selectable descriptor variant.
# The legacy ExtraTrees heads (schema2 / schema3 and their _repr variants) were
# retired on 2026-09-22 and are no longer part of the runtime layout.
DEFAULT_PHASE_FUSION_SCHEMA = "schema2_mlp"
PHASE_FUSION_SCHEMA_DIR = (
    EVAL_MODEL_DIR
    / "schema 2"
    / "phase_fusion_gdb820k_mlp_ens2d"
)
DEFAULT_PHASE_FUSION_MODEL_PATHS = {
    target: PHASE_FUSION_SCHEMA_DIR / f"{target}_fusion.joblib"
    for target in DEFAULT_PHASE_TARGETS
}
BASE_AUX_COLUMNS = (
    "density_g_cm3",
    "molar_volume_cm3_mol",
)
OPTIONAL_OUTPUT_COLUMNS = (
    "formation_enthalpy_kj_mol",
    "isp",
)
PHASE_COLUMNS = ("melting_point_k", "boiling_point_k", "flash_point_k")
REFERENCE_ONLY_COLUMNS = (
    "molecular_weight",
    "exact_mol_wt",
)
METADATA_COLUMNS = (
    "phase_fusion_feature_schema",
    "source_row_index",
)

class ReferenceOnlyFeatureModelError(ValueError):
    pass

@dataclass(frozen=True)
class FusionModelMetrics:
    target: str
    feature_set: str
    model_name: str
    train_rows: int
    validation_rows: int
    test_rows: int
    validation_mae: float
    validation_rmse: float
    validation_r2: float
    test_mae: float
    test_rmse: float
    test_r2: float
    model_path: str
    prediction_path: str
    feature_columns: tuple[str, ...]
    conformal90_halfwidth: float | None = None
    fit_seconds: float | None = None

@dataclass(frozen=True)
class PhaseFusionTrainingSummary:
    data_path: str
    feature_cache_path: str
    output_dir: str
    model_dir: str
    rows: int
    targets: tuple[str, ...]
    metrics: dict[str, FusionModelMetrics]

def predict_phase_fusion_property(
    smiles: str,
    target: str,
    model_path: str | Path | None = None,
    aux_values: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Predict one phase property from structural features plus optional aux values."""
    if target not in DEFAULT_PHASE_TARGETS:
        raise ValueError(f"unknown phase-fusion target: {target}")
    path = Path(model_path or DEFAULT_PHASE_FUSION_MODEL_PATHS[target])
    artifact = _load_phase_fusion_artifact(str(path))
    artifact_target = str(artifact.get("target") or target)
    if artifact_target != target:
        raise ValueError(f"phase-fusion model target mismatch: {artifact_target} != {target}")

    feature_row = phase_fusion_features_from_smiles(smiles)
    for column, value in (aux_values or {}).items():
        parsed = _optional_float(value)
        if parsed is not None:
            feature_row[column] = parsed
    columns = list(artifact.get("feature_columns") or [])
    if not columns:
        raise ValueError(f"phase-fusion model has no feature columns: {path}")
    forbidden_columns = sorted(set(columns).intersection(REFERENCE_ONLY_COLUMNS))
    artifact_schema = int(artifact.get("feature_schema_version", 0))
    if forbidden_columns or artifact_schema not in (
        *LEGACY_FEATURE_SCHEMA_VERSIONS,
        FEATURE_SCHEMA_VERSION,
    ):
        raise ReferenceOnlyFeatureModelError(
            "phase-fusion artifact is incompatible with the current calculated "
            f"molar-volume feature policy (schema={artifact_schema}, "
            f"forbidden={forbidden_columns}); retrain it"
        )
    if any(column.startswith(E3NN_FEATURE_PREFIX) for column in columns):
        three_d = e3nn_phase_features_from_smiles(smiles)
        if not three_d.get(f"{E3NN_FEATURE_PREFIX}available"):
            raise ValueError(f"3D features unavailable for {smiles}; cannot use the 3D phase head")
        feature_row.update(three_d)
    missing = [column for column in columns if column not in feature_row]
    if missing:
        raise ValueError(f"missing phase-fusion features: {missing}")

    frame = pd.DataFrame([feature_row])
    prediction = float(artifact["model"].predict(_numeric_matrix(frame, columns))[0])
    metrics = artifact.get("metrics") or {}
    uncertainty = _optional_float(metrics.get("test_mae"))
    uncertainty_source = "phase_fusion_test_mae"
    if uncertainty is None:
        uncertainty = _optional_float(metrics.get("validation_mae"))
        uncertainty_source = "phase_fusion_validation_mae"
    if uncertainty is None:
        uncertainty = _optional_float(metrics.get("test_rmse"))
        uncertainty_source = "phase_fusion_test_rmse"
    if uncertainty is None or not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("phase-fusion artifact has no valid empirical error scale")
    if not np.isfinite(prediction):
        raise ValueError("phase-fusion prediction must be finite")
    return {
        "target": target,
        "value": round(prediction, 6),
        "uncertainty": round(float(uncertainty), 6),
        "uncertainty_source": uncertainty_source,
        "uncertainty_kind": "empirical_error_scale_not_calibrated_coverage",
        "model": "phase_fusion",
        "model_path": str(path),
        "feature_set": str(artifact.get("feature_set") or ""),
    }

@lru_cache(maxsize=8)
def _load_phase_fusion_artifact(path: str) -> dict[str, object]:
    return joblib.load(path)

def train_phase_fusion_models(
    data_path: str | Path,
    output_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    feature_cache_path: str | Path | None = None,
    targets: Iterable[str] = DEFAULT_PHASE_TARGETS,
    random_state: int = 42,
    max_rows: int | None = None,
    include_e3nn_features: bool = False,
    workers: int = 1,
    model_family: str = "mlp_ensemble",
    ensemble_size: int = 2,
    mlp_epochs: int = 100,
    mlp_batch_size: int = 1024,
    mlp_learning_rate: float = 1e-3,
    mlp_patience: int = 12,
) -> PhaseFusionTrainingSummary:
    """Train phase-property heads in the same artifact format used at runtime.

    The default reproduces the reported two-member 2D MLP ensemble. Set
    ``include_e3nn_features=True`` for the optional 3D descriptor variant or
    ``model_family='legacy_tree'`` to rerun the retired tree-model baseline.
    """
    if model_family not in {"mlp_ensemble", "legacy_tree"}:
        raise ValueError("model_family must be 'mlp_ensemble' or 'legacy_tree'")
    data_file = Path(data_path)
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else RESULT_ROOT
        / (
            "phase_fusion_total_reference_mlp_ens3d"
            if include_e3nn_features
            else "phase_fusion_total_reference_mlp_ens2d"
        )
    )
    model_root = (
        Path(model_dir)
        if model_dir is not None
        else EVAL_MODEL_DIR
        / ("schema 3" if include_e3nn_features else "schema 2")
        / (
            "phase_fusion_gdb820k_mlp_ens3d"
            if include_e3nn_features
            else "phase_fusion_gdb820k_mlp_ens2d"
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    cache_file = (
        Path(feature_cache_path)
        if feature_cache_path is not None
        else output_root
        / (
            "total_reference_phase_fusion_features_3d.csv"
            if include_e3nn_features
            else "total_reference_phase_fusion_features_2d.csv"
        )
    )

    features = load_or_build_phase_fusion_features(
        data_file,
        cache_file,
        max_rows=max_rows,
        include_e3nn_features=include_e3nn_features,
        workers=workers,
    )
    metrics: dict[str, FusionModelMetrics] = {}
    all_predictions: pd.DataFrame | None = features[["canonical_smiles"]].copy()
    for target in tuple(targets):
        metric = _train_one_target(
            features,
            target=target,
            output_dir=output_root,
            model_dir=model_root,
            random_state=random_state,
            model_family=model_family,
            ensemble_size=ensemble_size,
            mlp_epochs=mlp_epochs,
            mlp_batch_size=mlp_batch_size,
            mlp_learning_rate=mlp_learning_rate,
            mlp_patience=mlp_patience,
        )
        metrics[target] = metric
        predictions = pd.read_csv(metric.prediction_path)
        all_predictions[f"{target}_true"] = predictions[target]
        all_predictions[f"{target}_predicted"] = predictions[f"fusion_{target}"]
        all_predictions[f"{target}_error"] = predictions[f"{target}_error"]
        all_predictions[f"{target}_split"] = predictions["split"]

    if all_predictions is not None:
        all_predictions.to_csv(
            output_root / "phase_fusion_all_predictions.csv",
            index=False,
            encoding="utf-8",
        )
    summary = PhaseFusionTrainingSummary(
        data_path=str(data_file),
        feature_cache_path=str(cache_file),
        output_dir=str(output_root),
        model_dir=str(model_root),
        rows=len(features),
        targets=tuple(targets),
        metrics=metrics,
    )
    (output_root / "phase_fusion_summary.json").write_text(
        json.dumps(_summary_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary

def _build_feature_row_worker(
    payload: tuple[int, str, object, dict[str, object], bool],
) -> dict[str, object] | None:
    """Build one phase-fusion feature row (multiprocessing worker entry point)."""
    index, smiles, density_text, row, include_e3nn = payload
    try:
        feature_row = phase_fusion_features_from_smiles(smiles)
        if include_e3nn:
            feature_row.update(e3nn_phase_features_from_smiles(smiles))
    except Exception:
        return None
    density = _optional_float(density_text)
    feature_row["source_row_index"] = index
    feature_row["density_g_cm3"] = density
    feature_row["molar_volume_cm3_mol"] = _derived_molar_volume(
        formula=str(feature_row.get("formula") or ""),
        density_g_cm3=density,
    )
    for column in (
        "formation_enthalpy_kj_mol",
        "isp",
        *PHASE_COLUMNS,
    ):
        feature_row[column] = _optional_float(row.get(column))
    return feature_row

def load_or_build_phase_fusion_features(
    data_path: str | Path,
    feature_cache_path: str | Path,
    max_rows: int | None = None,
    include_e3nn_features: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    cache = Path(feature_cache_path)
    if cache.exists() and max_rows is None:
        cached = pd.read_csv(cache)
        if (
            "phase_fusion_feature_schema" in cached.columns
            and "molar_volume_cm3_mol" in cached.columns
            and not cached.empty
            and int(cached["phase_fusion_feature_schema"].iloc[0])
            == FEATURE_SCHEMA_VERSION
        ):
            has_e3nn = any(
                column.startswith(E3NN_FEATURE_PREFIX)
                and column != f"{E3NN_FEATURE_PREFIX}available"
                for column in cached.columns
            )
            if has_e3nn or not include_e3nn_features:
                return _select_phase_feature_schema(cached, include_e3nn_features)

    partial = cache.with_suffix(cache.suffix + ".partial")
    cache.parent.mkdir(parents=True, exist_ok=True)
    completed_source_rows = 0
    if partial.exists():
        resumed = pd.read_csv(partial)
        if (
            not resumed.empty
            and "phase_fusion_feature_schema" in resumed.columns
            and int(resumed["phase_fusion_feature_schema"].iloc[0])
            == FEATURE_SCHEMA_VERSION
            and "source_row_index" in resumed.columns
        ):
            resumed_has_e3nn = any(
                column.startswith(E3NN_FEATURE_PREFIX)
                and column != f"{E3NN_FEATURE_PREFIX}available"
                for column in resumed.columns
            )
            if resumed_has_e3nn == include_e3nn_features:
                completed_source_rows = int(resumed["source_row_index"].max())
            else:
                partial.unlink()
        else:
            partial.unlink()

    def pending_rows():
        # Stream the source instead of materialising hundreds of thousands of
        # dictionaries. Ordered mapping also makes partial-cache resume safe.
        with Path(data_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                if max_rows is not None and index > max_rows:
                    break
                if index <= completed_source_rows:
                    continue
                smiles = str(
                    row.get("canonical_smiles") or row.get("smiles") or ""
                ).strip()
                if not smiles:
                    continue
                yield (
                    index,
                    smiles,
                    row.get("density_g_cm3"),
                    row,
                    include_e3nn_features,
                )

    rows: list[dict[str, object]] = []
    wrote_header = partial.exists() and partial.stat().st_size > 0
    flush_size = 5000

    def flush() -> None:
        nonlocal rows, wrote_header
        if not rows:
            return
        pd.DataFrame(rows).to_csv(
            partial,
            mode="a",
            header=not wrote_header,
            index=False,
            encoding="utf-8",
        )
        wrote_header = True
        last_index = max(int(row["source_row_index"]) for row in rows)
        rows.clear()
        print(f"phase fusion features: {last_index} source rows processed", flush=True)

    if workers > 1:
        import multiprocessing

        with multiprocessing.Pool(workers) as pool:
            for feature_row in pool.imap(
                _build_feature_row_worker,
                pending_rows(),
                chunksize=64,
            ):
                if feature_row is None:
                    continue
                rows.append(feature_row)
                if len(rows) >= flush_size:
                    flush()
    else:
        for payload in pending_rows():
            feature_row = _build_feature_row_worker(payload)
            if feature_row is None:
                continue
            rows.append(feature_row)
            if len(rows) >= flush_size:
                flush()
    flush()
    if not partial.exists():
        raise ValueError("no phase-fusion feature rows were generated")
    partial.replace(cache)
    frame = pd.read_csv(cache)
    return _select_phase_feature_schema(frame, include_e3nn_features)


def _select_phase_feature_schema(
    frame: pd.DataFrame,
    include_e3nn_features: bool,
) -> pd.DataFrame:
    """Return an explicit 2D or 3D feature frame, independent of cache history."""

    e3nn_columns = [
        column for column in frame.columns if column.startswith(E3NN_FEATURE_PREFIX)
    ]
    descriptor_columns = [
        column for column in e3nn_columns
        if column != f"{E3NN_FEATURE_PREFIX}available"
    ]
    if include_e3nn_features:
        if len(descriptor_columns) != E3NN_FEATURE_DIM:
            raise ValueError(
                f"3D phase training requires {E3NN_FEATURE_DIM} e3nn descriptors; "
                f"found {len(descriptor_columns)}"
            )
        return frame
    if not e3nn_columns:
        return frame
    return frame.drop(columns=e3nn_columns)

def _compute_phase_fusion_features(smiles: str) -> dict[str, float | str]:
    Chem, Descriptors, Lipinski, rdMolDescriptors, GraphDescriptors = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    atoms = list(mol.GetAtoms())
    carbon_count = sum(1 for atom in atoms if atom.GetAtomicNum() == 6)
    hydrogen_count = sum(atom.GetTotalNumHs() for atom in atoms)
    degrees = [int(atom.GetDegree()) for atom in atoms if atom.GetAtomicNum() == 6]
    degree_counts = {degree: degrees.count(degree) for degree in range(5)}
    ring_info = mol.GetRingInfo()
    ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    rank_counts: dict[int, int] = {}
    for rank in ranks:
        rank_counts[int(rank)] = rank_counts.get(int(rank), 0) + 1
    equivalence_classes = len(rank_counts)
    largest_equivalence_class = max(rank_counts.values()) if rank_counts else 0
    edge_count = int(mol.GetNumBonds())
    cyclomatic = edge_count - int(mol.GetNumAtoms()) + 1

    return {
        "canonical_smiles": canonical,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "phase_fusion_feature_schema": float(FEATURE_SCHEMA_VERSION),
        "heavy_atom_count": float(mol.GetNumHeavyAtoms()),
        "carbon_count": float(carbon_count),
        "hydrogen_count": float(hydrogen_count),
        "h_to_c_ratio": float(hydrogen_count / carbon_count) if carbon_count else 0.0,
        "edge_count": float(edge_count),
        "cyclomatic_number": float(cyclomatic),
        "avg_carbon_degree": float(np.mean(degrees)) if degrees else 0.0,
        "carbon_degree_variance": float(np.var(degrees)) if degrees else 0.0,
        "terminal_carbon_count": float(degree_counts.get(1, 0)),
        "secondary_carbon_count": float(degree_counts.get(2, 0)),
        "tertiary_carbon_count": float(degree_counts.get(3, 0)),
        "quaternary_carbon_count": float(degree_counts.get(4, 0)),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "aliphatic_ring_count": float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "saturated_ring_count": float(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "bridgehead_atom_count": float(_safe_descriptor(rdMolDescriptors.CalcNumBridgeheadAtoms, mol)),
        "spiro_atom_count": float(_safe_descriptor(rdMolDescriptors.CalcNumSpiroAtoms, mol)),
        "small_ring_count": float(sum(1 for size in ring_sizes if size in (3, 4))),
        "ring3_count": float(sum(1 for size in ring_sizes if size == 3)),
        "ring4_count": float(sum(1 for size in ring_sizes if size == 4)),
        "ring5_count": float(sum(1 for size in ring_sizes if size == 5)),
        "ring6_count": float(sum(1 for size in ring_sizes if size == 6)),
        "large_ring_count": float(sum(1 for size in ring_sizes if size >= 7)),
        "min_ring_size": float(min(ring_sizes)) if ring_sizes else 0.0,
        "max_ring_size": float(max(ring_sizes)) if ring_sizes else 0.0,
        "mean_ring_size": float(np.mean(ring_sizes)) if ring_sizes else 0.0,
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "fraction_sp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "labute_asa": float(rdMolDescriptors.CalcLabuteASA(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "mol_logp": float(Descriptors.MolLogP(mol)),
        "mol_mr": float(Descriptors.MolMR(mol)),
        "bertz_ct": float(GraphDescriptors.BertzCT(mol)),
        "balaban_j": float(GraphDescriptors.BalabanJ(mol)),
        "kappa1": float(Descriptors.Kappa1(mol)),
        "kappa2": float(Descriptors.Kappa2(mol)),
        "kappa3": float(Descriptors.Kappa3(mol)),
        "chi0": float(Descriptors.Chi0(mol)),
        "chi1": float(Descriptors.Chi1(mol)),
        "chi2n": float(Descriptors.Chi2n(mol)),
        "chi3n": float(Descriptors.Chi3n(mol)),
        "chi4n": float(Descriptors.Chi4n(mol)),
        "equivalence_class_count": float(equivalence_classes),
        "largest_equivalence_class": float(largest_equivalence_class),
        "symmetry_fraction": float(largest_equivalence_class / len(ranks)) if ranks else 0.0,
    }


@lru_cache(maxsize=8192)
def _phase_fusion_features_cached(smiles: str) -> tuple[tuple[str, float | str], ...]:
    """Cache the 45-dimensional 2D descriptor row per molecule.

    The three phase targets share the same descriptor row, so caching it removes
    two redundant RDKit descriptor computations per molecule without changing a
    single value.
    """

    return tuple(_compute_phase_fusion_features(smiles).items())


def phase_fusion_features_from_smiles(smiles: str) -> dict[str, float | str]:
    """2D descriptor row for one molecule (cached, returned as a fresh copy)."""

    return dict(_phase_fusion_features_cached(smiles))

@lru_cache(maxsize=8192)
def e3nn_phase_features_from_smiles(smiles: str) -> dict[str, float]:
    """Return deterministic cached 3D/e3nn features shared by all phase heads."""
    from property_model.data import geometry_from_smiles
    from property_model.e3nn_regressor import molecular_e3nn_features

    features = {
        f"{E3NN_FEATURE_PREFIX}feature_{index:02d}": 0.0
        for index in range(E3NN_FEATURE_DIM)
    }
    features[f"{E3NN_FEATURE_PREFIX}available"] = 0.0
    try:
        canonical = _canonical_smiles(smiles)
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "little") & 0x7FFFFFFF
        geometry = geometry_from_smiles(canonical, seed=seed)
        values = molecular_e3nn_features(geometry).detach().cpu().numpy().astype(float)
        if len(values) != E3NN_FEATURE_DIM:
            raise ValueError(
                f"unexpected e3nn feature width: {len(values)} != {E3NN_FEATURE_DIM}"
            )
        for index, value in enumerate(values.tolist()):
            features[f"{E3NN_FEATURE_PREFIX}feature_{index:02d}"] = float(value)
        features[f"{E3NN_FEATURE_PREFIX}available"] = 1.0
    except Exception:
        pass
    return features

def _canonical_smiles(smiles: str) -> str:
    Chem, *_ = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return str(Chem.MolToSmiles(mol, canonical=True))

def _train_one_target(
    features: pd.DataFrame,
    target: str,
    output_dir: Path,
    model_dir: Path,
    random_state: int,
    model_family: str = "mlp_ensemble",
    ensemble_size: int = 2,
    mlp_epochs: int = 100,
    mlp_batch_size: int = 1024,
    mlp_learning_rate: float = 1e-3,
    mlp_patience: int = 12,
) -> FusionModelMetrics:
    frame = features.dropna(subset=[target]).copy()
    train_valid, test = train_test_split(
        frame,
        test_size=0.1,
        random_state=random_state,
    )
    train, validation = train_test_split(
        train_valid,
        test_size=1.0 / 9.0,
        random_state=random_state + 1,
    )
    feature_sets = _candidate_feature_sets(target, frame)
    if model_family == "mlp_ensemble":
        columns = feature_sets["structural"]
        return _train_one_target_mlp(
            frame,
            train,
            validation,
            test,
            target=target,
            columns=columns,
            output_dir=output_dir,
            model_dir=model_dir,
            random_state=random_state,
            ensemble_size=ensemble_size,
            epochs=mlp_epochs,
            batch_size=mlp_batch_size,
            learning_rate=mlp_learning_rate,
            patience=mlp_patience,
        )
    candidates = _candidate_models(random_state)
    best: tuple[float, FusionModelMetrics, object, list[str]] | None = None
    leaderboard_rows: list[dict[str, object]] = []
    for feature_set_name, columns in feature_sets.items():
        x_train = _numeric_matrix(train, columns)
        y_train = train[target].to_numpy(dtype=np.float32)
        x_val = _numeric_matrix(validation, columns)
        y_val = validation[target].to_numpy(dtype=np.float32)
        for model_name, model in candidates.items():
            print(f"training fusion {target}: {feature_set_name}/{model_name}", flush=True)
            fitted_model = clone(model)
            fitted_model.fit(x_train, y_train)
            val_pred = fitted_model.predict(x_val)
            val_mae, val_rmse, val_r2 = _regression_metrics(y_val, val_pred)
            leaderboard_rows.append(
                {
                    "target": target,
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    "validation_mae": round(val_mae, 6),
                    "validation_rmse": round(val_rmse, 6),
                    "validation_r2": round(val_r2, 6),
                    "feature_count": len(columns),
                }
            )
            score = val_rmse
            if best is None or score < best[0]:
                x_test = _numeric_matrix(test, columns)
                y_test = test[target].to_numpy(dtype=np.float32)
                test_pred = fitted_model.predict(x_test)
                test_mae, test_rmse, test_r2 = _regression_metrics(y_test, test_pred)
                metric = FusionModelMetrics(
                    target=target,
                    feature_set=feature_set_name,
                    model_name=model_name,
                    train_rows=len(train),
                    validation_rows=len(validation),
                    test_rows=len(test),
                    validation_mae=round(val_mae, 6),
                    validation_rmse=round(val_rmse, 6),
                    validation_r2=round(val_r2, 6),
                    test_mae=round(test_mae, 6),
                    test_rmse=round(test_rmse, 6),
                    test_r2=round(test_r2, 6),
                    model_path=str(model_dir / f"{target}_fusion.joblib"),
                    prediction_path=str(output_dir / f"{target}_fusion_predictions.csv"),
                    feature_columns=tuple(columns),
                )
                best = (score, metric, fitted_model, columns)

    if best is None:
        raise ValueError(f"no model candidate could be trained for {target}")
    pd.DataFrame(leaderboard_rows).sort_values(
        ["validation_rmse", "validation_mae"],
    ).to_csv(
        output_dir / f"{target}_fusion_leaderboard.csv",
        index=False,
        encoding="utf-8",
    )
    _, metric, model, columns = best
    joblib.dump(
        {
            "model": model,
            "target": target,
            "feature_set": metric.feature_set,
            "feature_columns": columns,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "metrics": asdict(metric),
        },
        metric.model_path,
    )
    all_frame = frame.copy()
    all_pred = model.predict(_numeric_matrix(all_frame, columns))
    all_frame["split"] = "train"
    all_frame.loc[validation.index, "split"] = "validation"
    all_frame.loc[test.index, "split"] = "test"
    prediction = all_frame[["canonical_smiles", target, "split"]].copy()
    prediction[f"fusion_{target}"] = all_pred
    prediction[f"{target}_error"] = prediction[f"fusion_{target}"] - prediction[target]
    prediction[f"{target}_abs_error"] = prediction[f"{target}_error"].abs()
    prediction.to_csv(metric.prediction_path, index=False, encoding="utf-8")
    return metric


def _train_one_target_mlp(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str,
    columns: list[str],
    output_dir: Path,
    model_dir: Path,
    random_state: int,
    ensemble_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
) -> FusionModelMetrics:
    from mlp_phase_head import fit_mlp_ensemble

    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive")
    has_e3nn = any(column.startswith(E3NN_FEATURE_PREFIX) for column in columns)
    feature_set = "mlp_ens3d" if has_e3nn else "mlp_ens2d"
    started = time.perf_counter()
    model, training_details = fit_mlp_ensemble(
        _numeric_matrix(train, columns),
        train[target].to_numpy(dtype=np.float32),
        _numeric_matrix(validation, columns),
        validation[target].to_numpy(dtype=np.float32),
        seeds=tuple(random_state + index for index in range(ensemble_size)),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        target=target,
        feature_set=feature_set,
    )
    fit_seconds = time.perf_counter() - started
    y_validation = validation[target].to_numpy(dtype=np.float32)
    validation_pred = model.predict(_numeric_matrix(validation, columns))
    val_mae, val_rmse, val_r2 = _regression_metrics(y_validation, validation_pred)
    y_test = test[target].to_numpy(dtype=np.float32)
    test_pred = model.predict(_numeric_matrix(test, columns))
    test_mae, test_rmse, test_r2 = _regression_metrics(y_test, test_pred)
    conformal90 = float(training_details["conformal90_halfwidth"])
    model_path = model_dir / f"{target}_fusion.joblib"
    prediction_path = output_dir / f"{target}_fusion_predictions.csv"
    metric = FusionModelMetrics(
        target=target,
        feature_set=feature_set,
        model_name="torch_mlp_ensemble",
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        validation_mae=round(val_mae, 6),
        validation_rmse=round(val_rmse, 6),
        validation_r2=round(val_r2, 6),
        test_mae=round(test_mae, 6),
        test_rmse=round(test_rmse, 6),
        test_r2=round(test_r2, 6),
        model_path=str(model_path),
        prediction_path=str(prediction_path),
        feature_columns=tuple(columns),
        conformal90_halfwidth=round(conformal90, 6),
        fit_seconds=round(fit_seconds, 3),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_metrics = {
        "test_mae": metric.test_mae,
        "test_rmse": metric.test_rmse,
        "test_r2": metric.test_r2,
        "validation_mae": metric.validation_mae,
        "validation_rmse": metric.validation_rmse,
        "validation_r2": metric.validation_r2,
        "conformal90_halfwidth": metric.conformal90_halfwidth,
        "train_rows": metric.train_rows,
        "validation_rows": metric.validation_rows,
        "test_rows": metric.test_rows,
        "fit_seconds": metric.fit_seconds,
        "training_details": training_details,
    }
    joblib.dump(
        {
            "model": model,
            "target": target,
            "feature_set": feature_set,
            "feature_columns": columns,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "metrics": artifact_metrics,
        },
        model_path,
    )
    all_frame = frame.copy()
    all_pred = model.predict(_numeric_matrix(all_frame, columns))
    all_frame["split"] = "train"
    all_frame.loc[validation.index, "split"] = "validation"
    all_frame.loc[test.index, "split"] = "test"
    prediction = all_frame[["canonical_smiles", target, "split"]].copy()
    prediction[f"fusion_{target}"] = all_pred
    prediction[f"{target}_error"] = prediction[f"fusion_{target}"] - prediction[target]
    prediction[f"{target}_abs_error"] = prediction[f"{target}_error"].abs()
    prediction.to_csv(prediction_path, index=False, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "target": target,
                "feature_set": feature_set,
                "model_name": metric.model_name,
                "validation_mae": metric.validation_mae,
                "validation_rmse": metric.validation_rmse,
                "validation_r2": metric.validation_r2,
                "feature_count": len(columns),
            }
        ]
    ).to_csv(
        output_dir / f"{target}_fusion_leaderboard.csv",
        index=False,
        encoding="utf-8",
    )
    return metric

def _candidate_feature_sets(target: str, frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric_columns = [
        column
        for column in frame.columns
        if column != "canonical_smiles"
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    structural = [
        column
        for column in numeric_columns
        if column not in (
            *BASE_AUX_COLUMNS,
            *OPTIONAL_OUTPUT_COLUMNS,
            *PHASE_COLUMNS,
            *REFERENCE_ONLY_COLUMNS,
            *METADATA_COLUMNS,
            f"{E3NN_FEATURE_PREFIX}available",
        )
    ]
    base_aux = [column for column in BASE_AUX_COLUMNS if column in frame.columns]
    return {
        "structural": structural,
        "structural_density_molar_volume": structural + base_aux,
    }

def _candidate_models(random_state: int) -> dict[str, object]:
    return {
        "hist_gradient_boosting_deep": HistGradientBoostingRegressor(
            max_iter=900,
            learning_rate=0.045,
            max_leaf_nodes=63,
            min_samples_leaf=18,
            l2_regularization=0.02,
            random_state=random_state,
        ),
        "hist_gradient_boosting_smooth": HistGradientBoostingRegressor(
            max_iter=700,
            learning_rate=0.055,
            max_leaf_nodes=31,
            min_samples_leaf=35,
            l2_regularization=0.08,
            random_state=random_state + 11,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=96,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

def _numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    matrix = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return matrix.to_numpy(dtype=np.float32)

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2

def _summary_to_jsonable(summary: PhaseFusionTrainingSummary) -> dict[str, object]:
    payload = asdict(summary)
    payload["metrics"] = {
        target: asdict(metric) for target, metric in summary.metrics.items()
    }
    return payload

def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _derived_molar_volume(
    formula: str,
    density_g_cm3: float | None,
) -> float | None:
    if not formula or density_g_cm3 is None or density_g_cm3 <= 0.0:
        return None
    return round(molar_mass_from_formula(formula) / density_g_cm3, 6)

def _safe_descriptor(function, mol) -> float:
    try:
        return float(function(mol))
    except Exception:
        return 0.0

def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, GraphDescriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise RuntimeError("RDKit is required for phase-fusion features.") from exc
    return Chem, Descriptors, Lipinski, rdMolDescriptors, GraphDescriptors
