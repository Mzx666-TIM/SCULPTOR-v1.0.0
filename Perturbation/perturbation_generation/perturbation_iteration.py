# -*- coding: utf-8 -*-
"""Iteration model of the perturbation line: predict the property change of one step.

The model is trained on the pairs a run folder already contains - the generation
library supplies the parents, the first perturbation campaign the products - so
what it learns is the property-change trend of the operator mix itself: given the
parent's descriptors, its current distance to JP-10 and the operator, how do
density, mass heating value and melting point move?

A later iteration uses that trend to steer the search: every enumerated
perturbation is scored by the margin the child is predicted to gain, the round
keeps the candidates whose predicted child is closest to Tier 1 (plus an
exploration share), and only those go through screening and property prediction.
The goal of the loop is to move molecules that are not in Tier 1 into Tier 1.

Margins are measured against the JP-10 reference of the *same* prediction
pipeline, so a positive margin means "at or better than JP-10" for that
property. The three margins are the tier axis of :mod:`tier_routing`.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np

from chemistry import (
    FIVE_MEMBER_RING_PREFERENCE_WEIGHT,
    FOUR_MEMBER_RING_STRAIN_WEIGHT,
    RING_COUNT_DEFINITION,
    SIX_MEMBER_RING_PREFERENCE_WEIGHT,
    THREE_MEMBER_RING_STRAIN_WEIGHT,
    ring_count_from_smiles,
)


PERTURBATION_OPERATIONS = (
    "bond_rewire",
    "ch2_bridge_move",
    "local_skeleton_rearrange",
    "methyl_shift",
    "methylation",
    "ring_adjust",
    # no training pairs yet: the round treats its property change as neutral and
    # lets the real prediction pipeline decide, while its ring gain is positive by
    # construction (a strained ring becomes a larger one).
    "ring_expand",
)

#: descriptor columns of a molecule row that the model learns from. These are
#: the descriptors recorded for *every* scored molecule - generation candidates
#: and perturbation products alike - so training (parents from the generation
#: library) and scoring (parents from an earlier round) always see the same
#: input.
DESCRIPTOR_KEYS = (
    "carbon_count",
    "ring_count",
    "fraction_sp3",
    "h_to_c_ratio",
    "sa_score",
    "molecular_weight_g_mol",
)

#: margin of one compared property: how much better than JP-10 the molecule is
MARGIN_KEYS = ("margin_density", "margin_heating_value", "margin_melting_point")
MARGIN_SOURCE_COLUMNS = {
    "margin_density": "external_density_g_cm3",
    "margin_heating_value": "external_heating_value_mj_kg",
    "margin_melting_point": "external_melting_point_k",
}

#: JP-10 values used when a run folder does not record its own reference
DOCUMENTED_JP10_REFERENCE = {
    "density_g_cm3": 0.962674,
    "heating_value_mj_kg": 42.082709,
    "melting_point_k": 230.126,
}

FEATURE_SCHEMA_VERSION = "perturbation_delta_v1"
#: Ring-size preference of the iteration: five/six-membered rings are the wanted
#: motifs, three/four-membered rings are the thermal-stability risk. Mirrors the
#: full-molecule descriptor weights and scales them into the promise score.
RING_REWARD_FACTOR = 2.0
RING_STRAIN_FACTOR = 6.0
RING_STRAIN_REFERENCE = 6.0
RING_REWARD_REFERENCE = 2.0


def ring_preference(smiles: str) -> tuple[float, float, int, int]:
    """(reward, penalty, strained rings, preferred rings) of one molecule."""

    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
    except Exception:
        molecule = None
    if molecule is None:
        return 0.0, 0.0, 0, 0
    strained = 0
    preferred = 0
    for ring in molecule.GetRingInfo().AtomRings():
        size = len(ring)
        if size == 3:
            strained += THREE_MEMBER_RING_STRAIN_WEIGHT
        elif size == 4:
            strained += FOUR_MEMBER_RING_STRAIN_WEIGHT
        elif size == 5:
            preferred += FIVE_MEMBER_RING_PREFERENCE_WEIGHT
        elif size == 6:
            preferred += SIX_MEMBER_RING_PREFERENCE_WEIGHT
    reward = min(preferred / RING_REWARD_REFERENCE, 1.0)
    penalty = min(strained / RING_STRAIN_REFERENCE, 1.0)
    return reward, penalty, strained, preferred


def ring_profile(smiles: str) -> tuple[int, int]:
    """(strained three/four-membered rings weighted, five/six-membered rings weighted)."""

    reward, penalty, strained, preferred = ring_preference(smiles)
    return strained, preferred


#: how strongly the ring direction steers the operator mix, relative to the
#: property-change term the weights were first built from
OPERATOR_RING_DIRECTION_GAIN = 1.5


def operator_ring_gains(dataset: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Mean ring-direction gain of each operator (positive = towards JP-10-like).

    ``(five/six-membered rings built) - (strained three/four-membered rings
    built)`` per perturbation step, averaged over the training pairs of the
    operator. It is the number that decides which operators deserve more of a
    round's perturbations.
    """

    totals = {operation: 0.0 for operation in PERTURBATION_OPERATIONS}
    counts = {operation: 0 for operation in PERTURBATION_OPERATIONS}
    for row in dataset:
        operation = str(row.get("operation") or "")
        if operation not in totals:
            continue
        parent = str(row.get("parent_smiles") or row.get("smiles") or "")
        child = str(row.get("smiles") or "")
        parent_strained, parent_preferred = ring_profile(parent)
        child_strained, child_preferred = ring_profile(child)
        totals[operation] += (child_preferred - parent_preferred) - (
            child_strained - parent_strained
        )
        counts[operation] += 1
    return {
        operation: (totals[operation] / counts[operation]) if counts[operation] else 0.0
        for operation in PERTURBATION_OPERATIONS
    }
#: family used when the caller does not pick one
DEFAULT_MODEL_FAMILY = "hist_gradient_boosting"
#: parallel workers of the ensemble families. The training matrix is small, so a
#: handful of threads is enough and a wide fan-out would only cost memory.
MAX_TRAINING_WORKERS = 4


def _worker_count() -> int:
    return max(1, min(MAX_TRAINING_WORKERS, os.cpu_count() or 1))


def _fit_candidate(
    model,
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray | None,
    anchor_mask: np.ndarray | None,
) -> str:
    """Fit one candidate model and report which samples it could use.

    A model that takes sample weights sees the whole training set (the low-weight
    anchors included); one that does not is fitted on the parent/product pairs
    alone, so every family is compared on a comparable footing.
    """

    anchors = (
        np.zeros(len(features), dtype=bool)
        if anchor_mask is None
        else np.asarray(anchor_mask, dtype=bool)
    )
    if weights is not None:
        try:
            model.fit(features, targets, sample_weight=weights)
            return "all_samples"
        except (TypeError, ValueError):
            pass
    if anchors.any():
        keep = ~anchors
        model.fit(features[keep], targets[keep])
        return "pairs_only"
    model.fit(features, targets)
    return "all_samples"


def candidate_model_factories() -> dict[str, "object"]:
    """Every model family the comparison can fit, keyed by family name."""

    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def scaled(estimator) -> object:
        # MLP / KNN / Ridge need centred, finite inputs; the trees do not.
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), estimator
        )

    return {
        "hist_gradient_boosting": lambda seed: HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.06,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1e-3,
            random_state=seed,
        ),
        "hist_gradient_boosting_deep": lambda seed: HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=700,
            max_leaf_nodes=63,
            min_samples_leaf=10,
            l2_regularization=1e-3,
            random_state=seed,
        ),
        "random_forest": lambda seed: RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=_worker_count(),
            random_state=seed,
        ),
        "extra_trees": lambda seed: ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=_worker_count(),
            random_state=seed,
        ),
        "gradient_boosting": lambda seed: GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        ),
        "mlp": lambda seed: scaled(
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=seed,
            )
        ),
        "knn": lambda seed: scaled(
            KNeighborsRegressor(n_neighbors=25, weights="distance")
        ),
        "ridge": lambda seed: scaled(Ridge(alpha=1.0)),
    }


def feature_columns() -> list[str]:
    """Column order of the model input, shared by training and scoring."""

    columns = ["op_%s" % operation for operation in PERTURBATION_OPERATIONS]
    columns += ["parent_%s" % key for key in DESCRIPTOR_KEYS]
    columns += [
        "parent_margin_density",
        "parent_margin_heating_value",
        "parent_margin_melting_point",
        "parent_beats",
    ]
    columns += ["delta_%s" % key for key in DESCRIPTOR_KEYS]
    return columns


def _number(value: object, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _carbon_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name[:1] == "C" and path.name[1:].isdigit()
        ),
        key=lambda path: int(path.name[1:]),
    )


@dataclass(frozen=True)
class JP10Reference:
    density_g_cm3: float
    heating_value_mj_kg: float
    melting_point_k: float
    #: where the values came from: the models of the run, or the documented defaults
    source: str = "run_models"

    def as_dict(self) -> dict[str, float]:
        return {
            "density_g_cm3": self.density_g_cm3,
            "heating_value_mj_kg": self.heating_value_mj_kg,
            "melting_point_k": self.melting_point_k,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "JP10Reference":
        defaults = DOCUMENTED_JP10_REFERENCE
        return cls(
            density_g_cm3=_number(values.get("density_g_cm3"), defaults["density_g_cm3"]),
            heating_value_mj_kg=_number(
                values.get("heating_value_mj_kg"), defaults["heating_value_mj_kg"]
            ),
            melting_point_k=_number(
                values.get("melting_point_k"), defaults["melting_point_k"]
            ),
            source=str(values.get("source") or "run_models"),
        )


def margins_of_row(
    row: Mapping[str, object], reference: JP10Reference
) -> dict[str, float]:
    """Distance to JP-10 of the three compared properties (positive = better)."""

    density = _number(row.get(MARGIN_SOURCE_COLUMNS["margin_density"]))
    heating = _number(row.get(MARGIN_SOURCE_COLUMNS["margin_heating_value"]))
    melting = _number(row.get(MARGIN_SOURCE_COLUMNS["margin_melting_point"]))
    return {
        "margin_density": density - reference.density_g_cm3,
        "margin_heating_value": heating - reference.heating_value_mj_kg,
        "margin_melting_point": reference.melting_point_k - melting,
    }


def beats_from_margins(margins: Mapping[str, float]) -> int:
    """How many compared properties are at or better than JP-10."""

    beats = 0
    for key in MARGIN_KEYS:
        value = margins.get(key)
        if value is None or not math.isfinite(float(value)):
            continue
        if float(value) >= 0.0:
            beats += 1
    return beats


def _descriptor_vector(row: Mapping[str, object]) -> dict[str, float]:
    return {key: _number(row.get(key)) for key in DESCRIPTOR_KEYS}


def _with_current_ring_count(
    row: Mapping[str, object],
    cache: dict[str, int],
) -> dict[str, object]:
    """Normalise pre-release rows that predate the full-molecule ring contract."""

    normalised = dict(row)
    if (
        normalised.get("ring_count_definition") == RING_COUNT_DEFINITION
        and normalised.get("ring_count") not in (None, "")
    ):
        return normalised
    smiles = str(row.get("canonical_smiles") or "").strip()
    if not smiles:
        return normalised
    if smiles not in cache:
        cache[smiles] = ring_count_from_smiles(smiles)
    normalised["ring_count"] = cache[smiles]
    normalised["ring_count_definition"] = RING_COUNT_DEFINITION
    return normalised


def build_features(
    parent: Mapping[str, object],
    child: Mapping[str, object],
    operation: str | None,
    reference: JP10Reference,
) -> dict[str, float]:
    """Model input for one perturbation step (parent -> child)."""

    weight = 1.0 / len(PERTURBATION_OPERATIONS) if operation is None else 1.0
    features: dict[str, float] = {
        "op_%s" % name: weight if operation is None or name == operation else 0.0
        for name in PERTURBATION_OPERATIONS
    }
    parent_descriptors = _descriptor_vector(parent)
    child_descriptors = _descriptor_vector(child)
    for key, value in parent_descriptors.items():
        features["parent_%s" % key] = value
        features["delta_%s" % key] = child_descriptors[key] - value
    parent_margins = margins_of_row(parent, reference)
    for key in MARGIN_KEYS:
        features["parent_%s" % key] = parent_margins[key]
    features["parent_beats"] = float(beats_from_margins(parent_margins))
    return features


def build_targets(
    parent: Mapping[str, object],
    child: Mapping[str, object],
    reference: JP10Reference,
) -> dict[str, float]:
    """Property change of one perturbation step: the trend the model learns."""

    parent_margins = margins_of_row(parent, reference)
    child_margins = margins_of_row(child, reference)
    return {
        "delta_%s" % key: child_margins[key] - parent_margins[key]
        for key in MARGIN_KEYS
    }


def _parent_index(run_root: Path) -> dict[str, dict[str, object]]:
    """Every scored molecule of the run folder, by canonical SMILES."""

    index: dict[str, dict[str, object]] = {}
    ring_count_cache: dict[str, int] = {}
    sources = [run_root / "perturbation_property_predictions.csv"]
    sources.extend(
        carbon_dir / "property_predictions.csv"
        for carbon_dir in _carbon_directories(run_root)
    )
    for source in sources:
        for row in _read_rows(source):
            smiles = str(row.get("canonical_smiles") or "").strip()
            if smiles and smiles not in index:
                index[smiles] = _with_current_ring_count(row, ring_count_cache)
    # Products that never reached property prediction still carry descriptors.
    for row in _read_rows(run_root / "perturbed_candidates.csv"):
        smiles = str(row.get("canonical_smiles") or "").strip()
        if smiles and smiles not in index:
            index[smiles] = _with_current_ring_count(row, ring_count_cache)
    return index


def parent_index(
    run_root: str | Path,
    extra_rows: Iterable[Mapping[str, object]] = (),
) -> dict[str, dict[str, object]]:
    """Property record of every possible parent of a perturbation step.

    The run folder supplies the products of earlier rounds and the generation
    library; ``extra_rows`` adds the current round's seeds, so a round can score
    its own candidates even on a folder that has no stored products yet.
    """

    index: dict[str, dict[str, object]] = {}
    ring_count_cache: dict[str, int] = {}
    for row in extra_rows:
        smiles = str(row.get("canonical_smiles") or "").strip()
        if smiles:
            index[smiles] = _with_current_ring_count(row, ring_count_cache)
    for smiles, row in _parent_index(Path(run_root)).items():
        index.setdefault(smiles, row)
    return index


def build_iteration_dataset(
    run_root: str | Path,
    *,
    reference: JP10Reference | None = None,
    include_anchors: bool = True,
    anchor_weight: float = 0.15,
) -> dict[str, object]:
    """Every predicted molecule of a run folder, as a training set.

    Two kinds of sample are written:

    * ``pair`` - one perturbation step (parent -> product, weight 1.0). These are
      the samples that teach the property-change trend of the operators.
    * ``anchor`` - a molecule of the generation library with itself as child, i.e.
      "no structural change, no property change" (low weight). They add the
      absolute property records of the generation stage, so the model sees every
      molecule the run predicted, not just the products.
    """

    root = Path(run_root)
    reference = reference or reference_from_run(root)
    parents = _parent_index(root)
    ring_count_cache: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    skipped_missing_parent = 0
    skipped_missing_property = 0
    for child in _read_rows(root / "perturbation_property_predictions.csv"):
        child = _with_current_ring_count(child, ring_count_cache)
        parent_smiles = str(child.get("parent_canonical_smiles") or "").strip()
        raw_parent = parents.get(parent_smiles)
        parent = (
            _with_current_ring_count(raw_parent, ring_count_cache)
            if raw_parent is not None
            else None
        )
        if parent is None:
            skipped_missing_parent += 1
            continue
        if not all(
            math.isfinite(_number(parent.get(column)))
            and math.isfinite(_number(child.get(column)))
            for column in MARGIN_SOURCE_COLUMNS.values()
        ):
            skipped_missing_property += 1
            continue
        operation = str(child.get("operation") or "").strip()
        if operation not in PERTURBATION_OPERATIONS:
            continue
        rows.append(
            {
                "smiles": str(child.get("canonical_smiles") or ""),
                "parent_smiles": parent_smiles,
                "operation": operation,
                "sample_kind": "pair",
                "weight": 1.0,
                **build_features(parent, child, operation, reference),
                **build_targets(parent, child, reference),
            }
        )
    anchor_count = 0
    if include_anchors:
        known_children = {str(row["smiles"]) for row in rows}
        for carbon_dir in _carbon_directories(root):
            for molecule in _read_rows(carbon_dir / "property_predictions.csv"):
                molecule = _with_current_ring_count(molecule, ring_count_cache)
                smiles = str(molecule.get("canonical_smiles") or "").strip()
                if not smiles or smiles in known_children:
                    continue
                if not all(
                    math.isfinite(_number(molecule.get(column)))
                    for column in MARGIN_SOURCE_COLUMNS.values()
                ):
                    continue
                known_children.add(smiles)
                rows.append(
                    {
                        "smiles": smiles,
                        "parent_smiles": smiles,
                        "operation": "",
                        "sample_kind": "anchor",
                        "weight": float(anchor_weight),
                        **build_features(molecule, molecule, None, reference),
                        **build_targets(molecule, molecule, reference),
                    }
                )
                anchor_count += 1
    return {
        "rows": rows,
        "sample_count": len(rows),
        "pair_count": len(rows) - anchor_count,
        "anchor_count": anchor_count,
        "anchor_weight": float(anchor_weight),
        "skipped_missing_parent": skipped_missing_parent,
        "skipped_missing_property": skipped_missing_property,
        "parents_known": len(parents),
        "reference": reference.as_dict(),
        "reference_source": reference.source,
        "feature_columns": feature_columns(),
    }


def train_iteration_model(
    dataset: Sequence[Mapping[str, object]],
    model_path: str | Path,
    *,
    reference: JP10Reference,
    model_family: str = DEFAULT_MODEL_FAMILY,
    random_state: int = 16,
    metrics_path: str | Path | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fit one gradient-boosted regressor per compared property."""

    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    if len(dataset) < 32:
        raise ValueError(
            "not enough parent/product pairs to train the iteration model: %d"
            % len(dataset)
        )
    factory = candidate_model_factories().get(model_family)
    if factory is None:
        raise ValueError(
            "unknown iteration model family: %s (available: %s)"
            % (model_family, ", ".join(sorted(candidate_model_factories())))
        )
    columns = feature_columns()
    features = np.asarray(
        [[_number(row.get(column)) for column in columns] for row in dataset],
        dtype=float,
    )
    weights = np.asarray(
        [_number(row.get("weight"), 1.0) for row in dataset], dtype=float
    )
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
        weights = None
    weights = np.asarray(
        [_number(row.get("weight"), 1.0) for row in dataset], dtype=float
    )
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
        weights = None
    anchor_mask = np.asarray(
        [str(row.get("sample_kind") or "pair") == "anchor" for row in dataset],
        dtype=bool,
    )
    # A feature that is missing everywhere carries no information (and a
    # gradient-boosted tree cannot bin an empty column), so it is dropped from
    # the artifact and from every later scoring call.
    usable = [index for index, column in enumerate(columns) if np.isfinite(features[:, index]).any()]
    dropped = [column for index, column in enumerate(columns) if index not in usable]
    columns = [columns[index] for index in usable]
    features = features[:, usable]
    models: dict[str, object] = {}
    metrics: dict[str, object] = {}
    for key in MARGIN_KEYS:
        target_key = "delta_%s" % key
        targets = np.asarray(
            [_number(row.get(target_key)) for row in dataset], dtype=float
        )
        train_index, test_index = train_test_split(
            np.arange(len(dataset)), test_size=0.2, random_state=random_state
        )
        train_x, test_x = features[train_index], features[test_index]
        train_y, test_y = targets[train_index], targets[test_index]
        model = factory(random_state)
        samples_used = _fit_candidate(
            model,
            train_x,
            train_y,
            None if weights is None else weights[train_index],
            anchor_mask[train_index],
        )
        predictions = model.predict(test_x)
        sign_agreement = (
            float(np.mean(np.sign(predictions) == np.sign(test_y)))
            if test_y.size
            else 0.0
        )
        metrics[target_key] = {
            "samples_used": samples_used,
            "test_mae": float(mean_absolute_error(test_y, predictions)),
            # The model is only useful when it beats "the property does not move".
            "baseline_mae_no_change": float(
                mean_absolute_error(test_y, np.zeros_like(test_y))
            ),
            "test_sign_agreement": sign_agreement,
            "train_rows": int(train_y.size),
            "test_rows": int(test_y.size),
        }
        models[key] = model
    artifact = {
        "model": models,
        "model_family": model_family,
        "feature_columns": columns,
        "dropped_features": dropped,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "ring_count_definition": RING_COUNT_DEFINITION,
        "targets": ["delta_%s" % key for key in MARGIN_KEYS],
        "jp10_reference": reference.as_dict(),
        #: static sampling guidance of the operator mix (never re-fitted by a round)
        "operator_weights": operator_gain_weights(dataset, reference=reference),
        "operator_ring_gain": operator_ring_gains(dataset),
        "metrics": metrics,
        "metadata": dict(metadata or {}),
    }
    target_path = Path(model_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, target_path)
    summary = {
        "model_path": str(target_path),
        "sample_count": len(dataset),
        "feature_count": len(columns),
        "dropped_features": dropped,
        "metrics": metrics,
        "jp10_reference": reference.as_dict(),
        "operator_weights": artifact["operator_weights"],
        "operator_ring_gain": artifact["operator_ring_gain"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "ring_count_definition": RING_COUNT_DEFINITION,
    }
    if metrics_path is not None:
        metrics_target = Path(metrics_path)
        metrics_target.parent.mkdir(parents=True, exist_ok=True)
        metrics_target.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def load_iteration_model(path: str | Path) -> dict[str, object]:
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("%s is not a perturbation iteration model" % path)
    if artifact.get("ring_count_definition") != RING_COUNT_DEFINITION:
        raise ValueError(
            "%s predates the full-molecule ring-count contract; retrain the "
            "perturbation iteration model before using it" % path
        )
    return artifact


def iteration_model_contract_is_current(path: str | Path) -> bool:
    """Return whether an artifact uses the v1 full-molecule ring contract."""

    model_path = Path(path)
    if not model_path.exists():
        return False
    try:
        artifact = joblib.load(model_path)
    except Exception:
        return False
    return bool(
        isinstance(artifact, dict)
        and "model" in artifact
        and artifact.get("ring_count_definition") == RING_COUNT_DEFINITION
    )


def _beat_prediction_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float]:
    """How well a candidate model answers the question the loop actually asks.

    The loop ranks products by the beat count it expects them to reach, so what
    matters is not only the numeric error: does the model get the count right,
    and when it promises Tier 1, is it right?
    """

    predicted_beats = np.asarray(
        [beats_from_margins({key: float(value) for key, value in zip(MARGIN_KEYS, row)})
         for row in predicted],
        dtype=float,
    )
    actual_beats = np.asarray(
        [beats_from_margins({key: float(value) for key, value in zip(MARGIN_KEYS, row)})
         for row in actual],
        dtype=float,
    )
    promised = predicted_beats >= 3.0
    real = actual_beats >= 3.0
    precision = float(np.sum(promised & real) / np.sum(promised)) if promised.any() else 0.0
    recall = float(np.sum(promised & real) / np.sum(real)) if real.any() else 0.0
    return {
        "beat_exact_accuracy": float(np.mean(predicted_beats == actual_beats)),
        "beats_mae": float(np.mean(np.abs(predicted_beats - actual_beats))),
        "tier1_precision": precision,
        "tier1_recall": recall,
        "predicted_tier1_rows": int(np.sum(promised)),
        "actual_tier1_rows": int(np.sum(real)),
    }


def evaluate_iteration_models(
    dataset: Sequence[Mapping[str, object]],
    *,
    reference: JP10Reference,
    folds: int = 5,
    random_state: int = 16,
    model_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Cross-validate every candidate family on the property-change task.

    One fold-split is used for all families, so the numbers are comparable. Each
    family is scored on the three property deltas (MAE against the "the property
    does not move" baseline) and on the beat-count question the iteration loop
    uses to pick candidates.
    """

    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error

    if len(dataset) < 64:
        raise ValueError(
            "not enough parent/product pairs to compare models: %d" % len(dataset)
        )
    columns = feature_columns()
    features = np.asarray(
        [[_number(row.get(column)) for column in columns] for row in dataset],
        dtype=float,
    )
    usable = [
        index
        for index, column in enumerate(columns)
        if np.isfinite(features[:, index]).any()
    ]
    columns = [columns[index] for index in usable]
    features = features[:, usable]
    parent_margins = np.asarray(
        [
            [
                _number(row.get("parent_margin_density")),
                _number(row.get("parent_margin_heating_value")),
                _number(row.get("parent_margin_melting_point")),
            ]
            for row in dataset
        ],
        dtype=float,
    )
    targets = {
        key: np.asarray(
            [_number(row.get("delta_%s" % key)) for row in dataset], dtype=float
        )
        for key in MARGIN_KEYS
    }
    weights = np.asarray(
        [_number(row.get("weight"), 1.0) for row in dataset], dtype=float
    )
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
        weights = None
    anchor_mask = np.asarray(
        [str(row.get("sample_kind") or "pair") == "anchor" for row in dataset],
        dtype=bool,
    )
    families = candidate_model_factories()
    names = list(model_names or families)
    folds = max(int(folds), 2)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = list(splitter.split(features))
    results: list[dict[str, object]] = []
    for name in names:
        factory = families.get(name)
        if factory is None:
            continue
        predicted_deltas = {key: np.zeros(len(dataset)) for key in MARGIN_KEYS}
        fit_seconds = 0.0
        predict_seconds = 0.0
        for train_index, test_index in splits:
            for key in MARGIN_KEYS:
                model = factory(random_state)
                start = time.perf_counter()
                _fit_candidate(
                    model,
                    features[train_index],
                    targets[key][train_index],
                    None if weights is None else weights[train_index],
                    anchor_mask[train_index],
                )
                fit_seconds += time.perf_counter() - start
                start = time.perf_counter()
                predicted_deltas[key][test_index] = model.predict(features[test_index])
                predict_seconds += time.perf_counter() - start
        metrics: dict[str, object] = {
            "model_family": name,
            "fit_seconds": round(fit_seconds, 2),
            "predict_seconds": round(predict_seconds, 2),
            "properties": {},
        }
        gains: list[float] = []
        signs: list[float] = []
        for key in MARGIN_KEYS:
            actual = targets[key]
            prediction = predicted_deltas[key]
            mae = float(mean_absolute_error(actual, prediction))
            baseline = float(mean_absolute_error(actual, np.zeros_like(actual)))
            gain = 1.0 - mae / baseline if baseline > 0 else 0.0
            sign = float(np.mean(np.sign(prediction) == np.sign(actual)))
            gains.append(gain)
            signs.append(sign)
            metrics["properties"][key] = {  # type: ignore[index]
                "test_mae": mae,
                "baseline_mae_no_change": baseline,
                "relative_gain": gain,
                "sign_agreement": sign,
            }
        predicted_matrix = np.column_stack(
            [parent_margins[:, index] + predicted_deltas[key] for index, key in enumerate(MARGIN_KEYS)]
        )
        actual_matrix = np.column_stack(
            [parent_margins[:, index] + targets[key] for index, key in enumerate(MARGIN_KEYS)]
        )
        beat_metrics = _beat_prediction_metrics(predicted_matrix, actual_matrix)
        metrics.update(beat_metrics)
        mean_gain = float(np.mean(gains))
        metrics["mean_relative_gain"] = mean_gain
        metrics["mean_sign_agreement"] = float(np.mean(signs))
        # Transparent ranking: how much of the property trend it captures, how
        # often it gets the beat count right, and whether its Tier 1 calls hold.
        metrics["rank_score"] = (
            0.45 * mean_gain
            + 0.35 * float(beat_metrics["beat_exact_accuracy"])
            + 0.20 * float(beat_metrics["tier1_precision"])
        )
        results.append(metrics)
    results.sort(key=lambda row: -float(row["rank_score"]))
    peak_mb = None
    try:  # Windows reports peak working set, Linux the peak RSS, both in bytes.
        import resource

        peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0), 1)
    except Exception:
        peak_mb = None
    return {
        "sample_count": len(dataset),
        "feature_count": len(columns),
        "folds": folds,
        "random_state": random_state,
        "peak_memory_mb": peak_mb,
        "reference": reference.as_dict(),
        "ranking_rule": (
            "0.45 * mean relative MAE gain + 0.35 * beat-count accuracy + "
            "0.20 * Tier 1 precision"
        ),
        "models": results,
    }


def operator_gain_weights(
    dataset: Sequence[Mapping[str, object]],
    *,
    reference: JP10Reference,
    # A small floor keeps every operator reachable while leaving most of the
    # budget with the operators the data says move towards JP-10-like skeletons.
    floor_fraction: float = 0.2,
) -> dict[str, float]:
    """Average property gain each operator produced, as sampling weights.

    The model is static, so the operator mix of a round is steered by what the
    training data itself showed: operators that moved the three margins towards
    JP-10 get a larger share of the round's attempts, the rest keeps a uniform
    floor so no operator is switched off.
    """

    scale = {
        "margin_density": max(abs(reference.density_g_cm3), 1e-6),
        "margin_heating_value": max(abs(reference.heating_value_mj_kg), 1e-6),
        "margin_melting_point": max(abs(reference.melting_point_k), 1e-6),
    }
    totals: dict[str, float] = {operation: 0.0 for operation in PERTURBATION_OPERATIONS}
    counts: dict[str, int] = {operation: 0 for operation in PERTURBATION_OPERATIONS}
    ring_gains: dict[str, float] = {operation: 0.0 for operation in PERTURBATION_OPERATIONS}
    for row in dataset:
        operation = str(row.get("operation") or "")
        if operation not in totals:
            continue
        gain = 0.0
        for key in MARGIN_KEYS:
            value = _number(row.get("delta_%s" % key), 0.0)
            gain += value / scale[key]
        # Direction towards JP-10-like skeletons: operators that build five/six
        # membered rings and take strained three/four membered rings out are worth
        # more attempts than ones that just pile on cyclopropanes.
        parent_strained, parent_preferred = ring_profile(
            str(row.get("parent_smiles") or row.get("smiles") or "")
        )
        child_strained, child_preferred = ring_profile(str(row.get("smiles") or ""))
        ring_gain = (child_preferred - parent_preferred) - (
            child_strained - parent_strained
        )
        ring_gains[operation] += ring_gain
        gain += OPERATOR_RING_DIRECTION_GAIN * ring_gain / max(
            RING_STRAIN_REFERENCE, 1.0
        )
        totals[operation] += gain
        counts[operation] += 1
    means = {
        operation: (totals[operation] / counts[operation]) if counts[operation] else 0.0
        for operation in PERTURBATION_OPERATIONS
    }
    floor = min(max(float(floor_fraction), 0.0), 1.0)
    uniform = 1.0 / len(PERTURBATION_OPERATIONS)
    values = np.asarray(
        [means[operation] for operation in PERTURBATION_OPERATIONS], dtype=float
    )
    # Operators are compared with each other, not with zero: a family where every
    # operator loses some margin still has better and worse ways to move.
    # Scale-invariant softmax: the gains are standardised first, so adding a new
    # direction (the ring preference) sharpens the choice instead of flattening it.
    spread = float(np.std(values))
    standardised = (values - float(np.mean(values))) / max(spread, 1e-9)
    shifted = standardised - float(np.max(standardised))
    weights = np.exp(shifted)
    weights = weights / float(np.sum(weights))
    blended = {
        operation: floor * uniform
        + (1.0 - floor) * float(weights[index])
        for index, operation in enumerate(PERTURBATION_OPERATIONS)
    }
    # An operator with no training pairs (a brand-new one such as ring expansion)
    # starts at the level of the best known operator; the rounds then adapt it
    # from what it actually produces.
    best = max(blended.values()) if blended else 1.0
    for operation in PERTURBATION_OPERATIONS:
        if not counts.get(operation):
            blended[operation] = best
    total = sum(blended.values()) or 1.0
    return {op: blended[op] / total for op in PERTURBATION_OPERATIONS}


def attempts_per_operation(
    weights: Mapping[str, float],
    *,
    attempts_per_seed_operation: int = 0,
    total_attempts_per_molecule: int | None = None,
    minimum_attempts: int = 0,
) -> dict[str, int]:
    """Turn sampling weights into an integer attempt count per operator.

    ``total_attempts_per_molecule`` gives every seed exactly that many
    perturbations in the round, distributed over the operators by weight (an
    operator may end up with none). Without it the total is
    ``len(operations) * attempts_per_seed_operation``, the per-operator budget of
    a plain campaign.
    """

    total = (
        int(total_attempts_per_molecule)
        if total_attempts_per_molecule is not None
        else int(attempts_per_seed_operation) * len(PERTURBATION_OPERATIONS)
    )
    total = max(total, 0)
    positive = {
        operation: max(float(weights.get(operation, 0.0)), 0.0)
        for operation in PERTURBATION_OPERATIONS
    }
    share_total = sum(positive.values()) or 1.0
    raw = {
        operation: total * positive[operation] / share_total
        for operation in PERTURBATION_OPERATIONS
    }
    counts = {operation: 0 for operation in PERTURBATION_OPERATIONS}
    remaining = total
    # Largest fractional parts first, so a small budget lands on the operators the
    # training data favoured and the total is exactly the requested one.
    # The leftover attempts go to the *most promising* operators, not to whoever
    # happens to have the largest rounding remainder.
    order = sorted(
        PERTURBATION_OPERATIONS,
        key=lambda operation: (-positive[operation], operation),
    )
    for operation in PERTURBATION_OPERATIONS:
        whole = int(raw[operation])
        if whole and remaining > 0:
            counts[operation] = min(whole, remaining)
            remaining -= counts[operation]
    index = 0
    while remaining > 0 and order:
        counts[order[index % len(order)]] += 1
        remaining -= 1
        index += 1
    if minimum_attempts > 0:
        for operation in PERTURBATION_OPERATIONS:
            if counts[operation] < int(minimum_attempts):
                counts[operation] = int(minimum_attempts)
    return counts


def reference_from_run(run_root: str | Path) -> JP10Reference:
    """JP-10 reference of a run folder, from the models its rows were made with."""

    root = Path(run_root)
    for row in _read_rows(root / "perturbation_property_predictions.csv"):
        density_model = str(
            row.get("sta_density_model_path")
            or row.get("sta_density_model")
            or row.get("sta_property_model_checkpoint")
            or ""
        ).strip()
        enthalpy_model = str(
            row.get("sta_heating_value_model_path")
            or row.get("sta_formation_enthalpy_model")
            or row.get("sta_formation_enthalpy_model_path")
            or row.get("sta_property_model_checkpoint")
            or ""
        ).strip()
        melting_model = str(row.get("external_melting_point_model_path") or "").strip()
        if not (density_model and enthalpy_model and melting_model):
            continue
        try:
            from tier_routing import benchmark_from_models

            benchmark = benchmark_from_models(
                sta_density_model_path=density_model,
                sta_formation_enthalpy_model_path=enthalpy_model,
                phase_fusion_melting_point_model_path=melting_model,
            )
        except Exception:
            continue
        values = {
            "density_g_cm3": float(benchmark.density),
            "heating_value_mj_kg": float(benchmark.mass_heating_value),
            "melting_point_k": float(benchmark.melting_point),
        }
        if all(math.isfinite(value) for value in values.values()):
            return JP10Reference.from_mapping({**values, "source": "run_models"})
    return JP10Reference.from_mapping(
        {**DOCUMENTED_JP10_REFERENCE, "source": "documented_defaults"}
    )


def select_iteration_candidates(
    scored: Sequence[Mapping[str, object]],
    *,
    budget: int,
    exploration_fraction: float = 0.15,
    rng: random.Random | None = None,
) -> dict[str, object]:
    """Keep the most promising candidates plus an exploration share.

    Mirrors the exploration rule of the generation line: most of the budget goes
    to the best-predicted products, the rest is sampled at random so a round can
    still discover a move the model does not expect.
    """

    rng = rng or random.Random(0)
    budget = max(int(budget), 0)
    if budget <= 0 or not scored:
        return {"selected": [], "selected_smiles": [], "explored": 0, "ranked": 0}
    ranked = sorted(
        scored, key=lambda row: (-float(row["promise"]), str(row["canonical_smiles"]))
    )
    exploration_fraction = min(max(float(exploration_fraction), 0.0), 0.95)
    explore_count = int(round(budget * exploration_fraction))
    exploit_count = max(budget - explore_count, 0)
    exploit = ranked[:exploit_count]
    remaining = ranked[exploit_count:]
    explore = (
        rng.sample(remaining, min(explore_count, len(remaining))) if remaining else []
    )
    selected = exploit + explore
    return {
        "selected": selected,
        "selected_smiles": [str(row["canonical_smiles"]) for row in selected],
        "explored": len(explore),
        "ranked": len(ranked),
    }
