from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from property_screening import (
    FuelPropertyThresholds,
)


def _panel_stage(stage: str, message: str) -> None:
    print(f"__FUEL_PANEL_STAGE__:{stage}:{message}", flush=True)


@dataclass(frozen=True)
class RescoreSummary:
    total_rows: int
    rescored_rows: int
    skipped_rows: int
    output_path: str
    summary_path: str | None
    composite_score_weights: CompositeScoreWeights
    property_score_weights: PropertyScoreWeights
    thresholds: FuelPropertyThresholds


@dataclass(frozen=True)
class CompositeScoreWeights:
    property_quality: float = 0.65
    e3nn_geometry: float = 0.10
    e3nn_symmetry: float = 0.10
    synthetic_accessibility: float = 0.0
    novelty: float = 0.15

    def normalized(self) -> "CompositeScoreWeights":
        # The top-level SA share belongs to the property aggregate.
        raw_values = (
            self.property_quality,
            self.e3nn_geometry,
            self.e3nn_symmetry,
            self.synthetic_accessibility,
            self.novelty,
        )
        if any(value < 0.0 for value in raw_values):
            raise ValueError("composite score weights must be non-negative")
        property_quality = self.property_quality + self.synthetic_accessibility
        values = (
            property_quality,
            self.e3nn_geometry,
            self.e3nn_symmetry,
            self.novelty,
        )
        total = sum(values)
        if total <= 0.0:
            raise ValueError("at least one composite score weight must be positive")
        return CompositeScoreWeights(
            property_quality=property_quality / total,
            e3nn_geometry=self.e3nn_geometry / total,
            e3nn_symmetry=self.e3nn_symmetry / total,
            synthetic_accessibility=0.0,
            novelty=self.novelty / total,
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PropertyScoreWeights:
    density: float = 1.0
    heating_value: float = 1.0
    volumetric_heating_value: float = 0.0
    melting_point: float = 0.0
    boiling_point: float = 0.0
    flash_point: float = 0.0
    synthetic_accessibility: float = 1.0
    structure_validity: float = 1.0
    # Retained only so old saved commands and summaries remain readable.
    formation_enthalpy: float = 0.0
    molar_volume: float = 0.0
    isp: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PropertyTargetValues:
    density: float | None = None
    heating_value: float | None = None
    volumetric_heating_value: float | None = None
    melting_point: float | None = None
    boiling_point: float | None = None
    flash_point: float | None = None
    formation_enthalpy: float | None = None
    molar_volume: float | None = None
    isp: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def rescore_candidate_csv(
    input_path: str | Path,
    output_path: str | Path,
    summary_output_path: str | Path | None = None,
    density_column: str = "density_g_cm3",
    heating_value_column: str = "heating_value_mj_kg",
    melting_point_column: str = "melting_point_k",
    boiling_point_column: str = "external_boiling_point_k",
    flash_point_column: str = "external_flash_point_k",
    formation_enthalpy_column: str = "external_formation_enthalpy_kj_mol",
    molar_volume_column: str = "external_molar_volume_cm3_mol",
    isp_column: str = "external_isp",
    novelty_column: str = "fingerprint_novelty",
    e3nn_geometry_column: str = "e3nn_geometry_score",
    e3nn_symmetry_column: str = "symmetry_consistency",
    sa_score_column: str = "sa_score",
    thresholds: FuelPropertyThresholds = FuelPropertyThresholds(),
    score_weights: CompositeScoreWeights = CompositeScoreWeights(),
    property_score_weights: PropertyScoreWeights = PropertyScoreWeights(),
) -> RescoreSummary:
    source_path = Path(input_path)
    _panel_stage("score", "Reading tables and rescoring property quality")
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rescored_rows: list[dict[str, object]] = []
    skipped_rows = 0
    for row in rows:
        try:
            property_score = property_quality_score(
                density_g_cm3=_optional_float_value(row.get(density_column)),
                heating_value_mj_kg=_optional_float_value(row.get(heating_value_column)),
                melting_point_k=_optional_float_value(row.get(melting_point_column)),
                boiling_point_k=_optional_float_value(row.get(boiling_point_column)),
                flash_point_k=_optional_float_value(row.get(flash_point_column)),
                sa_score=_optional_float_value(row.get(sa_score_column)),
                formation_enthalpy_kj_mol=_optional_float_value(
                    row.get(formation_enthalpy_column)
                ),
                molar_volume_cm3_mol=_optional_float_value(row.get(molar_volume_column)),
                isp=_optional_float_value(row.get(isp_column)),
                thresholds=thresholds,
                weights=property_score_weights,
            )
            row = {
                **row,
                "property_quality_score": property_score,
                "composite_ranking_score": composite_ranking_score(
                    property_quality_score=property_score,
                    novelty_score=float(row.get(novelty_column) or 0.0),
                    geometry_score=_optional_float_value(row.get(e3nn_geometry_column)),
                    symmetry_score=_optional_float_value(row.get(e3nn_symmetry_column)),
                    sa_score=_optional_float_value(row.get(sa_score_column)),
                    weights=score_weights,
                ),
            }
            rescored_rows.append(row)
        except (TypeError, ValueError, KeyError):
            skipped_rows += 1

    rescored_rows.sort(
        key=lambda row: (
            float(row.get("composite_ranking_score") or 0.0),
            float(row.get("property_quality_score") or 0.0),
            float(row.get(novelty_column) or 0.0),
        ),
        reverse=True,
    )

    output = Path(output_path)
    _panel_stage("sort", "Sorting and writing rescored results")
    _write_dict_rows(output, rescored_rows)
    summary_path = Path(summary_output_path) if summary_output_path is not None else None
    summary = RescoreSummary(
        total_rows=len(rows),
        rescored_rows=len(rescored_rows),
        skipped_rows=skipped_rows,
        output_path=str(output),
        summary_path="" if summary_path is None else str(summary_path),
        composite_score_weights=score_weights.normalized(),
        property_score_weights=property_score_weights,
        thresholds=thresholds,
    )
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _panel_stage("done", "Rescore completed")
    return summary


def _optional_float_value(value: object) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def property_quality_score(
    density_g_cm3: float | None,
    heating_value_mj_kg: float | None,
    melting_point_k: float | None,
    volumetric_heating_value_mj_l: float | None = None,
    boiling_point_k: float | None = None,
    flash_point_k: float | None = None,
    sa_score: float | None = None,
    structure_validity_score: float | None = None,
    formation_enthalpy_kj_mol: float | None = None,
    molar_volume_cm3_mol: float | None = None,
    isp: float | None = None,
    thresholds: FuelPropertyThresholds = FuelPropertyThresholds(),
    weights: PropertyScoreWeights = PropertyScoreWeights(),
    relative_window: float = 0.15,
) -> float:
    """Score fuel properties continuously around the hard thresholds.

    A property exactly on its threshold scores 0.5. A value that is better than
    the threshold by ``relative_window`` or more scores 1.0; a value worse than
    the threshold by the same margin scores 0.0. Missing properties are omitted
    and the remaining property weights are renormalized.
    """

    components: list[tuple[float, float]] = []
    if density_g_cm3 is not None:
        components.append(
            (
                weights.density,
                _minimum_property_score(
                    density_g_cm3,
                    thresholds.density_min_g_cm3,
                    relative_window,
                ),
            )
        )
    if heating_value_mj_kg is not None:
        components.append(
            (
                weights.heating_value,
                _minimum_property_score(
                    heating_value_mj_kg,
                    thresholds.heating_value_min_mj_kg,
                    relative_window,
                ),
            )
        )
    if volumetric_heating_value_mj_l is not None:
        components.append(
            (
                weights.volumetric_heating_value,
                _minimum_property_score(
                    volumetric_heating_value_mj_l,
                    thresholds.volumetric_heating_value_min_mj_l,
                    relative_window,
                ),
            )
        )
    if melting_point_k is not None:
        components.append(
            (
                weights.melting_point,
                _maximum_property_score(
                    melting_point_k,
                    thresholds.melting_point_max_k,
                    relative_window,
                ),
            )
        )
    if boiling_point_k is not None:
        components.append(
            (
                weights.boiling_point,
                _minimum_property_score(
                    boiling_point_k,
                    thresholds.boiling_point_min_k,
                    relative_window,
                ),
            )
        )
    if flash_point_k is not None:
        components.append(
            (
                weights.flash_point,
                _minimum_property_score(
                    flash_point_k,
                    thresholds.flash_point_min_k,
                    relative_window,
                ),
            )
        )
    if sa_score is not None:
        components.append(
            (
                weights.synthetic_accessibility,
                synthetic_accessibility_property_score(sa_score),
            )
        )
    if structure_validity_score is not None:
        components.append(
            (
                weights.structure_validity,
                _clamp01(structure_validity_score),
            )
        )

    # These arguments remain accepted for old callers, but are reference outputs
    # and must never affect ranking or feedback.
    _ = (formation_enthalpy_kj_mol, molar_volume_cm3_mol, isp)

    positive_components = [
        (weight, score)
        for weight, score in components
        if weight > 0.0
    ]
    if not positive_components:
        return 0.5
    total_weight = sum(weight for weight, _ in positive_components)
    return round(
        sum(weight * score for weight, score in positive_components) / total_weight,
        6,
    )


def target_property_score(
    density_g_cm3: float | None,
    heating_value_mj_kg: float | None,
    melting_point_k: float | None,
    targets: PropertyTargetValues,
    volumetric_heating_value_mj_l: float | None = None,
    boiling_point_k: float | None = None,
    flash_point_k: float | None = None,
    sa_score: float | None = None,
    structure_validity_score: float | None = None,
    formation_enthalpy_kj_mol: float | None = None,
    molar_volume_cm3_mol: float | None = None,
    isp: float | None = None,
    weights: PropertyScoreWeights = PropertyScoreWeights(),
    relative_window: float = 0.15,
) -> float:
    """Score how close the predicted properties are to user-supplied targets.

    A property exactly at its target scores 1.0. The score decreases linearly to
    0.0 when the absolute deviation reaches ``relative_window`` of the target.
    Missing targets or predictions are omitted and remaining positive weights
    are renormalized.
    """

    components: list[tuple[float, float]] = []
    if targets.density is not None and density_g_cm3 is not None:
        components.append(
            (
                weights.density,
                _target_closeness_score(
                    density_g_cm3,
                    targets.density,
                    relative_window,
                ),
            )
        )
    if targets.heating_value is not None and heating_value_mj_kg is not None:
        components.append(
            (
                weights.heating_value,
                _target_closeness_score(
                    heating_value_mj_kg,
                    targets.heating_value,
                    relative_window,
                ),
            )
        )
    if (
        targets.volumetric_heating_value is not None
        and volumetric_heating_value_mj_l is not None
    ):
        components.append(
            (
                weights.volumetric_heating_value,
                _target_closeness_score(
                    volumetric_heating_value_mj_l,
                    targets.volumetric_heating_value,
                    relative_window,
                ),
            )
        )
    if targets.melting_point is not None and melting_point_k is not None:
        components.append(
            (
                weights.melting_point,
                _target_closeness_score(
                    melting_point_k,
                    targets.melting_point,
                    relative_window,
                ),
            )
        )
    if targets.boiling_point is not None and boiling_point_k is not None:
        components.append(
            (
                weights.boiling_point,
                _target_closeness_score(
                    boiling_point_k,
                    targets.boiling_point,
                    relative_window,
                ),
            )
        )
    if targets.flash_point is not None and flash_point_k is not None:
        components.append(
            (
                weights.flash_point,
                _target_closeness_score(
                    flash_point_k,
                    targets.flash_point,
                    relative_window,
                ),
            )
        )
    if sa_score is not None:
        components.append(
            (
                weights.synthetic_accessibility,
                synthetic_accessibility_property_score(sa_score),
            )
        )
    if structure_validity_score is not None:
        components.append(
            (
                weights.structure_validity,
                _clamp01(structure_validity_score),
            )
        )

    _ = (formation_enthalpy_kj_mol, molar_volume_cm3_mol, isp)

    positive_components = [
        (weight, score)
        for weight, score in components
        if weight > 0.0
    ]
    if not positive_components:
        return 0.5
    total_weight = sum(weight for weight, _ in positive_components)
    return round(
        sum(weight * score for weight, score in positive_components) / total_weight,
        6,
    )


def composite_ranking_score(
    property_quality_score: float,
    novelty_score: float,
    geometry_score: float | None,
    symmetry_score: float | None,
    sa_score: float | None,
    weights: CompositeScoreWeights = CompositeScoreWeights(),
) -> float:
    normalized_weights = weights.normalized()
    property_quality = min(max(property_quality_score, 0.0), 1.0)
    geometry = 0.5 if geometry_score is None else min(max(geometry_score, 0.0), 1.0)
    symmetry = 0.5 if symmetry_score is None else min(max(symmetry_score, 0.0), 1.0)
    novelty = min(max(novelty_score, 0.0), 1.0)
    _ = sa_score
    return round(
        normalized_weights.property_quality * property_quality
        + normalized_weights.e3nn_geometry * geometry
        + normalized_weights.e3nn_symmetry * symmetry
        + normalized_weights.novelty * novelty,
        6,
    )


def synthetic_accessibility_property_score(sa_score: float) -> float:
    """Map SA to [0, 1]: <=1.5 -> 1, 3.0 -> 0.5, >=4.5 -> 0."""

    value = float(sa_score)
    if not math.isfinite(value):
        raise ValueError("SA score must be finite")
    return round(_clamp01((4.5 - value) / 3.0), 6)


def _minimum_property_score(
    value: float,
    threshold: float,
    relative_window: float,
) -> float:
    scale = max(abs(threshold) * relative_window, 1e-9)
    return _clamp01(0.5 + 0.5 * ((value - threshold) / scale))


def _maximum_property_score(
    value: float,
    threshold: float,
    relative_window: float,
) -> float:
    scale = max(abs(threshold) * relative_window, 1e-9)
    return _clamp01(0.5 + 0.5 * ((threshold - value) / scale))


def _target_closeness_score(
    value: float,
    target: float,
    relative_window: float,
) -> float:
    scale = max(abs(target) * relative_window, 1e-9)
    return _clamp01(1.0 - abs(value - target) / scale)


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
