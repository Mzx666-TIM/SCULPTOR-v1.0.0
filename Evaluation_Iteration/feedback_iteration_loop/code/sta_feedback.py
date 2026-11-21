from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from property_consensus import synthetic_accessibility_property_score
from property_screening import FuelPropertyThresholds, screening_status
from thesis_property_tools import mass_heating_value_from_formation_enthalpy


STA_FEEDBACK_COLUMNS = [
    "candidate_id",
    "canonical_smiles",
    "pipeline_route",
    "screening_status",
    "fingerprint_novelty",
    "high_novelty",
    "temporary_smiles_status",
    "label_source",
    "label_kind",
    "label_weight",
    "sta_weak_feedback_reward",
    "sta_feedback_use",
    "sta_density_g_cm3",
    "sta_density_g_cm3_uncertainty",
    "sta_formation_enthalpy_kj_mol",
    "sta_formation_enthalpy_kj_mol_uncertainty",
    "sta_derived_heating_value_mj_kg",
    "sta_derived_heating_value_uncertainty_mj_kg",
    "sta_melting_point_k",
    "sta_melting_point_k_uncertainty",
    "sta_molar_volume_cm3_mol",
    "sta_molar_volume_cm3_mol_uncertainty",
    "sta_flash_point_k",
    "sta_flash_point_k_uncertainty",
    "sta_isp",
    "sta_isp_uncertainty",
    "density_component",
    "heating_value_component",
    "sa_component",
    "low_novelty_reliability_component",
    "density_pass_by_sta",
    "heating_value_pass_by_sta",
    "sta_property_prediction_status",
    "sta_property_failure_reason",
]


@dataclass(frozen=True)
class StaWeakFeedbackExportSummary:
    input_path: str
    output_path: str
    rows_read: int
    rows_written: int
    novelty_threshold: float
    base_label_weight: float


def build_sta_weak_feedback_rows(
    rows: list[dict[str, object]],
    thresholds: FuelPropertyThresholds = FuelPropertyThresholds(),
    novelty_threshold: float = 0.75,
    base_label_weight: float = 0.35,
    water_phase: str = "gas",
) -> list[dict[str, object]]:
    """Build weak iteration-feedback labels from low-novelty STA predictions.

    These rows are deliberately weaker than the finalized property feedback: they
    are useful as iteration rewards, but they should not be mixed with
    high-confidence labels.
    """

    feedback_rows: list[dict[str, object]] = []
    for row in rows:
        if not _is_low_novelty_new_candidate(row, novelty_threshold):
            continue
        if not _has_sta_prediction(row):
            continue

        smiles = str(row.get("canonical_smiles") or "").strip()
        novelty = _optional_float(row.get("fingerprint_novelty")) or 0.0
        reliability = _low_novelty_reliability(novelty, novelty_threshold)
        density = _optional_float(row.get("sta_density_g_cm3"))
        formation = _optional_float(row.get("sta_formation_enthalpy_kj_mol"))
        formation_uncertainty = _optional_float(
            row.get("sta_formation_enthalpy_kj_mol_uncertainty")
        )
        heating_value = None
        heating_uncertainty = None
        if formation is not None:
            try:
                heating = mass_heating_value_from_formation_enthalpy(
                    smiles,
                    formation,
                    water_phase=water_phase,
                )
                heating_value = heating.heating_value_mj_kg
                if formation_uncertainty is not None:
                    heating_uncertainty = round(
                        formation_uncertainty / heating.molecular_weight_g_mol,
                        6,
                    )
            except Exception:
                heating_value = None
                heating_uncertainty = None

        density_component = _minimum_score(
            density,
            thresholds.density_min_g_cm3,
        )
        heating_component = _minimum_score(
            heating_value,
            thresholds.heating_value_min_mj_kg,
        )
        sa_component = _sa_score_component(row.get("sa_score"))
        reward = (
            0.45 * density_component
            + 0.35 * heating_component
            + 0.10 * sa_component
            + 0.10 * reliability
        )
        effective_weight = base_label_weight * (0.5 + 0.5 * reliability)

        feedback_rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "canonical_smiles": smiles,
                "pipeline_route": row.get("pipeline_route", ""),
                "screening_status": screening_status(row.get("hard_threshold_verdict")),
                "fingerprint_novelty": novelty,
                "high_novelty": row.get("high_novelty", ""),
                "temporary_smiles_status": row.get("temporary_smiles_status", ""),
                "label_source": "sta_prediction",
                "label_kind": "weak_iteration_feedback",
                "label_weight": round(effective_weight, 6),
                "sta_weak_feedback_reward": round(reward, 6),
                "sta_feedback_use": "iteration_reward_only",
                "sta_density_g_cm3": _blank_none(density),
                "sta_density_g_cm3_uncertainty": _blank_none(
                    _optional_float(row.get("sta_density_g_cm3_uncertainty"))
                ),
                "sta_formation_enthalpy_kj_mol": _blank_none(formation),
                "sta_formation_enthalpy_kj_mol_uncertainty": _blank_none(
                    formation_uncertainty
                ),
                "sta_derived_heating_value_mj_kg": _blank_none(heating_value),
                "sta_derived_heating_value_uncertainty_mj_kg": _blank_none(
                    heating_uncertainty
                ),
                "sta_melting_point_k": _blank_none(
                    _optional_float(row.get("sta_melting_point_k"))
                ),
                "sta_melting_point_k_uncertainty": _blank_none(
                    _optional_float(row.get("sta_melting_point_k_uncertainty"))
                ),
                "sta_molar_volume_cm3_mol": _blank_none(
                    _optional_float(row.get("sta_molar_volume_cm3_mol"))
                ),
                "sta_molar_volume_cm3_mol_uncertainty": _blank_none(
                    _optional_float(row.get("sta_molar_volume_cm3_mol_uncertainty"))
                ),
                "sta_flash_point_k": _blank_none(
                    _optional_float(row.get("sta_flash_point_k"))
                ),
                "sta_flash_point_k_uncertainty": _blank_none(
                    _optional_float(row.get("sta_flash_point_k_uncertainty"))
                ),
                "sta_isp": _blank_none(_optional_float(row.get("sta_isp"))),
                "sta_isp_uncertainty": _blank_none(
                    _optional_float(row.get("sta_isp_uncertainty"))
                ),
                "density_component": round(density_component, 6),
                "heating_value_component": round(heating_component, 6),
                "sa_component": round(sa_component, 6),
                "low_novelty_reliability_component": round(reliability, 6),
                "density_pass_by_sta": (
                    "" if density is None else density >= thresholds.density_min_g_cm3
                ),
                "heating_value_pass_by_sta": (
                    ""
                    if heating_value is None
                    else heating_value >= thresholds.heating_value_min_mj_kg
                ),
                "sta_property_prediction_status": row.get(
                    "sta_property_prediction_status",
                    "",
                ),
                "sta_property_failure_reason": row.get(
                    "sta_property_failure_reason",
                    "",
                ),
            }
        )
    return feedback_rows


def export_sta_weak_feedback_csv(
    input_path: str | Path,
    output_path: str | Path,
    thresholds: FuelPropertyThresholds = FuelPropertyThresholds(),
    novelty_threshold: float = 0.75,
    base_label_weight: float = 0.35,
    water_phase: str = "gas",
) -> StaWeakFeedbackExportSummary:
    input_file = Path(input_path)
    with input_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    feedback_rows = build_sta_weak_feedback_rows(
        rows,
        thresholds=thresholds,
        novelty_threshold=novelty_threshold,
        base_label_weight=base_label_weight,
        water_phase=water_phase,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STA_FEEDBACK_COLUMNS)
        writer.writeheader()
        writer.writerows(feedback_rows)
    return StaWeakFeedbackExportSummary(
        input_path=str(input_file),
        output_path=str(output_file),
        rows_read=len(rows),
        rows_written=len(feedback_rows),
        novelty_threshold=novelty_threshold,
        base_label_weight=base_label_weight,
    )


def sta_feedback_summary_to_dict(
    summary: StaWeakFeedbackExportSummary,
) -> dict[str, object]:
    return asdict(summary)


def _is_low_novelty_new_candidate(
    row: dict[str, object],
    novelty_threshold: float,
) -> bool:
    if _truthy(row.get("high_novelty")):
        return False
    novelty = _optional_float(row.get("fingerprint_novelty"))
    if novelty is None or novelty >= novelty_threshold:
        return False
    if str(row.get("temporary_smiles_status") or "").strip() not in (
        "",
        "accepted",
    ):
        return False
    route = str(row.get("pipeline_route") or "").strip()
    if route and route != "exact_new_fingerprint_low":
        return False
    return True


def _has_sta_prediction(row: dict[str, object]) -> bool:
    return any(
        _optional_float(row.get(column)) is not None
        for column in (
            "sta_density_g_cm3",
            "sta_formation_enthalpy_kj_mol",
            "sta_melting_point_k",
            "sta_molar_volume_cm3_mol",
            "sta_flash_point_k",
            "sta_isp",
        )
    )


def _minimum_score(value: float | None, threshold: float) -> float:
    if value is None or threshold <= 0:
        return 0.0
    return _clamp(value / threshold, 0.0, 1.0)


def _sa_score_component(value: object) -> float:
    sa = _optional_float(value)
    if sa is None:
        return 0.0
    return synthetic_accessibility_property_score(sa)


def _low_novelty_reliability(novelty: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0
    return _clamp(1.0 - novelty / threshold, 0.0, 1.0)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _blank_none(value: float | None) -> float | str:
    return "" if value is None else round(float(value), 6)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
