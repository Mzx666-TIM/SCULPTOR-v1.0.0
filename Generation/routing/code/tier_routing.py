"""Two-axis channel classification for fuel candidates (Tier 1-4, Potential 1-3).

Only the three hard-screened properties are compared with JP-10: density, mass
heating value, and melting point (lower melting point is better). Boiling point,
flash point and volumetric heating value stay reference-only outputs.

The first axis holds the molecules that meet **every** enabled hard threshold and
splits them by how many of the three compared properties beat JP-10:

* ``tier1`` - beats JP-10 on all three.
* ``tier2`` - beats it on two.
* ``tier3`` - beats it on one.
* ``tier4`` - meets every hard threshold but beats it on none.

The second axis holds the molecules that miss a hard threshold, and every channel
on it is named by *(thresholds missed, properties beating JP-10)*:

* ``potential1`` - misses one hard threshold and beats JP-10 on the other two.
* ``potential2`` - misses one hard threshold and beats it on one property.
* ``potential3`` - misses two hard thresholds and beats it on the one that is
  left, i.e. the molecule is below the thresholds yet still wins on one axis.

A molecule that misses a threshold and beats JP-10 on nothing is not pooled: no
part of it is better than the reference, so it stays in the property table with
no channel instead of entering a salvage pool.

Channels are a *pooling* device: the run writes one table per channel and the
user picks which molecules to send to high-level calculation.

The JP-10 reference must come from the *same* prediction pipeline as the
candidates (predicted values, not experimental), so the comparison is
precision-matched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Iterable, Mapping

TIER1 = "tier1"
TIER2 = "tier2"
TIER3 = "tier3"
TIER4 = "tier4"
TIER_ORDER = (TIER1, TIER2, TIER3, TIER4)
#: how many of the three compared properties each tier beats JP-10 on
TIER_BEATS = {TIER1: 3, TIER2: 2, TIER3: 1, TIER4: 0}
TIER_LABELS = {
    TIER1: "Tier 1 (beats JP-10 on all three)",
    TIER2: "Tier 2 (beats JP-10 on two of three)",
    TIER3: "Tier 3 (beats JP-10 on one of three)",
    TIER4: "Tier 4 (meets every hard threshold, beats JP-10 on none)",
}

#: salvage pools for molecules that miss at least one hard threshold, split the
#: by how many thresholds are missed *and* how many properties beat JP-10
POTENTIAL1 = "potential1"
POTENTIAL2 = "potential2"
POTENTIAL3 = "potential3"
POTENTIAL_ORDER = (POTENTIAL1, POTENTIAL2, POTENTIAL3)
#: channel -> (hard thresholds missed, compared properties beating JP-10)
POTENTIAL_PROFILE = {
    POTENTIAL1: (1, 2),
    POTENTIAL2: (1, 1),
    POTENTIAL3: (2, 1),
}
POTENTIAL_BEATS = {
    channel: beats for channel, (_, beats) in POTENTIAL_PROFILE.items()
}
POTENTIAL_MISSED = {
    channel: missed for channel, (missed, _) in POTENTIAL_PROFILE.items()
}
POTENTIAL_LABELS = {
    POTENTIAL1: (
        "Potential 1 (misses one hard threshold, beats JP-10 on the other two)"
    ),
    POTENTIAL2: (
        "Potential 2 (misses one hard threshold, beats JP-10 on one property)"
    ),
    POTENTIAL3: (
        "Potential 3 (misses two hard thresholds, beats JP-10 on the remaining one)"
    ),
}
POTENTIAL_FILENAMES = {
    POTENTIAL1: "potential1_candidates.csv",
    POTENTIAL2: "potential2_candidates.csv",
    POTENTIAL3: "potential3_candidates.csv",
}
#: every channel a scored molecule can land in, strongest first
CHANNEL_ORDER = TIER_ORDER + POTENTIAL_ORDER
CHANNEL_LABELS = {**TIER_LABELS, **POTENTIAL_LABELS}
#: label of a scored molecule that matches no pool (misses thresholds and beats
#: JP-10 on nothing, so there is nothing to salvage)
NO_CHANNEL_LABEL = "No channel (no property better than JP-10)"
CHANNEL_FILENAMES = {
    **{tier: f"{tier}_candidates.csv" for tier in TIER_ORDER},
    **POTENTIAL_FILENAMES,
}
_PASS_BY_BEATS = {3: TIER1, 2: TIER2, 1: TIER3, 0: TIER4}
#: (missed thresholds, properties beating JP-10) -> potential channel. Profiles
#: outside this table get no channel at all.
_FAIL_BY_PROFILE = {
    profile: channel for channel, profile in POTENTIAL_PROFILE.items()
}

# Only the three hard-screened properties are compared with JP-10: density,
# calorific value and melting point. Boiling point, flash point and volumetric
# heating value stay as plain reference outputs. Lower melting point is better.
LOWER_IS_BETTER = ("melting_point",)
BENCHMARK_PROPERTIES = ("density", "mass_heating_value", "melting_point")
REFERENCE_ONLY_PROPERTIES = ("volumetric_heating_value", "boiling_point", "flash_point")


@dataclass(frozen=True)
class JP10Benchmark:
    """JP-10 reference values produced by the same models as the candidates."""

    density: float
    mass_heating_value: float
    volumetric_heating_value: float
    melting_point: float
    boiling_point: float
    flash_point: float
    model_paths: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_values(
        cls,
        *,
        density: float,
        mass_heating_value: float,
        melting_point: float,
        boiling_point: float,
        flash_point: float,
        volumetric_heating_value: float | None = None,
    ) -> "JP10Benchmark":
        if volumetric_heating_value is None:
            volumetric_heating_value = float(density) * float(mass_heating_value)
        return cls(
            density=float(density),
            mass_heating_value=float(mass_heating_value),
            volumetric_heating_value=float(volumetric_heating_value),
            melting_point=float(melting_point),
            boiling_point=float(boiling_point),
            flash_point=float(flash_point),
        )


@dataclass(frozen=True)
class TierDecision:
    tier: str
    label: str
    beats_jp10: tuple[str, ...]
    full_criteria: bool
    reason: str

    @property
    def beats_count(self) -> int:
        """How many of the three hard-screened properties match or beat JP-10."""

        return len(self.beats_jp10)


def _compare(value: float | None, reference: float, *, lower_is_better: bool) -> bool | None:
    if value is None:
        return None
    if lower_is_better:
        return float(value) <= float(reference)
    return float(value) >= float(reference)


def channel_for(
    beats_count: int,
    *,
    missed_count: int | None = None,
    hard_criteria_pass: bool | None = None,
) -> tuple[str, str]:
    """Channel and reason of one molecule.

    Two axes share one yardstick - how many of the three compared properties
    (density, mass calorific value, melting point) beat JP-10. A molecule that
    meets every enabled hard threshold lands on the tier axis and is named by
    that count alone. A molecule that misses thresholds lands on the potential
    axis, which is named by *how many thresholds it misses* and *how many
    properties still beat JP-10*: ``potential1`` (1 missed, 2 beaten),
    ``potential2`` (1 missed, 1 beaten), ``potential3`` (2 missed, 1 beaten).

    ``missed_count`` is the number of enabled hard thresholds the molecule fails
    to reach. When a caller only knows the combined verdict it passes
    ``hard_criteria_pass`` instead, and the molecule is treated as the strongest
    salvage profile its beat count can still reach. A profile that beats JP-10 on
    nothing returns an empty channel: it is not pooled.
    """

    beats = max(min(int(beats_count), 3), 0)
    if missed_count is None:
        # Only the combined verdict is known: a passing molecule misses nothing,
        # a failing one is read as the strongest salvage profile it can reach.
        missed = 0 if hard_criteria_pass else 1
    else:
        missed = max(min(int(missed_count), 3), 0)
        if hard_criteria_pass is True:
            missed = 0
    if missed == 0:
        channel = _PASS_BY_BEATS[beats]
        return (
            channel,
            "meets every enabled hard threshold; beats JP-10 on %d of %d compared "
            "properties (density, calorific value, melting point)"
            % (beats, len(BENCHMARK_PROPERTIES)),
        )
    channel = _FAIL_BY_PROFILE.get((missed, beats), "")
    if not channel:
        return (
            "",
            "misses %d enabled hard threshold(s) and beats JP-10 on %d of %d "
            "compared properties, so no salvage pool matches: no property is "
            "better than the reference"
            % (missed, beats, len(BENCHMARK_PROPERTIES)),
        )
    return (
        channel,
        "misses %d enabled hard threshold(s); beats JP-10 on %d of %d compared "
        "properties (density, calorific value, melting point)"
        % (missed, beats, len(BENCHMARK_PROPERTIES)),
    )


def classify_tier(
    values: Mapping[str, float | None],
    benchmark: JP10Benchmark,
    *,
    hard_criteria_pass: bool,
    hard_fail_count: int | None = None,
) -> TierDecision:
    """Assign one candidate to its channel: Tier 1-4 or Potential 1-3.

    ``values`` uses the same keys as :data:`BENCHMARK_PROPERTIES`.
    ``hard_criteria_pass`` is the existing threshold decision (all enabled hard
    thresholds met) computed by :mod:`property_screening`.
    ``hard_fail_count`` is how many of the three compared hard thresholds the
    molecule misses; the potential channel needs it, because Potential 2 and
    Potential 3 beat JP-10 on the same number of properties and only differ in
    how many thresholds they miss.
    """

    reference: dict[str, float] = {
        "density": benchmark.density,
        "mass_heating_value": benchmark.mass_heating_value,
        "volumetric_heating_value": benchmark.volumetric_heating_value,
        "melting_point": benchmark.melting_point,
        "boiling_point": benchmark.boiling_point,
        "flash_point": benchmark.flash_point,
    }
    beats = tuple(
        key
        for key in BENCHMARK_PROPERTIES
        if _compare(values.get(key), reference[key], lower_is_better=key in LOWER_IS_BETTER)
    )
    beats_count = len(beats)
    channel, reason = channel_for(
        beats_count,
        missed_count=hard_fail_count,
        hard_criteria_pass=bool(hard_criteria_pass),
    )
    return TierDecision(
        channel,
        CHANNEL_LABELS.get(channel, "No channel (no property better than JP-10)"),
        beats,
        bool(hard_criteria_pass),
        reason,
    )


def tier_summary(decisions: Iterable[TierDecision]) -> dict[str, int]:
    counts = {channel: 0 for channel in CHANNEL_ORDER}
    for decision in decisions:
        counts[decision.tier] = counts.get(decision.tier, 0) + 1
    return counts


def is_potential_candidate(decision: TierDecision) -> bool:
    """True for a molecule that landed in one of the salvage pools."""

    return decision.tier in POTENTIAL_ORDER


#: per-property threshold verdicts written by :mod:`property_screening`, for the
#: three compared properties, in the same order as :data:`BENCHMARK_PROPERTIES`
DECISION_COLUMNS = ("density_decision", "heating_value_decision", "melting_point_decision")


def hard_fail_count(row: Mapping[str, object]) -> int | None:
    """How many of the three compared hard thresholds a scored row misses.

    ``None`` when the row carries no per-property verdict at all, which only
    happens for folders written before those columns existed; the caller then
    falls back to the combined pass/fail verdict.
    """

    decisions = [str(row.get(column) or "").strip().lower() for column in DECISION_COLUMNS]
    if not any(decisions):
        return None
    return sum(1 for decision in decisions if decision != "pass")


# JP-10 (exo-THDC) carbon skeleton and molar mass (C10H16).
JP10_SMILES = "C1CC2C3CCC(C3)C2C1"
JP10_MOLAR_MASS_G_MOL = 136.234


def benchmark_from_models(
    *,
    sta_density_model_path: str,
    sta_formation_enthalpy_model_path: str,
    phase_fusion_melting_point_model_path: str | None = None,
    phase_fusion_boiling_point_model_path: str | None = None,
    phase_fusion_flash_point_model_path: str | None = None,
) -> JP10Benchmark:
    """Predict JP-10 with the *same* models that score the candidates.

    This keeps the coarse stage a like-for-like comparison: group-contribution
    trained models on both sides. The c-infoGAN Group y values (0.980 g/cm3,
    231 K, 42.2 MJ/kg) stay in the manuscript as the domain-level reference.
    """

    from phase_fusion import DEFAULT_PHASE_FUSION_MODEL_PATHS, predict_phase_fusion_property
    from sta_model import predict_sta_properties_batch
    from thesis_property_tools import mass_heating_value_from_formation_enthalpy

    density = float(
        predict_sta_properties_batch(
            [JP10_SMILES], sta_density_model_path, batch_size=1
        )[0]["sta_density_g_cm3"]
    )
    enthalpy = float(
        predict_sta_properties_batch(
            [JP10_SMILES], sta_formation_enthalpy_model_path, batch_size=1
        )[0]["sta_formation_enthalpy_kj_mol"]
    )
    mass_heating_value = float(
        mass_heating_value_from_formation_enthalpy(JP10_SMILES, enthalpy).heating_value_mj_kg
    )
    aux = {
        "density_g_cm3": density,
        "molar_volume_cm3_mol": JP10_MOLAR_MASS_G_MOL / density,
    }

    def phase(target: str, path: str | None, column: str) -> float:
        model_path = path or DEFAULT_PHASE_FUSION_MODEL_PATHS.get(target)
        if model_path is None:
            return float("nan")
        return float(
            predict_phase_fusion_property(
                JP10_SMILES, target=target, model_path=model_path, aux_values=aux
            )["value"]
        )

    result = JP10Benchmark.from_values(
        density=density,
        mass_heating_value=mass_heating_value,
        volumetric_heating_value=density * mass_heating_value,
        melting_point=phase("melting_point_k", phase_fusion_melting_point_model_path, "external_melting_point_k"),
        boiling_point=phase("boiling_point_k", phase_fusion_boiling_point_model_path, "external_boiling_point_k"),
        flash_point=phase("flash_point_k", phase_fusion_flash_point_model_path, "external_flash_point_k"),
    )
    from dataclasses import replace
    return replace(result, model_paths=tuple((name, str(Path(path or DEFAULT_PHASE_FUSION_MODEL_PATHS[name + "_k"]).resolve()))
        for name, path in (("melting_point", phase_fusion_melting_point_model_path),
                           ("boiling_point", phase_fusion_boiling_point_model_path),
                           ("flash_point", phase_fusion_flash_point_model_path))))


def apply_tier_fields(
    row: dict[str, object],
    benchmark: JP10Benchmark,
    *,
    hard_criteria_pass: bool | None = None,
) -> TierDecision:
    """Classify one already-scored row and write the tier columns into it."""

    for name, reference_path in benchmark.model_paths:
        candidate_path = str(row.get(f"external_{name}_model_path") or "")
        if candidate_path and Path(candidate_path).resolve() != Path(reference_path).resolve():
            raise ValueError(f"JP-10/candidate model mismatch for {name}: {reference_path} != {candidate_path}")
    for name in BENCHMARK_PROPERTIES:
        reference_value = getattr(benchmark, name)
        if not math.isfinite(reference_value):
            raise ValueError(f"JP-10 benchmark must be finite: {name}")
        row[f"jp10_reference_{name}"] = reference_value
    for name, reference_path in benchmark.model_paths:
        row[f"jp10_reference_{name}_model_path"] = reference_path

    def value(key: str) -> float | None:
        raw = row.get(key)
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    density = value("external_density_g_cm3")
    mass_hv = value("external_heating_value_mj_kg")
    volumetric_hv = value("volumetric_heating_value_mj_l")
    if volumetric_hv is None and density is not None and mass_hv is not None:
        volumetric_hv = density * mass_hv
    if hard_criteria_pass is None:
        verdict = str(
            row.get("hard_threshold_verdict") or row.get("screening_status") or ""
        )
        hard_criteria_pass = verdict.strip().lower() == "pass"
    missed = hard_fail_count(row)

    decision = classify_tier(
        {
            "density": density,
            "mass_heating_value": mass_hv,
            "volumetric_heating_value": volumetric_hv,
            "melting_point": value("external_melting_point_k"),
            "boiling_point": value("external_boiling_point_k"),
            "flash_point": value("external_flash_point_k"),
        },
        benchmark,
        hard_criteria_pass=bool(hard_criteria_pass),
        hard_fail_count=missed,
    )
    row["tier"] = decision.tier
    row["tier_label"] = decision.label
    row["tier_reason"] = decision.reason
    row["jp10_beats"] = ";".join(decision.beats_jp10)
    row["jp10_beat_count"] = decision.beats_count
    row["hard_fail_count"] = "" if missed is None else int(missed)
    row["potential_candidate"] = is_potential_candidate(decision)
    return decision
