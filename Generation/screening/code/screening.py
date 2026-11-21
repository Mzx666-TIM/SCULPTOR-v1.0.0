from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from chemistry import (
    RING_COUNT_DEFINITION,
    MoleculeDescriptors,
    descriptors_from_smiles,
)
from evaluation import GenerationEvaluation
from geometry_analysis import MMFFGeometryAnalysis, analyze_mmff_geometry
from graph import HalfGraph
from novelty import (
    fingerprint_novelty,
    load_known_fingerprint_library,
)
from ring_analysis import half_graph_ring_counts
from parallel import parallel_map, resolve_worker_count

# Below this many pending molecules the process-pool start-up cost dominates and
# the serial loop is used regardless of the requested worker count.
PARALLEL_MMFF_MIN_ITEMS = 32


def _mmff_worker(item: tuple[str, int]) -> MMFFGeometryAnalysis:
    """Top-level worker so the process pool can pickle it (spawn-safe)."""
    smiles, seed = item
    return analyze_mmff_geometry(smiles, seed=seed)


@dataclass(frozen=True)
class ScreeningConfig:
    """Soft hydrocarbon screening thresholds used before property-model training."""

    require_connected: bool = True
    require_novel: bool = True
    min_h_to_c: float = 0.7
    max_h_to_c: float = 2.2
    min_ring_count: int = 1
    max_ring_count: int = 6
    max_rotatable_bonds: int = 4
    min_fraction_sp3: float = 0.95
    max_sa_score: float = 4.5
    target_h_to_c: float = 1.35
    target_ring_count: int = 4
    # Raised on 2026-09-23: five/six-membered rings are rewarded more strongly,
    # and the strain of three/four-membered rings now costs score as well.
    preferred_ring_weight: float = 0.18
    preferred_ring_target: int = 2
    strain_ring_penalty_weight: float = 0.20
    strain_ring_reference: int = 6
    require_mmff_geometry: bool = True
    min_mmff_stability_score: float = 0.05
    max_mmff_energy_per_heavy_atom: float = 120.0
    # Bootstrap value used until `calibrate-novelty-threshold` is run on e3nn
    # validation errors for the target property.
    # Set to 0.75 on 2026-08-06 after the fingerprint upgrade (Morgan r4/8192):
    # at 0.60 the upgraded fingerprint routes 44/170 C10-12 candidates as
    # high-novelty; 0.75 keeps only the genuinely novel tail (2/170).
    high_novelty_threshold: float = 0.75
    min_carbon_count: int | None = None
    max_carbon_count: int | None = None


@dataclass(frozen=True)
class ScreenedCandidate:
    index: int
    generation_iteration: int
    passes_screen: bool
    screen_score: float
    failure_reasons: str
    is_connected: bool
    is_novel: bool | None
    max_known_similarity: float | None
    fingerprint_novelty: float | None
    high_novelty: bool | None
    nearest_known_smiles: str | None
    n1: int
    n2: int
    n3: int
    selected_count: int
    candidate_count: int
    edges_half_predicted: tuple[tuple[int, int], ...]
    edges_full_predicted: tuple[tuple[int, int], ...] | None
    canonical_smiles: str
    formula: str
    molecular_weight: float
    carbon_count: int
    hydrogen_count: int
    h_to_c_ratio: float
    ring_count: int
    ring_count_definition: str
    rdkit_ring_count: int
    three_member_rings: int
    four_member_rings: int
    five_member_rings: int
    six_member_rings: int
    half_real_only_rings: int
    half_virtual_rings: int
    half_projected_ring_count: int
    projected_three_member_rings: int
    projected_four_member_rings: int
    projected_five_member_rings: int
    projected_six_member_rings: int
    strain_ring_score: int
    preferred_ring_score: int
    rotatable_bonds: int
    fraction_sp3: float
    sa_score: float
    mmff_embed_success: bool
    mmff_optimize_success: bool
    mmff_converged: bool
    mmff_energy_kcal_mol: float | None
    mmff_energy_per_atom: float | None
    mmff_energy_per_heavy_atom: float | None
    mmff_stability_score: float
    mmff_failure_reason: str
    edges_base: tuple[tuple[int, int], ...]
    predicted_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ScreeningSummary:
    total_generated: int
    valid_candidates: int
    screened_candidates: int
    passed_screen: int
    connected_candidates: int
    novel_candidates: int
    high_novelty_candidates: int
    high_novelty_passed_screen: int
    unique_screened_smiles: int
    mean_screen_score: float
    ring_count_definition: str


def screen_generation(
    evaluation: GenerationEvaluation,
    config: ScreeningConfig = ScreeningConfig(),
    known_smiles_path: str | Path | None = None,
    descriptor_cache: dict[str, object] | None = None,
    mmff_cache: dict[str, object] | None = None,
    novelty_cache: dict[str, object] | None = None,
    mmff_workers: int | None = None,
) -> tuple[tuple[ScreenedCandidate, ...], ScreeningSummary]:
    """Compute descriptors, apply filters, and score generated candidates."""

    novelty_library = load_known_fingerprint_library(known_smiles_path)
    candidates: list[ScreenedCandidate] = []
    scores: list[float] = []
    descriptor_cache = {} if descriptor_cache is None else descriptor_cache
    mmff_cache = {} if mmff_cache is None else mmff_cache
    novelty_cache = {} if novelty_cache is None else novelty_cache

    # MMFF geometry is the dominant cost of screening. The computation for a
    # molecule depends only on its canonical SMILES and on the enumeration index
    # of its first occurrence, so it can be distributed over worker processes
    # without changing any recorded value.
    pending_mmff: list[tuple[str, int]] = []
    queued: set[str] = set()
    for index, result in enumerate(evaluation.results, start=1):
        if not result.is_valid or not result.canonical_smiles:
            continue
        smiles = result.canonical_smiles
        if smiles in mmff_cache or smiles in queued:
            continue
        queued.add(smiles)
        pending_mmff.append((smiles, index))
    if pending_mmff:
        workers = resolve_worker_count(mmff_workers, memory_per_worker_gb=0.4)
        if len(pending_mmff) < PARALLEL_MMFF_MIN_ITEMS:
            workers = 1
        for (smiles, _seed), analysis in zip(
            pending_mmff, parallel_map(_mmff_worker, pending_mmff, workers)
        ):
            mmff_cache[smiles] = analysis

    for index, result in enumerate(evaluation.results, start=1):
        if not result.is_valid or not result.canonical_smiles:
            continue
        if result.canonical_smiles not in descriptor_cache:
            descriptor_cache[result.canonical_smiles] = descriptors_from_smiles(
                result.canonical_smiles
            )
        descriptors = descriptor_cache[result.canonical_smiles]
        if result.canonical_smiles not in mmff_cache:
            mmff_cache[result.canonical_smiles] = analyze_mmff_geometry(
                result.canonical_smiles,
                seed=index,
            )
        mmff = mmff_cache[result.canonical_smiles]
        rings = half_graph_ring_counts(
            HalfGraph(result.n1, result.n2, result.n3, result.edges_half_predicted)
        )
        if result.canonical_smiles not in novelty_cache:
            novelty_cache[result.canonical_smiles] = fingerprint_novelty(
                result.canonical_smiles,
                novelty_library,
            )
        novelty = novelty_cache[result.canonical_smiles]
        high_novelty = (
            novelty.novelty_score >= config.high_novelty_threshold
            if novelty is not None
            else None
        )
        passes, reasons = passes_screen(
            descriptors,
            is_connected=result.is_connected,
            is_novel=result.is_novel,
            config=config,
            ring_count=descriptors.ring_count,
            mmff=mmff,
        )
        score = fuel_candidate_score(
            descriptors,
            is_connected=result.is_connected,
            is_novel=result.is_novel,
            config=config,
            ring_count=descriptors.ring_count,
            preferred_ring_score=descriptors.preferred_ring_score,
            strain_ring_score=descriptors.strain_ring_score,
            mmff_stability_score=mmff.mmff_stability_score,
        )
        scores.append(score)
        descriptor_values = asdict(descriptors)
        descriptor_values.pop("ring_count")
        rdkit_ring_count = int(descriptor_values.pop("rdkit_ring_count"))
        candidates.append(
            ScreenedCandidate(
                index=index,
                generation_iteration=result.generation_iteration,
                passes_screen=passes,
                screen_score=score,
                failure_reasons=";".join(reasons),
                is_connected=result.is_connected,
                is_novel=result.is_novel,
                max_known_similarity=novelty.max_similarity if novelty else None,
                fingerprint_novelty=novelty.novelty_score if novelty else None,
                high_novelty=high_novelty,
                nearest_known_smiles=novelty.nearest_smiles if novelty else None,
                n1=result.n1,
                n2=result.n2,
                n3=result.n3,
                selected_count=result.selected_count,
                candidate_count=result.candidate_count,
                edges_base=result.edges_base,
                predicted_edges=result.predicted_edges,
                edges_half_predicted=result.edges_half_predicted,
                edges_full_predicted=result.edges_full_predicted,
                ring_count=descriptors.ring_count,
                ring_count_definition=RING_COUNT_DEFINITION,
                rdkit_ring_count=rdkit_ring_count,
                half_real_only_rings=rings.real_only_rings,
                half_virtual_rings=rings.virtual_rings,
                half_projected_ring_count=rings.projected_total_rings,
                projected_three_member_rings=rings.projected_three_member_rings,
                projected_four_member_rings=rings.projected_four_member_rings,
                projected_five_member_rings=rings.projected_five_member_rings,
                projected_six_member_rings=rings.projected_six_member_rings,
                mmff_embed_success=mmff.embed_success,
                mmff_optimize_success=mmff.optimize_success,
                mmff_converged=mmff.mmff_converged,
                mmff_energy_kcal_mol=mmff.mmff_energy_kcal_mol,
                mmff_energy_per_atom=mmff.mmff_energy_per_atom,
                mmff_energy_per_heavy_atom=mmff.mmff_energy_per_heavy_atom,
                mmff_stability_score=mmff.mmff_stability_score,
                mmff_failure_reason=mmff.mmff_failure_reason,
                **descriptor_values,
            )
        )

    passed = [candidate for candidate in candidates if candidate.passes_screen]
    high_novelty_candidates = [
        candidate for candidate in candidates if candidate.high_novelty is True
    ]
    summary = ScreeningSummary(
        total_generated=evaluation.generated_bases,
        valid_candidates=len(candidates),
        screened_candidates=len(candidates),
        passed_screen=len(passed),
        connected_candidates=sum(1 for candidate in candidates if candidate.is_connected),
        novel_candidates=sum(1 for candidate in candidates if candidate.is_novel),
        high_novelty_candidates=len(high_novelty_candidates),
        high_novelty_passed_screen=sum(
            1 for candidate in high_novelty_candidates if candidate.passes_screen
        ),
        unique_screened_smiles=len({candidate.canonical_smiles for candidate in candidates}),
        mean_screen_score=round(sum(scores) / len(scores), 6) if scores else 0.0,
        ring_count_definition=RING_COUNT_DEFINITION,
    )
    return tuple(candidates), summary


def passes_screen(
    descriptors: MoleculeDescriptors,
    is_connected: bool,
    is_novel: bool | None,
    config: ScreeningConfig,
    ring_count: int | None = None,
    mmff: MMFFGeometryAnalysis | None = None,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if config.min_carbon_count is not None and descriptors.carbon_count < config.min_carbon_count:
        reasons.append("carbon_count_below_min")
    if config.max_carbon_count is not None and descriptors.carbon_count > config.max_carbon_count:
        reasons.append("carbon_count_above_max")
    effective_ring_count = descriptors.ring_count if ring_count is None else ring_count
    if config.require_connected and not is_connected:
        reasons.append("not_connected")
    if config.require_novel and is_novel is not True:
        reasons.append("not_novel")
    if descriptors.h_to_c_ratio < config.min_h_to_c:
        reasons.append("h_to_c_below_min")
    if descriptors.h_to_c_ratio > config.max_h_to_c:
        reasons.append("h_to_c_above_max")
    if effective_ring_count < config.min_ring_count:
        reasons.append("ring_count_below_min")
    if effective_ring_count > config.max_ring_count:
        reasons.append("ring_count_above_max")
    if descriptors.rotatable_bonds > config.max_rotatable_bonds:
        reasons.append("rotatable_bonds_above_max")
    if descriptors.fraction_sp3 < config.min_fraction_sp3:
        reasons.append("fraction_sp3_below_min")
    if descriptors.sa_score > config.max_sa_score:
        reasons.append("sa_score_above_max")
    if config.require_mmff_geometry:
        if mmff is None:
            reasons.append("mmff_missing")
        elif not mmff.embed_success:
            reasons.append("mmff_embed_failed")
        elif not mmff.optimize_success:
            reasons.append("mmff_optimize_failed")
        elif mmff.mmff_stability_score < config.min_mmff_stability_score:
            reasons.append("mmff_stability_below_min")
        elif (
            mmff.mmff_energy_per_heavy_atom is not None
            and mmff.mmff_energy_per_heavy_atom > config.max_mmff_energy_per_heavy_atom
        ):
            reasons.append("mmff_energy_above_max")
    return not reasons, tuple(reasons)


def fuel_candidate_score(
    descriptors: MoleculeDescriptors,
    is_connected: bool,
    is_novel: bool | None,
    config: ScreeningConfig,
    ring_count: int | None = None,
    preferred_ring_score: int = 0,
    strain_ring_score: int = 0,
    mmff_stability_score: float | None = None,
) -> float:
    """A transparent proxy score until measured or predicted fuel properties exist."""

    effective_ring_count = descriptors.ring_count if ring_count is None else ring_count
    h_score = _triangular_score(
        descriptors.h_to_c_ratio,
        low=config.min_h_to_c,
        target=config.target_h_to_c,
        high=config.max_h_to_c,
    )
    ring_score = _triangular_score(
        float(effective_ring_count),
        low=float(config.min_ring_count),
        target=float(config.target_ring_count),
        high=float(config.max_ring_count),
    )
    flexibility_score = 1.0 - min(
        descriptors.rotatable_bonds / max(config.max_rotatable_bonds, 1), 1.0
    )
    sp3_score = min(max(descriptors.fraction_sp3, 0.0), 1.0)
    connectivity_score = 1.0 if is_connected else 0.0
    novelty_score = 1.0 if is_novel is True else 0.0
    preferred_ring_reward = min(
        preferred_ring_score / max(config.preferred_ring_target, 1),
        1.0,
    )
    # Thermal-stability cost of the strained rings: normalised by the reference
    # count, so a molecule carrying many three/four-membered rings is pushed down.
    strain_penalty = min(
        strain_ring_score / max(float(config.strain_ring_reference), 1.0),
        1.0,
    )
    mmff_score = 0.5 if mmff_stability_score is None else min(
        max(float(mmff_stability_score), 0.0),
        1.0,
    )

    score = (
        0.19 * h_score
        + 0.16 * ring_score
        + 0.15 * flexibility_score
        + 0.12 * sp3_score
        + 0.08 * connectivity_score
        + 0.08 * novelty_score
        + 0.12 * mmff_score
        + config.preferred_ring_weight * preferred_ring_reward
        - config.strain_ring_penalty_weight * strain_penalty
    )
    return round(float(score), 6)


def _triangular_score(value: float, low: float, target: float, high: float) -> float:
    if high <= low:
        return 0.0
    if value < low or value > high:
        return 0.0
    if value == target:
        return 1.0
    if value < target:
        return max(0.0, (value - low) / max(target - low, 1e-9))
    return max(0.0, (high - value) / max(high - target, 1e-9))
