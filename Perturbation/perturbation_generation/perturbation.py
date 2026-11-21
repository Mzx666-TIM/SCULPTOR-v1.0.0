from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from campaign_layout import ALLOWED_PERTURBATION_OPERATIONS, PERTURBED_STAGE
from chemistry import RING_COUNT_DEFINITION, descriptors_from_smiles
from finalization import require_perturbation_seed_rows
from geometry_analysis import analyze_mmff_geometry
from known_fuel_library import match_known_fuel
from perturbation_iteration import (
    DOCUMENTED_JP10_REFERENCE,
    JP10Reference,
    MARGIN_KEYS,
    RING_REWARD_FACTOR,
    RING_STRAIN_FACTOR,
    attempts_per_operation,
    beats_from_margins,
    build_features,
    feature_columns,
    iteration_model_contract_is_current,
    load_iteration_model,
    margins_of_row,
    parent_index,
    ring_preference,
    select_iteration_candidates,
)
from perturbation_iteration import _number as _iteration_number

from pipeline import (
    _add_objective_scores,
    _batch_sta_predictions_for_smiles,
    _parallel_property_deltas,
    _property_delta_worker,
    _property_worker_init,
    _route_enriched_property_row,
    _selected_sta_model_paths,
    public_rows,
)
from parallel import parallel_map, resolve_worker_count
from property_cache import PropertyCache, config_fingerprint
from tier_routing import (
    CHANNEL_FILENAMES,
    CHANNEL_ORDER,
    POTENTIAL1,
    POTENTIAL2,
    POTENTIAL3,
    TIER1,
    TIER2,
    TIER3,
    TIER4,
    apply_tier_fields,
    benchmark_from_models,
)
from property_consensus import (
    CompositeScoreWeights,
    PropertyScoreWeights,
    PropertyTargetValues,
)
from property_model.e3nn_geometry_scorer import predict_geometry_score_batch
from property_screening import FuelPropertyThresholds
from screening import ScreeningConfig, fuel_candidate_score, passes_screen
from sta_feedback import STA_FEEDBACK_COLUMNS, build_sta_weak_feedback_rows
from fuel_design.paths import CBH_BDC_DIR, EVAL_DATA_DIR, SFS_VR_DIR

# Default paths of non-STA methods are kept only as historical defaults;
# the modules themselves are archived in addition.
DEFAULT_CBH_BDC_MODEL_PATH = CBH_BDC_DIR / "cbh_bdc_delta_ml_formation.joblib"
DEFAULT_CBH_BDC_FORMATION_FEATURE_PATH = (
    EVAL_DATA_DIR / "qm9_cbh_bdc_formation_features.csv"
)
DEFAULT_YANG_CBH_BDC_ROOT = CBH_BDC_DIR / "CBH-BDC-deltaML-main"
DEFAULT_SFS_VR_MP_MODEL_PATH = SFS_VR_DIR / "sfs_vr_melting_point.joblib"
DEFAULT_SFS_VR_MP_DATA_PATH = (
    SFS_VR_DIR / "SFS-VR-main" / "MP" / "MP-454-select-FEATURE - 15.xlsx"
)


PERTURBATION_OPERATIONS = tuple(sorted(ALLOWED_PERTURBATION_OPERATIONS))


def campaign_tag(finalized_hits_path: str | Path) -> str:
    """Short label of a perturbation campaign, taken from its seed table name.

    ``perturbation_seeds_T123.csv`` becomes ``T123`` and
    ``perturbation_seeds_potential.csv`` becomes ``potential``. The label is part
    of every product id, so the products of two campaigns that share one set of
    tables stay distinguishable.
    """

    stem = Path(str(finalized_hits_path)).stem
    prefix = "perturbation_seeds_"
    if stem.lower().startswith(prefix):
        stem = stem[len(prefix) :]
    label = "".join(ch for ch in stem if ch.isalnum() or ch in "-_")
    return label.strip("_-") or "campaign"


def _panel_stage(stage: str, message: str) -> None:
    print(f"__FUEL_PANEL_STAGE__:{stage}:{message}", flush=True)


@dataclass(frozen=True)
class PerturbationPreparationResult:
    finalized_seed_rows: int
    selected_seed_rows: int
    samples: int
    output_dir: str
    seed_table_path: str
    seed_manifest_path: str
    operation_count: int
    summary_path: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationGenerationResult:
    finalized_seed_rows: int
    selected_seed_rows: int
    attempted_samples: int
    generated_rows: int
    duplicate_rows: int
    failed_attempts: int
    output_dir: str
    candidates_path: str
    seed_manifest_path: str
    summary_path: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationPipelineResult:
    attempted_samples: int
    generated_rows: int
    screened_rows: int
    passed_screen_rows: int
    rejected_pre_property_rows: int
    property_prediction_rows: int
    property_failed_rows: int
    tier1_rows: int
    tier2_rows: int
    tier3_rows: int
    tier4_rows: int
    potential1_rows: int
    potential2_rows: int
    potential3_rows: int
    carried_over_product_rows: int
    merged_product_rows: int
    replaced_existing_products: bool
    iteration_round: int
    iteration_evaluated_rows: int
    iteration_selected_rows: int
    iteration_predicted_tier1_rows: int
    missing_property_rows: int
    manual_review_rows: int
    sta_weak_feedback_rows: int
    output_dir: str
    candidates_path: str
    screened_path: str
    property_predictions_path: str
    sta_weak_feedback_path: str
    rejected_pre_property_path: str
    property_failed_path: str
    missing_property_path: str
    manual_review_path: str
    summary_path: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def prepare_perturbation_campaign(
    finalized_hits_path: str | Path,
    output_dir: str | Path,
    samples: int,
    seed: int = 0,
    operations: tuple[str, ...] = PERTURBATION_OPERATIONS,
    max_seeds: int | None = None,
    thresholds: FuelPropertyThresholds = FuelPropertyThresholds(),
) -> PerturbationPreparationResult:
    """Prepare perturbation folders from the Tier 1-3 symmetric hits."""

    if samples < 0:
        raise ValueError("samples must be non-negative")
    for operation in operations:
        if operation not in ALLOWED_PERTURBATION_OPERATIONS:
            raise ValueError(f"unknown perturbation operation: {operation}")

    finalized_rows = list(
        require_perturbation_seed_rows(finalized_hits_path, thresholds=thresholds)
    )
    rng = random.Random(seed)
    seed_rows = list(finalized_rows)
    rng.shuffle(seed_rows)
    if max_seeds is not None:
        seed_rows = seed_rows[:max_seeds]

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    # The per-(seed, operation) folders were removed on 2026-09-22: they were
    # created for every combination before any molecule existed, which turned a
    # 1000-seed campaign into ~6k folders plus ~19k files of bookkeeping.

    seed_manifest = root / "perturbation_seeds.csv"
    _write_rows(seed_manifest, seed_rows)
    summary_path = root / "perturbation_preparation_summary.json"
    result = PerturbationPreparationResult(
        finalized_seed_rows=len(finalized_rows),
        selected_seed_rows=len(seed_rows),
        samples=samples,
        output_dir=str(root),
        seed_table_path=str(finalized_hits_path),
        seed_manifest_path=str(seed_manifest),
        operation_count=len(operations),
        summary_path=str(summary_path),
    )
    summary_path.write_text(
        json.dumps(
            {
                **result.as_dict(),
                "operations": list(operations),
                "ring_count_definition": RING_COUNT_DEFINITION,
                "run_rule": (
                    "samples is the perturbation attempt budget; the pipeline does "
                    "not try to reach a target number of usable molecules"
                ),
                "seed_rule": (
                    "perturbation seeds are the Tier 1-3 candidate pools of the run "
                    "(perturbation_seeds_T123.csv); confirmed results are not required "
                    "before perturbation"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def run_perturbation_generation(
    finalized_hits_path: str | Path,
    output_dir: str | Path,
    samples: int,
    seed: int = 0,
    operations: tuple[str, ...] = PERTURBATION_OPERATIONS,
    max_seeds: int | None = None,
    samples_per_seed_operation: bool = False,
    exclude_smiles_paths: tuple[str | Path, ...] = (),
    thresholds: FuelPropertyThresholds = FuelPropertyThresholds(),
    id_offset: int = 0,
    operation_attempts: Mapping[str, int] | None = None,
) -> PerturbationGenerationResult:
    """Generate perturbation candidates from the Tier 1-3 symmetric hits.

    By default, ``samples`` is the total attempt budget. When
    ``samples_per_seed_operation`` is enabled, every selected seed receives
    ``samples`` attempts for every enabled operation.
    """

    preparation = prepare_perturbation_campaign(
        finalized_hits_path=finalized_hits_path,
        output_dir=output_dir,
        samples=samples,
        seed=seed,
        operations=operations,
        max_seeds=max_seeds,
        thresholds=thresholds,
    )
    seed_rows = _read_rows(Path(preparation.seed_manifest_path))
    rng = random.Random(seed)
    root = Path(output_dir)
    id_prefix = "perturbed_%s_" % campaign_tag(finalized_hits_path)
    id_offset = max(int(id_offset), 0)
    seen_smiles = {
        str(row.get("canonical_smiles", "")).strip()
        for row in seed_rows
        if str(row.get("canonical_smiles", "")).strip()
    }
    seen_smiles.update(_load_smiles_from_paths(exclude_smiles_paths))
    candidate_rows: list[dict[str, object]] = []
    group_counts: dict[tuple[int, int, str, int, int, int], int] = {}
    duplicate_rows = 0
    failed_attempts = 0

    if samples_per_seed_operation:
        # An iterative round spends a different number of attempts per operator,
        # following the sampling guidance of the (static) iteration model; a plain
        # campaign gives every operator the same ``samples`` attempts.
        per_operation = {
            operation: int((operation_attempts or {}).get(operation, samples))
            for operation in operations
        }
        attempt_plan = [
            (seed_row, operation, operation_attempt)
            for seed_row in seed_rows
            for operation in operations
            for operation_attempt in range(1, per_operation[operation] + 1)
        ]
    else:
        attempt_plan = [
            (
                rng.choice(seed_rows) if seed_rows else None,
                rng.choice(operations) if operations else "",
                attempt_index,
            )
            for attempt_index in range(1, samples + 1)
        ]

    for attempt_index, (seed_row, operation, operation_attempt) in enumerate(
        attempt_plan,
        start=1,
    ):
        if seed_row is None:
            failed_attempts += 1
            continue
        parent_smiles = str(seed_row.get("canonical_smiles", "")).strip()
        try:
            canonical, detail = _perturb_smiles(parent_smiles, operation, rng)
        except Exception:
            failed_attempts += 1
            continue
        if not canonical:
            failed_attempts += 1
            continue
        if canonical in seen_smiles:
            duplicate_rows += 1
            continue
        seen_smiles.add(canonical)

        descriptors = descriptors_from_smiles(canonical)
        source_carbon_count = int(seed_row.get("carbon_count") or seed_row.get("total_carbons") or 0)
        seed_index = int(seed_row.get("carbon_group_index") or 0)
        n1 = int(seed_row.get("n1") or 0)
        n2 = int(seed_row.get("n2") or 0)
        n3 = int(seed_row.get("n3") or 0)
        known_fuel = match_known_fuel(canonical)
        group_key = "%s/%s/%s/%s" % (
            PERTURBED_STAGE,
            "C%d" % source_carbon_count,
            "seed_%04d" % seed_index if seed_index else "seed_unknown",
            operation,
        )
        row = {
            "candidate_id": f"{id_prefix}{id_offset + len(candidate_rows) + 1:06d}",
            "attempt_index": attempt_index,
            "operation_attempt_index": operation_attempt,
            "operation": operation,
            "canonical_smiles": canonical,
            "parent_canonical_smiles": parent_smiles,
            "parent_candidate_id": seed_row.get("candidate_id", ""),
            "parent_perturbation_seed_id": seed_row.get("perturbation_seed_id", ""),
            "source_carbon_count": source_carbon_count,
            "source_carbon_group_index": seed_index,
            "source_n1": n1,
            "source_n2": n2,
            "source_n3": n3,
            "carbon_count": descriptors.carbon_count,
            "hydrogen_count": descriptors.hydrogen_count,
            "h_to_c_ratio": round(descriptors.h_to_c_ratio, 6),
            "ring_count": descriptors.ring_count,
            "ring_count_definition": RING_COUNT_DEFINITION,
            "fraction_sp3": round(descriptors.fraction_sp3, 6),
            "sa_score": descriptors.sa_score,
            "mutation_detail": detail,
            # Group folders are no longer materialised; the key keeps the
            # provenance that used to live in the folder path.
            "group_key": group_key,
            "output_dir": "",
            "stage": "perturbed_from_hits",
        }
        row.update(asdict(descriptors))
        row["known_fuel"] = bool(known_fuel)
        row["known_fuel_names"] = (
            ";".join(known_fuel["known_fuel_names"]) if known_fuel else ""
        )
        candidate_rows.append(row)
        group_counts[(source_carbon_count, seed_index, operation, n1, n2, n3)] = (
            group_counts.get((source_carbon_count, seed_index, operation, n1, n2, n3), 0) + 1
        )

    candidates_path = root / "perturbed_candidates.csv"
    _write_rows(candidates_path, candidate_rows)
    group_summary_path = root / "perturbation_group_summary.csv"
    _write_rows(
        group_summary_path,
        [
            {
                "source_carbon_count": carbon,
                "source_carbon_group_index": seed_index,
                "operation": operation,
                "source_n1": n1,
                "source_n2": n2,
                "source_n3": n3,
                "generated": count,
            }
            for (carbon, seed_index, operation, n1, n2, n3), count in sorted(group_counts.items())
        ],
    )

    summary_path = root / "perturbation_generation_summary.json"
    result = PerturbationGenerationResult(
        finalized_seed_rows=preparation.finalized_seed_rows,
        selected_seed_rows=preparation.selected_seed_rows,
        attempted_samples=len(attempt_plan),
        generated_rows=len(candidate_rows),
        duplicate_rows=duplicate_rows,
        failed_attempts=failed_attempts,
        output_dir=str(root),
        candidates_path=str(candidates_path),
        seed_manifest_path=preparation.seed_manifest_path,
        summary_path=str(summary_path),
    )
    operation_counts: dict[str, int] = {}
    for row in candidate_rows:
        operation = str(row["operation"])
        operation_counts[operation] = operation_counts.get(operation, 0) + 1
    summary_path.write_text(
        json.dumps(
            {
                **result.as_dict(),
                "operations": list(operations),
                "ring_count_definition": RING_COUNT_DEFINITION,
                "operation_counts": operation_counts,
                "samples_per_seed_operation": samples_per_seed_operation,
                "requested_samples_per_group": (
                    samples if samples_per_seed_operation else None
                ),
                "attempts_per_operation": (
                    {
                        operation: int((operation_attempts or {}).get(operation, samples))
                        for operation in operations
                    }
                    if samples_per_seed_operation
                    else None
                ),
                "run_rule": (
                    "samples is applied to every seed-operation group"
                    if samples_per_seed_operation
                    else "samples is the total perturbation attempt budget"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def _perturbation_mmff_worker(item: tuple[str, int]):
    """Top-level MMFF worker (spawn-safe) for the perturbation screening pass."""
    smiles, geometry_seed = item
    return analyze_mmff_geometry(smiles, seed=geometry_seed)


def check_reference_libraries(
    *,
    known_smiles_path: str | Path | None,
    temporary_smiles_path: str | Path | None = None,
    perturbation_temporary_smiles_path: str | Path | None = None,
    products_already_present: bool = False,
) -> dict[str, list[str]]:
    """Report the dedup libraries a stage is about to use.

    The screening step only compares against files that exist, so a path that
    was typed wrongly - or one the previous stage never wrote - would silently
    drop a whole reference library. The panel gets a warning line for every such
    path instead, and the caller can record the two lists in its summary.

    ``products_already_present`` says whether the folder already holds perturbed
    products: the perturbation library is written by the campaign itself, so it
    is only *expected* on a follow-up campaign - on the first one its absence is
    the normal state, not a gap.
    """

    libraries = {
        "known_smiles_path": known_smiles_path,
        "temporary_smiles_path": temporary_smiles_path,
        "perturbation_temporary_smiles_path": perturbation_temporary_smiles_path,
    }
    self_written = (
        set() if products_already_present else {"perturbation_temporary_smiles_path"}
    )
    missing = [
        key for key, value in libraries.items() if not value and key not in self_written
    ]
    absent = [
        key
        for key, value in libraries.items()
        if value and key not in self_written and not Path(str(value)).exists()
    ]
    for key in missing:
        _panel_stage(
            "generate",
            "WARNING: no %s was configured - dedup only sees the libraries that "
            "were passed" % key.replace("_", " "),
        )
    for key in absent:
        _panel_stage(
            "generate",
            "WARNING: %s does not exist (%s) - that library is NOT used for dedup"
            % (key.replace("_", " "), libraries[key]),
        )
    return {"missing": missing, "absent": absent}


def _campaign_screening_config(root: Path, config: ScreeningConfig) -> ScreeningConfig:
    """Inherit the generation carbon range without narrowing explicitly set bounds."""
    carbons = [int(p.name[1:]) for p in root.glob("C*") if p.is_dir() and p.name[1:].isdigit()]
    if not carbons:
        return config
    return replace(config,
        min_carbon_count=config.min_carbon_count if config.min_carbon_count is not None else min(carbons),
        max_carbon_count=config.max_carbon_count if config.max_carbon_count is not None else max(carbons))


def run_perturbation_pipeline(
    finalized_hits_path: str | Path,
    output_dir: str | Path,
    samples: int,
    known_smiles_path: str | Path | None = None,
    temporary_smiles_path: str | Path | None = None,
    perturbation_temporary_smiles_path: str | Path | None = None,
    seed: int = 0,
    operations: tuple[str, ...] = PERTURBATION_OPERATIONS,
    max_seeds: int | None = None,
    samples_per_seed_operation: bool = False,
    novelty_threshold: float = 0.75,
    uncertainty_scale: float = 1.0,
    ignore_density: bool = False,
    ignore_heating_value: bool = True,
    heating_values_csv: str | Path | None = None,
    mp_model_path: str | Path = DEFAULT_SFS_VR_MP_MODEL_PATH,
    mp_data_path: str | Path = DEFAULT_SFS_VR_MP_DATA_PATH,
    e3nn_geometry_checkpoint: str | Path | None = None,
    e3nn_property_checkpoint: str | Path | None = None,
    e3nn_property_name: str = "qm9_h298_hartree",
    sta_property_model_path: str | Path | None = None,
    sta_density_model_path: str | Path | None = None,
    sta_heating_value_model_path: str | Path | None = None,
    sta_melting_point_model_path: str | Path | None = None,
    sta_boiling_point_model_path: str | Path | None = None,
    sta_flash_point_model_path: str | Path | None = None,
    sta_formation_enthalpy_model_path: str | Path | None = None,
    sta_molar_volume_model_path: str | Path | None = None,
    sta_isp_model_path: str | Path | None = None,
    density_model: str = "thesis_vdw_density",
    heating_value_model: str = "yang_cbh_bdc",
    melting_point_model: str = "phase_fusion",
    boiling_point_model: str = "none",
    flash_point_model: str = "none",
    formation_enthalpy_model: str = "yang_cbh_bdc",
    molar_volume_model: str = "derived",
    isp_model: str = "none",
    cbh_bdc_model_path: str | Path = DEFAULT_CBH_BDC_MODEL_PATH,
    cbh_bdc_feature_csv: str | Path = DEFAULT_CBH_BDC_FORMATION_FEATURE_PATH,
    cbh_bdc_yang_root: str | Path = DEFAULT_YANG_CBH_BDC_ROOT,
    phase_fusion_melting_point_model_path: str | Path | None = None,
    phase_fusion_boiling_point_model_path: str | Path | None = None,
    phase_fusion_flash_point_model_path: str | Path | None = None,
    use_cbh_bdc_heating: bool = True,
    sta_weak_feedback_enabled: bool = True,
    sta_weak_feedback_weight: float = 0.35,
    screening_config: ScreeningConfig = ScreeningConfig(),
    score_weights: CompositeScoreWeights = CompositeScoreWeights(),
    property_score_weights: PropertyScoreWeights = PropertyScoreWeights(),
    thresholds: FuelPropertyThresholds = FuelPropertyThresholds(),
    use_melting_point_screen: bool = False,
    use_boiling_point_screen: bool = False,
    use_flash_point_screen: bool = False,
    use_volumetric_heating_value_screen: bool = False,
    missing_property_penalty: float = 0.15,
    known_fuel_check: bool = False,
    screen_workers: int | None = None,
    property_workers: int | None = None,
    use_property_cache: bool = True,
    property_cache_path: str | Path | None = None,
    reproducibility_scope: str = "tier12",
    merge_existing_products: bool = True,
    iteration_model_path: str | Path | None = None,
    iteration_budget: int | None = None,
    iteration_round: int | None = None,
    iteration_exploration_fraction: float = 0.15,
    operation_attempts: Mapping[str, int] | None = None,
    iteration_ring_weight: float = 1.0,
) -> PerturbationPipelineResult:
    """Run perturbation generation, novelty/MMFF screening, and property routing.

    ``merge_existing_products`` keeps the products of earlier campaigns that
    already sit in ``output_dir``: the new campaign is perturbed, screened and
    predicted as usual and its rows are then appended to the same set of tables,
    so one run folder always shows every campaign together. Pass ``False`` to
    replace the stored products instead.

    ``iteration_model_path`` turns the campaign into one round of the iterative
    search for Tier 1: the enumerated products are scored by the pre-trained
    property-change model, only the most promising ``iteration_budget`` of them
    (plus an exploration share) go through screening and property prediction, and
    every attempt row records what the model predicted for it. The rounds
    themselves are driven by :func:`run_iterative_perturbation`.
    """

    _panel_stage("generate", "Generating perturbation candidates")
    check_reference_libraries(
        known_smiles_path=known_smiles_path,
        temporary_smiles_path=temporary_smiles_path,
        perturbation_temporary_smiles_path=perturbation_temporary_smiles_path,
        products_already_present=(Path(output_dir) / "perturbed_candidates.csv").exists(),
    )
    root = Path(output_dir)
    screened_path = root / "perturbation_screened.csv"
    screening_config = _campaign_screening_config(root, screening_config)
    property_path = root / "perturbation_property_predictions.csv"
    sta_weak_feedback_path = root / "sta_low_novelty_weak_feedback.csv"
    rejected_path = root / "rejected_pre_property.csv"
    property_failed_path = root / "property_failed_candidates.csv"
    missing_property_path = root / "missing_property_candidates.csv"
    manual_review_path = (
        root / "manual_review" / "candidates_for_manual_calculation.csv"
    )
    summary_path = root / "perturbation_pipeline_summary.json"
    previous_seed_rows: list[dict[str, object]] = []
    previous_generation_summary: dict[str, object] = {}
    campaign_id_offset = 0
    stored_candidates = root / "perturbed_candidates.csv"
    prior_candidates = stored_candidates.with_name(stored_candidates.name + ".prior")
    prior_candidates_path: Path | None = None
    if prior_candidates.exists():
        if stored_candidates.exists():
            # A run died between renaming and re-merging the attempt log: put the
            # two halves back together before anything else touches the tables.
            _merge_table_files(stored_candidates, prior_candidates)
        elif merge_existing_products:
            prior_candidates.replace(stored_candidates)
        else:
            prior_candidates.unlink()
    if merge_existing_products:
        # Refuse to mix two model versions in one table before anything is
        # generated, then keep the stored attempt log and seed manifest so the
        # merged result can be rebuilt at the end of the campaign.
        if property_path.exists():
            _assert_models_match_stored_products(
                property_path,
                {
                    "external_melting_point_model_path": phase_fusion_melting_point_model_path,
                    "external_boiling_point_model_path": phase_fusion_boiling_point_model_path,
                    "external_flash_point_model_path": phase_fusion_flash_point_model_path,
                    "e3nn_geometry_model_checkpoint": e3nn_geometry_checkpoint,
                    "sta_property_model_checkpoint": sta_property_model_path,
                },
            )
        previous_seed_rows = [
            dict(row) for row in _read_rows(root / "perturbation_seeds.csv")
        ]
        previous_generation_summary = _read_json_dict(
            root / "perturbation_generation_summary.json"
        )
        # Continue the candidate numbering of everything already stored under
        # this campaign name (the attempt log holds ids of products that never
        # reached property prediction), so every table of the folder agrees on
        # one id per molecule.
        campaign_prefix = "perturbed_%s_" % campaign_tag(finalized_hits_path)
        for table in (property_path, stored_candidates):
            if table.exists():
                campaign_id_offset = max(
                    campaign_id_offset,
                    _max_stored_id_suffix(table, campaign_prefix),
                )
        if stored_candidates.exists():
            stored_candidates.replace(prior_candidates)
            prior_candidates_path = prior_candidates
    exclude_paths = tuple(
        path
        for path in (
            known_smiles_path,
            temporary_smiles_path,
            perturbation_temporary_smiles_path,
        )
        if path is not None
    )
    generation = run_perturbation_generation(
        finalized_hits_path=finalized_hits_path,
        output_dir=output_dir,
        samples=samples,
        seed=seed,
        operations=operations,
        max_seeds=max_seeds,
        samples_per_seed_operation=samples_per_seed_operation,
        exclude_smiles_paths=exclude_paths,
        thresholds=thresholds,
        id_offset=campaign_id_offset,
        operation_attempts=operation_attempts,
    )
    # The candidate table is read twice (reference checks first, screening
    # second) instead of being held as one list: a large campaign produces six
    # figures of rows and only the rows that pass the screen need to stay in
    # memory for property prediction.
    candidates_path = Path(generation.candidates_path)
    if merge_existing_products and previous_generation_summary:
        _add_folder_generation_totals(
            Path(generation.summary_path), previous_generation_summary
        )
    iteration_selection: dict[str, object] | None = None
    iteration_selected_smiles: set[str] | None = None
    if iteration_model_path is not None:
        round_index = int(iteration_round or 1)
        _panel_stage(
            "route",
            "Round %d: scoring this round's products with the iteration model"
            % round_index,
        )
        iteration_seed_rows = _read_rows(Path(generation.seed_manifest_path))
        iteration_selection = _score_iteration_candidates(
            candidates_path,
            parent_index(root, extra_rows=iteration_seed_rows),
            model_path=iteration_model_path,
            budget=int(iteration_budget or 0)
            or max(len(iteration_seed_rows) * len(PERTURBATION_OPERATIONS), 1),
            exploration_fraction=iteration_exploration_fraction,
            round_index=round_index,
            ring_weight=iteration_ring_weight,
        )
        iteration_selected_smiles = set(
            str(smiles) for smiles in iteration_selection.get("selected_smiles") or []
        )
        _panel_stage(
            "route",
            "Round %d: kept %d of %d products (predicted Tier 1: %d, exploration %d)"
            % (
                round_index,
                int(iteration_selection.get("selected") or 0),
                int(iteration_selection.get("evaluated") or 0),
                int(iteration_selection.get("predicted_tier1") or 0),
                int(iteration_selection.get("explored") or 0),
            ),
        )
    if known_fuel_check:
        from known_fuel_library import build_known_fuel_hit_table

        build_known_fuel_hit_table(
            _iter_rows(candidates_path), root / "known_fuel_hits.csv"
        )
    all_candidate_smiles = [
        str(row.get("canonical_smiles") or "") for row in _iter_rows(candidates_path)
    ]
    candidate_smiles = (
        [smiles for smiles in all_candidate_smiles if smiles in iteration_selected_smiles]
        if iteration_selected_smiles is not None
        else all_candidate_smiles
    )
    # Build the screening reference set (known library + generation temporary
    # library + any pre-existing perturbation temporary library) BEFORE the
    # current batch is appended to the perturbation temporary library;
    # otherwise every generated product finds itself in the reference set and
    # is incorrectly rejected as not_novel.
    known_exact = _load_smiles_from_paths(exclude_paths)
    novelty_path = _write_combined_smiles_library(root, exclude_paths)
    if perturbation_temporary_smiles_path:
        pert_temp = _load_smiles_from_paths((perturbation_temporary_smiles_path,))
        for smiles in all_candidate_smiles:
            smiles = smiles.strip()
            if smiles:
                pert_temp.add(smiles)
        pert_temp_path = Path(perturbation_temporary_smiles_path)
        pert_temp_path.parent.mkdir(parents=True, exist_ok=True)
        pert_temp_path.write_text(
            "\n".join(sorted(pert_temp)) + "\n",
            encoding="utf-8",
        )
    heating_values = _load_heating_values(heating_values_csv)
    sta_model_paths = _selected_sta_model_paths(
        sta_property_model_path,
        density_model,
        heating_value_model,
        melting_point_model,
        boiling_point_model,
        flash_point_model,
        formation_enthalpy_model,
        molar_volume_model,
        isp_model,
        sta_density_model_path,
        sta_heating_value_model_path,
        sta_melting_point_model_path,
        sta_boiling_point_model_path,
        sta_flash_point_model_path,
        sta_formation_enthalpy_model_path,
        sta_molar_volume_model_path,
        sta_isp_model_path,
    )
    batch_sta_by_smiles = (
        _batch_sta_predictions_for_smiles(
            candidate_smiles,
            sta_model_paths,
            batch_size=256,
        )
        if sta_model_paths and candidate_smiles
        else {}
    )

    screened_path = root / "perturbation_screened.csv"
    rejected_path = root / "rejected_pre_property.csv"
    passed_rows: list[dict[str, object]] = []
    passed_indices: list[int] = []
    property_rows: list[dict[str, object]] = []
    property_failed_rows: list[dict[str, object]] = []
    screened_stream = _StageTableStream(
        screened_path, merge_existing=merge_existing_products
    )
    rejected_stream = _StageTableStream(
        rejected_path, merge_existing=merge_existing_products
    )

    _panel_stage("screen", "Running dedup, MMFF, and structure screening")
    # The MMFF geometry of a molecule depends only on its SMILES and on the
    # enumeration index of its first occurrence, so it can be produced by worker
    # processes up front. Results are identical to the serial loop below.
    mmff_analyses: dict[str, object] = {}
    pending_mmff: list[tuple[str, int]] = []
    seen_mmff: set[str] = set()
    for index, smiles in enumerate(candidate_smiles, start=1):
        smiles = smiles.strip()
        if not smiles or smiles in seen_mmff:
            continue
        seen_mmff.add(smiles)
        pending_mmff.append((smiles, seed + index))
    if pending_mmff:
        mmff_workers = resolve_worker_count(screen_workers, memory_per_worker_gb=0.4)
        if len(pending_mmff) < 32:
            mmff_workers = 1
        for (smiles, _s), analysis in zip(
            pending_mmff, parallel_map(_perturbation_mmff_worker, pending_mmff, mmff_workers)
        ):
            mmff_analyses[smiles] = analysis
    property_stage_announced = False
    candidate_count = len(candidate_smiles)
    for index, candidate in enumerate(_iter_rows(candidates_path), start=1):
        completed = index - 1
        if completed and completed % 100 == 0:
            if property_stage_announced:
                _panel_stage(
                    "property",
                    f"Screening and scoring perturbed molecules {completed}/{candidate_count}",
                )
            else:
                _panel_stage(
                    "screen",
                    f"Pre-screening perturbed molecules {completed}/{candidate_count}",
                )
        row: dict[str, object] = dict(candidate)
        smiles = str(row.get("canonical_smiles", "")).strip()
        if iteration_selected_smiles is not None and smiles not in iteration_selected_smiles:
            # An iterative round only screens what the iteration model kept.
            continue
        try:
            descriptors = descriptors_from_smiles(smiles)
            row.update(asdict(descriptors))
            row["ring_count_definition"] = RING_COUNT_DEFINITION
            # Perturbed molecules are local modifications of known molecules,
            # so fingerprint novelty is inherently low; skip the ~200 ms
            # nearest-neighbor search against the 810k library per molecule.
            novelty = None
            is_novel = smiles not in known_exact
            high_novelty = None
            mmff = mmff_analyses.get(smiles) or analyze_mmff_geometry(
                smiles, seed=seed + index
            )
            passes, reasons = passes_screen(
                descriptors=descriptors,
                is_connected=True,
                is_novel=is_novel,
                config=screening_config,
                ring_count=descriptors.ring_count,
                mmff=mmff,
            )
            score = fuel_candidate_score(
                descriptors=descriptors,
                is_connected=True,
                is_novel=is_novel,
                config=screening_config,
                ring_count=descriptors.ring_count,
                preferred_ring_score=descriptors.preferred_ring_score,
                strain_ring_score=descriptors.strain_ring_score,
                mmff_stability_score=mmff.mmff_stability_score,
            )
            row.update(
                {
                    "passes_screen": passes,
                    "screen_score": score,
                    "failure_reasons": ";".join(reasons),
                    "is_connected": True,
                    "is_novel": is_novel,
                    "max_known_similarity": "" if novelty is None else novelty.max_similarity,
                    "fingerprint_novelty": "" if novelty is None else novelty.novelty_score,
                    "high_novelty": "" if high_novelty is None else high_novelty,
                    "nearest_known_smiles": "" if novelty is None else novelty.nearest_smiles or "",
                    "mmff_embed_success": mmff.embed_success,
                    "mmff_optimize_success": mmff.optimize_success,
                    "mmff_converged": mmff.mmff_converged,
                    "mmff_energy_kcal_mol": _blank_if_none(mmff.mmff_energy_kcal_mol),
                    "mmff_energy_per_atom": _blank_if_none(mmff.mmff_energy_per_atom),
                    "mmff_energy_per_heavy_atom": _blank_if_none(mmff.mmff_energy_per_heavy_atom),
                    "mmff_stability_score": mmff.mmff_stability_score,
                    "mmff_failure_reason": mmff.mmff_failure_reason,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "passes_screen": False,
                    "screen_score": 0.0,
                    "failure_reasons": f"screening_failed:{exc}",
                }
            )
            rejected_stream.write(row)
            screened_stream.write(row)
            continue

        screened_stream.write(row)
        if row["passes_screen"] is not True:
            rejected_stream.write(row)
            continue
        passed_rows.append(row)
        passed_indices.append(index)

    screened_row_count = screened_stream.close()
    rejected_row_count = rejected_stream.close()
    if prior_candidates_path is not None:
        # The attempt log is the last table to join the earlier campaigns: the
        # screening loop above must only ever see this campaign's own rows.
        kept_candidates = _merge_table_files(candidates_path, prior_candidates_path)
        _panel_stage(
            "generate",
            "Attempt log: %d rows from earlier campaigns, %d from this one"
            % (kept_candidates, generation.generated_rows),
        )
    # Only the rows that continue to property prediction are kept from here on.
    del candidate_smiles

    # Second pass: batch-compute e3nn 3D geometry scores for screened
    # molecules, then fill property predictions and composite scores
    # per molecule (STA is already batched above).
    geometry_scores: dict[str, tuple[GeometryScore, float | None]] = {}
    if e3nn_geometry_checkpoint is not None and passed_rows:
        _panel_stage("property", "Batch-computing e3nn 3D geometry scores")
        geometry_scores = predict_geometry_score_batch(
            e3nn_geometry_checkpoint,
            [str(row.get("canonical_smiles") or "") for row in passed_rows],
            seeds=[seed + index for index in passed_indices],
        )

    # Per-molecule prediction deltas: served from the on-disk cache when the same
    # molecule was already predicted with the same geometry seed, otherwise
    # computed here (optionally on several worker processes).
    # JP-10 reference for the channel pools, scored with the same models
    # that score the candidates. Failure here must not stop the run.
    _jp10_benchmark: object | None = None
    try:
        _jp10_benchmark = benchmark_from_models(
            sta_density_model_path=str(sta_density_model_path),
            sta_formation_enthalpy_model_path=str(
                sta_formation_enthalpy_model_path or sta_property_model_path
            ),
            phase_fusion_melting_point_model_path=(
                None if phase_fusion_melting_point_model_path is None
                else str(phase_fusion_melting_point_model_path)
            ),
            phase_fusion_boiling_point_model_path=(
                None if phase_fusion_boiling_point_model_path is None
                else str(phase_fusion_boiling_point_model_path)
            ),
            phase_fusion_flash_point_model_path=(
                None if phase_fusion_flash_point_model_path is None
                else str(phase_fusion_flash_point_model_path)
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _panel_stage("property", "JP-10 tier benchmark unavailable: %s" % exc)

    _predict_kwargs: dict[str, object] = dict(
        heating_values=heating_values,
        mp_model_path=mp_model_path,
        mp_data_path=mp_data_path,
        e3nn_geometry_checkpoint=e3nn_geometry_checkpoint,
        e3nn_property_checkpoint=e3nn_property_checkpoint,
        e3nn_property_name=e3nn_property_name,
        sta_property_model_path=sta_property_model_path,
        sta_density_model_path=sta_density_model_path,
        sta_heating_value_model_path=sta_heating_value_model_path,
        sta_melting_point_model_path=sta_melting_point_model_path,
        sta_boiling_point_model_path=sta_boiling_point_model_path,
        sta_flash_point_model_path=sta_flash_point_model_path,
        sta_formation_enthalpy_model_path=sta_formation_enthalpy_model_path,
        sta_molar_volume_model_path=sta_molar_volume_model_path,
        sta_isp_model_path=sta_isp_model_path,
        density_model=density_model,
        heating_value_model=heating_value_model,
        melting_point_model=melting_point_model,
        boiling_point_model=boiling_point_model,
        flash_point_model=flash_point_model,
        formation_enthalpy_model=formation_enthalpy_model,
        molar_volume_model=molar_volume_model,
        isp_model=isp_model,
        cbh_bdc_model_path=cbh_bdc_model_path,
        cbh_bdc_feature_csv=cbh_bdc_feature_csv,
        cbh_bdc_yang_root=cbh_bdc_yang_root,
        phase_fusion_melting_point_model_path=phase_fusion_melting_point_model_path,
        phase_fusion_boiling_point_model_path=phase_fusion_boiling_point_model_path,
        phase_fusion_flash_point_model_path=phase_fusion_flash_point_model_path,
        use_cbh_bdc_heating=use_cbh_bdc_heating,
        jp10_benchmark=_jp10_benchmark,
        reproducibility_scope=reproducibility_scope,
    )
    _cache_obj: PropertyCache | None = None
    if use_property_cache:
        _cache_obj = PropertyCache(
            Path(property_cache_path)
            if property_cache_path
            else Path(output_dir) / "property_prediction_cache.pkl",
            config_fingerprint(_predict_kwargs, seed=seed),
        )
    deltas_by_pos: dict[int, dict[str, object]] = {}
    errors_by_pos: dict[int, tuple[str, dict[str, object]]] = {}
    payloads: list[tuple[dict[str, object], int, object, object]] = []
    queue_keys: list[tuple[int, str, int]] = []
    for index, row in enumerate(passed_rows, start=1):
        position = index - 1
        smiles = str(row.get("canonical_smiles") or "").strip()
        row_seed = seed + passed_indices[position]
        if _cache_obj is not None:
            cached = _cache_obj.get(smiles, row_seed)
            if cached is not None:
                deltas_by_pos[position] = cached
                continue
        payloads.append(
            (
                dict(row),
                passed_indices[position],
                batch_sta_by_smiles.get(smiles),
                geometry_scores,
            )
        )
        queue_keys.append((position, smiles, row_seed))
    if payloads:
        _workers = resolve_worker_count(property_workers, memory_per_worker_gb=1.3)
        if len(payloads) < 16:
            _workers = 1
        _context = dict(_predict_kwargs)
        _context["__seed__"] = seed
        _context["__workers__"] = _workers
        if _workers > 1:
            _panel_stage(
                "property",
                "Predicting properties for %d perturbed molecules on %d worker processes"
                % (len(payloads), _workers),
            )
            _outcomes = _parallel_property_deltas(payloads, _context, _workers)  # type: ignore[arg-type]
        else:
            _property_worker_init(_context)
            _outcomes = [_property_delta_worker(payload) for payload in payloads]  # type: ignore[arg-type]
        for (position, smiles, row_seed), (status, payload, partial) in zip(
            queue_keys, _outcomes
        ):
            if status == "ok":
                deltas_by_pos[position] = payload  # type: ignore[assignment]
                if _cache_obj is not None:
                    _cache_obj.put(smiles, row_seed, payload)  # type: ignore[arg-type]
            else:
                errors_by_pos[position] = (str(payload), partial)  # type: ignore[assignment]
        if _cache_obj is not None:
            _cache_obj.save(force=True)

    property_stage_announced = False
    for index, row in enumerate(passed_rows, start=1):
        enriched = dict(row)
        smiles = str(enriched.get("canonical_smiles") or "").strip()
        try:
            if not property_stage_announced:
                _panel_stage("property", "Computing perturbed molecule properties and composite scores")
                property_stage_announced = True
            _delta = deltas_by_pos.get(index - 1)
            if _delta is None:
                _message, _partial = errors_by_pos.get(
                    index - 1, ("unknown property failure", {})
                )
                enriched.update(_partial)
                raise RuntimeError(_message)
            enriched.update(_delta)
            _route_enriched_property_row(
                enriched,
                uncertainty_scale=uncertainty_scale,
                thresholds=thresholds,
                ignore_density=ignore_density or density_model == "none",
                ignore_heating_value=ignore_heating_value
                or heating_value_model == "none",
                use_melting_point_screen=(
                    use_melting_point_screen and melting_point_model != "none"
                ),
                use_boiling_point_screen=(
                    use_boiling_point_screen and boiling_point_model != "none"
                ),
                use_flash_point_screen=(
                    use_flash_point_screen and flash_point_model != "none"
                ),
                use_volumetric_heating_value_screen=(
                    use_volumetric_heating_value_screen
                ),
            )
            _add_objective_scores(
                enriched,
                objective_mode="property_maximize",
                thresholds=thresholds,
                score_weights=score_weights,
                property_score_weights=property_score_weights,
                property_targets=PropertyTargetValues(),
                objective_relative_window=0.15,
                missing_property_penalty=missing_property_penalty,
            )
            if _jp10_benchmark is not None:
                apply_tier_fields(enriched, _jp10_benchmark)  # type: ignore[arg-type]
            property_rows.append(enriched)
        except Exception as exc:
            enriched["property_stage_status"] = "property_failed"
            enriched["property_failure_reason"] = str(exc)
            property_failed_rows.append(enriched)

    if candidate_count:
        _panel_stage(
            "property" if property_stage_announced else "screen",
        f"Perturbation candidates processed {candidate_count}/{candidate_count}",
        )
    # ---- earlier campaigns in this run folder join the new products ---------
    carried_rows: list[dict[str, object]] = []
    carried_rescore_failures = 0
    if merge_existing_products and property_path.exists():
        _panel_stage(
            "route",
            "Merging the perturbation products already stored in this folder",
        )
        carried_rows = _carried_product_rows(
            property_path,
            exclude_smiles={
                str(row.get("canonical_smiles") or "").strip() for row in property_rows
            },
        )
        for row in carried_rows:
            if not _rescore_carried_product_row(
                row,
                uncertainty_scale=uncertainty_scale,
                thresholds=thresholds,
                ignore_density=ignore_density or density_model == "none",
                ignore_heating_value=ignore_heating_value
                or heating_value_model == "none",
                use_melting_point_screen=(
                    use_melting_point_screen and melting_point_model != "none"
                ),
                use_boiling_point_screen=(
                    use_boiling_point_screen and boiling_point_model != "none"
                ),
                use_flash_point_screen=(
                    use_flash_point_screen and flash_point_model != "none"
                ),
                use_volumetric_heating_value_screen=(
                    use_volumetric_heating_value_screen
                ),
                score_weights=score_weights,
                property_score_weights=property_score_weights,
                missing_property_penalty=missing_property_penalty,
                benchmark=_jp10_benchmark,
            ):
                carried_rescore_failures += 1
        if carried_rows:
            property_rows = carried_rows + property_rows
            _panel_stage(
                "route",
                "Kept %d products from the earlier campaign(s); %d new products"
                % (len(carried_rows), len(property_rows) - len(carried_rows)),
            )

    _panel_stage("route", "Classifying perturbation products into the four tiers")
    missing_property_rows = [
        row
        for row in property_rows
        if row.get("hard_threshold_verdict") == "missing_properties"
    ]
    sta_weak_feedback_rows = (
        build_sta_weak_feedback_rows(
            property_rows,
            thresholds=thresholds,
            novelty_threshold=novelty_threshold,
            base_label_weight=sta_weak_feedback_weight,
        )
        if sta_weak_feedback_enabled
        else []
    )

    _write_stage_table(
        property_path,
        public_rows(property_rows),
        merge_existing=merge_existing_products,
    )
    _write_stage_table(
        sta_weak_feedback_path,
        sta_weak_feedback_rows,
        merge_existing=merge_existing_products,
        fieldnames=STA_FEEDBACK_COLUMNS,
    )

    # ---- one table per channel: Tier 1-4 and Potential 1-3 ------------------
    def _tier_order(row: dict[str, object]) -> tuple[int, float]:
        try:
            beats = int(row.get("jp10_beat_count") or 0)
        except (TypeError, ValueError):
            beats = 0
        try:
            score = float(row.get("composite_ranking_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return (-beats, -score)

    tier_paths: dict[str, Path] = {}
    grouped_tiers: dict[str, list[dict[str, object]]] = {
        channel: [] for channel in CHANNEL_ORDER
    }
    if any("tier" in row for row in property_rows):
        for row in property_rows:
            # Only a molecule that carries a channel is pooled: a data gap, a
            # failed prediction or a molecule that misses thresholds and beats
            # JP-10 on nothing has no pool and must not be counted as Tier 4.
            channel = str(row.get("tier") or "").strip().lower()
            if channel in CHANNEL_FILENAMES:
                grouped_tiers[channel].append(row)
        # Every channel table keeps the columns of the property table, so an
        # empty channel is still a readable table with its header.
        channel_fieldnames = _fieldnames_of(public_rows(property_rows))
        for channel in CHANNEL_ORDER:
            tier_paths[channel] = root / CHANNEL_FILENAMES[channel]
            _write_rows(
                tier_paths[channel],
                public_rows(sorted(grouped_tiers.get(channel, []), key=_tier_order)),
                channel_fieldnames,
            )
        _panel_stage(
            "route",
            "Channels: T1 %d / T2 %d / T3 %d / T4 %d | Pot1 %d / Pot2 %d / Pot3 %d"
            % (
                len(grouped_tiers.get(TIER1, [])),
                len(grouped_tiers.get(TIER2, [])),
                len(grouped_tiers.get(TIER3, [])),
                len(grouped_tiers.get(TIER4, [])),
                len(grouped_tiers.get(POTENTIAL1, [])),
                len(grouped_tiers.get(POTENTIAL2, [])),
                len(grouped_tiers.get(POTENTIAL3, [])),
            ),
        )

    _write_stage_table(
        property_failed_path,
        public_rows(property_failed_rows),
        merge_existing=merge_existing_products,
    )
    _write_stage_table(
        missing_property_path,
        public_rows(missing_property_rows),
        merge_existing=merge_existing_products,
    )
    # Use the seeds this run actually perturbed (the preparation manifest already
    # applies --max-seeds), so a limited run does not list untouched seeds.
    original_seed_rows = list(
        require_perturbation_seed_rows(
            Path(generation.seed_manifest_path), thresholds=thresholds
        )
    )
    if merge_existing_products and previous_seed_rows:
        # The manifest of a run folder records every seed that has been
        # perturbed in it, so the wrapper tracks the union of the campaigns and
        # the manual-review pool never loses the seeds of an earlier campaign.
        merged_seed_rows: list[dict[str, object]] = []
        seen_seed_smiles: set[str] = set()
        for row in list(previous_seed_rows) + list(original_seed_rows):
            smiles = str(row.get("canonical_smiles") or "").strip()
            if not smiles or smiles in seen_seed_smiles:
                continue
            seen_seed_smiles.add(smiles)
            merged_seed_rows.append(row)
        _write_rows(Path(generation.seed_manifest_path), merged_seed_rows)
        original_seed_rows = merged_seed_rows
    manual_review_rows = _combine_manual_review_candidates(
        original_seed_rows,
        grouped_tiers.get("tier1", [])
        + grouped_tiers.get("tier2", [])
        + grouped_tiers.get("tier3", []),
    )
    _write_rows(manual_review_path, manual_review_rows)
    # Per-group copies of every stage table were dropped on 2026-09-22: the
    # run-level tables below carry the same rows (and their group key).

    result = PerturbationPipelineResult(
        attempted_samples=samples,
        generated_rows=generation.generated_rows,
        screened_rows=screened_row_count,
        passed_screen_rows=len(passed_rows),
        rejected_pre_property_rows=rejected_row_count,
        property_prediction_rows=len(property_rows),
        property_failed_rows=len(property_failed_rows),
        tier1_rows=len(grouped_tiers.get("tier1", [])),
        tier2_rows=len(grouped_tiers.get("tier2", [])),
        tier3_rows=len(grouped_tiers.get("tier3", [])),
        tier4_rows=len(grouped_tiers.get("tier4", [])),
        potential1_rows=len(grouped_tiers.get(POTENTIAL1, [])),
        potential2_rows=len(grouped_tiers.get(POTENTIAL2, [])),
        potential3_rows=len(grouped_tiers.get(POTENTIAL3, [])),
        carried_over_product_rows=len(carried_rows),
        merged_product_rows=len(property_rows),
        replaced_existing_products=not merge_existing_products,
        iteration_round=int(iteration_round or 0),
        iteration_evaluated_rows=int(
            (iteration_selection or {}).get("evaluated") or 0
        ),
        iteration_selected_rows=int((iteration_selection or {}).get("selected") or 0),
        iteration_predicted_tier1_rows=int(
            (iteration_selection or {}).get("predicted_tier1") or 0
        ),
        missing_property_rows=len(missing_property_rows),
        manual_review_rows=len(manual_review_rows),
        sta_weak_feedback_rows=len(sta_weak_feedback_rows),
        output_dir=str(root),
        candidates_path=generation.candidates_path,
        screened_path=str(screened_path),
        property_predictions_path=str(property_path),
        sta_weak_feedback_path=str(sta_weak_feedback_path),
        rejected_pre_property_path=str(rejected_path),
        property_failed_path=str(property_failed_path),
        missing_property_path=str(missing_property_path),
        manual_review_path=str(manual_review_path),
        summary_path=str(summary_path),
    )
    summary_path.write_text(
        json.dumps(
            {
                **result.as_dict(),
                "ring_count_definition": RING_COUNT_DEFINITION,
                "novelty_threshold": novelty_threshold,
                "uncertainty_scale": uncertainty_scale,
                "ignore_density": ignore_density,
                "ignore_heating_value": ignore_heating_value,
                "known_smiles_path": "" if known_smiles_path is None else str(known_smiles_path),
                "temporary_smiles_path": "" if temporary_smiles_path is None else str(temporary_smiles_path),
                "operations": list(operations),
                "samples_per_seed_operation": samples_per_seed_operation,
                "use_melting_point_screen": use_melting_point_screen,
                "use_boiling_point_screen": use_boiling_point_screen,
                "use_flash_point_screen": use_flash_point_screen,
                "use_volumetric_heating_value_screen": (
                    use_volumetric_heating_value_screen
                ),
                "missing_property_penalty": missing_property_penalty,
                "composite_score_weights": score_weights.normalized().as_dict(),
                "property_score_weights": property_score_weights.as_dict(),
                "carried_over_rescore_failures": carried_rescore_failures,
                "merge_rule": (
                    "products of every campaign in this output folder are kept in "
                    "one set of tables"
                    if merge_existing_products
                    else "this campaign replaced the products stored in the output folder"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _panel_stage("done", "Perturbation pipeline completed")
    return result


#: channels that still have room to improve, so an iteration perturbs them
#: Every Potential pool is a seed source: the three potential channels are the
#: same molecules the older two-pool split held, only named by how many hard
#: thresholds they miss, so a finished run stays reproducible round for round.
ITERATION_SEED_POOLS = (TIER2, TIER3, POTENTIAL1, POTENTIAL2, POTENTIAL3)
ITERATION_SEED_TABLE = "perturbation_seeds_iteration.csv"
ITERATION_SUMMARY_FILE = "iteration_summary.csv"

#: Controlled ablation of the iterative line. Every arm starts from the same
#: lead list and uses the same rounds, attempts and budget; only the named factor
#: changes:
#:   A - property ranking only, the original six operators, fixed split
#:   B - A plus the ring reward / strain penalty in the ranking
#:   C - B plus the ring-expansion operator
#:   D - C plus the per-round adaptive operator split
#:   E - ring expansion only (the whole per-molecule budget goes to it)
ABLATION_ARMS: dict[str, dict[str, object]] = {
    "A": {"ring_weight": 0.0, "ring_expand": False, "adaptation": 0.0,
          "split": {"bond_rewire": 2, "ch2_bridge_move": 2, "local_skeleton_rearrange": 1,
                    "methyl_shift": 4, "methylation": 0, "ring_adjust": 1}},
    "B": {"ring_weight": 1.0, "ring_expand": False, "adaptation": 0.0,
          "split": {"bond_rewire": 2, "ch2_bridge_move": 2, "local_skeleton_rearrange": 1,
                    "methyl_shift": 4, "methylation": 0, "ring_adjust": 1}},
    "C": {"ring_weight": 1.0, "ring_expand": True, "adaptation": 0.0, "split": None},
    "D": {"ring_weight": 1.0, "ring_expand": True, "adaptation": 0.5, "split": None},
    "E": {"ring_weight": 1.0, "ring_expand": True, "adaptation": 0.0,
          "split": {"ring_expand": 10}},
}


def _iteration_seed_rows(
    root: Path,
    pools: tuple[str, ...],
    *,
    already_perturbed: set[str] | None = None,
) -> list[dict[str, object]]:
    """The seed list of one round: the leads that still wait for their turn.

    A lead that has already been perturbed in this run folder is **removed** from
    the candidate list, so the list behaves as a work queue: the products that
    land in Tier 2/3 or in Potential 1-3 join it, the leads that were used leave
    it. A molecule that reaches Tier 1 never enters it, because it is no longer a
    lead to push - which is the goal.
    """

    collected: list[dict[str, object]] = []
    seen: set[str] = set()
    already = {str(smiles) for smiles in (already_perturbed or set())}
    for channel in pools:
        rows = _read_rows(root / CHANNEL_FILENAMES[channel])
        for row in rows:
            smiles = str(row.get("canonical_smiles") or "").strip()
            if not smiles or smiles in seen or smiles in already:
                continue
            seen.add(smiles)
            collected.append({**row, "seed_source_pool": channel})
    # Closest to Tier 1 first, so a capped round works on the best leads.
    collected.sort(
        key=lambda row: (
            -int(_optional_float(row.get("jp10_beat_count")) or 0),
            -(_optional_float(row.get("composite_ranking_score")) or 0.0),
            str(row.get("canonical_smiles") or ""),
        )
    )
    return collected


def _iteration_round_statistics(root: Path, round_index: int) -> dict[str, object]:
    """Channel split of one round's products plus the folder's Tier 1 total."""

    counts: dict[str, int] = {channel: 0 for channel in CHANNEL_ORDER}
    products = 0
    for row in _iter_rows(root / "perturbation_property_predictions.csv"):
        if str(row.get("iteration_round") or "").strip() != str(round_index):
            continue
        channel = str(row.get("tier") or "").strip().lower()
        if channel in counts:
            counts[channel] += 1
        products += 1
    tier1_total = len(_read_rows(root / CHANNEL_FILENAMES[TIER1]))
    return {"products": products, "tier1_total": tier1_total, **counts}


def _iteration_duplicate_rejections(root: Path, round_index: int) -> int:
    """Products of one round that the novelty screen threw out.

    The screen checks every product against the known library plus both
    temporary libraries, so a non-zero count is the visible proof that
    deduplication ran for this round instead of the products slipping through.
    """

    rejected = 0
    for row in _iter_rows(root / "perturbation_screened.csv"):
        if str(row.get("iteration_round") or "").strip() != str(round_index):
            continue
        if str(row.get("passes_screen")).strip().lower() in {"true", "1"}:
            continue
        reasons = str(row.get("failure_reasons") or "").lower()
        if "novel" in reasons or "duplicate" in reasons or "known" in reasons:
            rejected += 1
    return rejected


def _iteration_operator_gain(root: Path, round_index: int) -> dict[str, float]:
    """Realised quality of each operator in one round.

    The value the adaptive allocation learns from is the same one the design
    cares about: how many JP-10 properties the products beat, plus the ring
    reward and minus the strained-ring penalty. It is measured on the products of
    the round, not predicted, so the allocation follows what actually came out.
    """

    from perturbation_iteration import RING_REWARD_FACTOR, RING_STRAIN_FACTOR, ring_preference

    totals: dict[str, float] = {operation: 0.0 for operation in PERTURBATION_OPERATIONS}
    counts: dict[str, int] = {operation: 0 for operation in PERTURBATION_OPERATIONS}
    for row in _iter_rows(root / "perturbation_property_predictions.csv"):
        if str(row.get("iteration_round") or "").strip() != str(round_index):
            continue
        operation = str(row.get("operation") or "").strip()
        if operation not in totals:
            continue
        reward, penalty, _strained, _preferred = ring_preference(
            str(row.get("canonical_smiles") or "")
        )
        beats = _optional_float(row.get("jp10_beat_count")) or 0.0
        totals[operation] += (
            beats + RING_REWARD_FACTOR * reward - RING_STRAIN_FACTOR * penalty
        )
        counts[operation] += 1
    return {
        operation: (totals[operation] / counts[operation]) if counts[operation] else 0.0
        for operation in PERTURBATION_OPERATIONS
    } | {"__counts__": counts}


def _adapt_operator_weights(
    weights: dict[str, float],
    realised: dict[str, object],
    *,
    rate: float,
    floor_fraction: float = 0.05,
) -> dict[str, float]:
    """Shift the sampling allocation towards the operators that delivered.

    A multiplicative update: every operator keeps its weight and gains or loses a
    factor from the normalised quality it produced last round. The floor keeps
    every operator reachable, so a round can always rediscover a good move.
    """

    rate = min(max(float(rate), 0.0), 1.0)
    if rate <= 0.0:
        return dict(weights)
    values = {op: float(realised.get(op) or 0.0) for op in PERTURBATION_OPERATIONS}
    spread = max(max(values.values()) - min(values.values()), 1e-9)
    updated = {
        op: max(weights.get(op, 0.0), 0.0)
        * float(np.exp(rate * (values[op] - float(np.mean(list(values.values())))) / spread))
        for op in PERTURBATION_OPERATIONS
    }
    total = sum(updated.values()) or 1.0
    uniform = 1.0 / len(PERTURBATION_OPERATIONS)
    blended = {
        op: (updated[op] / total) * (1.0 - floor_fraction) + floor_fraction * uniform
        for op in PERTURBATION_OPERATIONS
    }
    total = sum(blended.values()) or 1.0
    return {op: blended[op] / total for op in PERTURBATION_OPERATIONS}


def _write_iteration_seed_table(
    root: Path,
    seeds: list[dict[str, object]],
    round_index: int,
) -> Path:
    """Write one round's seeds to a scratch table the pipeline can read.

    The table lives in the run folder's ``_iteration`` scratch directory only for
    the duration of the round: the run folder keeps the seeds that were really
    perturbed in ``perturbation_seeds.csv`` and the round number on every product
    row.  Each seed row also carries ``seed_perturbation_round`` so the merged
    ``perturbation_seeds.csv`` records *which* round perturbed a lead, which is
    not the same thing as the ``iteration_round`` column inherited from the
    product table (that one records when the molecule was produced).
    """

    from finalization import mark_perturbation_seed_rows

    scratch = root / "_iteration"
    scratch.mkdir(parents=True, exist_ok=True)
    table = scratch / ITERATION_SEED_TABLE
    marked = mark_perturbation_seed_rows(
        [{**row, "seed_perturbation_round": int(round_index)} for row in seeds],
        selection_basis="iteration_pool",
    )
    _write_rows(table, list(marked))
    return table


def default_iteration_model_path() -> Path:
    from fuel_design.paths import DEFAULT_PERTURBATION_ITERATION_MODEL_PATH

    return Path(DEFAULT_PERTURBATION_ITERATION_MODEL_PATH)


def train_iteration_model_from_run(
    output_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    model_family: str | None = None,
) -> dict[str, object]:
    """Pre-train the iteration model from the data a run folder already holds.

    The metrics sidecar defaults to ``<model>.json``, so an experimental model
    writes next to itself and the production sidecar is only replaced when the
    production model itself is retrained.
    """

    from perturbation_iteration import build_iteration_dataset, train_iteration_model

    dataset = build_iteration_dataset(output_dir)
    target = Path(model_path or default_iteration_model_path())
    reference = JP10Reference.from_mapping(
        {**dataset["reference"], "source": dataset["reference_source"]}
    )
    summary = train_iteration_model(
        dataset["rows"],
        target,
        reference=reference,
        model_family=model_family or "hist_gradient_boosting",
        metrics_path=Path(metrics_path) if metrics_path else target.with_suffix(".json"),
        metadata={
            "output_dir": str(output_dir),
            "reference_source": dataset["reference_source"],
            "sample_count": dataset["sample_count"],
            "skipped_missing_parent": dataset["skipped_missing_parent"],
        },
    )
    summary["dataset"] = {
        key: value for key, value in dataset.items() if key != "rows"
    }
    return summary


def compare_iteration_models_from_run(
    output_dir: str | Path,
    *,
    folds: int = 5,
    top: int = 3,
    model_dir: str | Path | None = None,
    save_best: bool = True,
    model_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Cross-validate the candidate iteration models and keep the best one.

    Every family is fitted on the whole dataset afterwards and stored next to the
    production model, so any of them can be selected later with
    ``--iteration-model``; the winner is additionally written to the production
    path. The comparison table is what the choice is made from.
    """

    from perturbation_iteration import (
        build_iteration_dataset,
        evaluate_iteration_models,
        train_iteration_model,
    )

    dataset = build_iteration_dataset(output_dir)
    reference = JP10Reference.from_mapping(
        {**dataset["reference"], "source": dataset["reference_source"]}
    )
    comparison = evaluate_iteration_models(
        dataset["rows"],
        reference=reference,
        folds=folds,
        model_names=model_names,
    )
    root = Path(model_dir or default_iteration_model_path().parent)
    candidates_dir = root / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    stored: dict[str, str] = {}
    for entry in comparison["models"]:
        family = str(entry["model_family"])
        path = candidates_dir / ("%s.joblib" % family)
        train_iteration_model(
            dataset["rows"],
            path,
            reference=reference,
            model_family=family,
            metadata={
                "output_dir": str(output_dir),
                "reference_source": dataset["reference_source"],
                "comparison_rank_score": entry["rank_score"],
            },
        )
        stored[family] = str(path)
    best = comparison["models"][0] if comparison["models"] else None
    production_path = ""
    if save_best and best is not None:
        family = str(best["model_family"])
        production_path = str(default_iteration_model_path())
        train_iteration_model(
            dataset["rows"],
            production_path,
            reference=reference,
            model_family=family,
            metrics_path=str(Path(production_path).with_suffix(".json")),
            metadata={
                "output_dir": str(output_dir),
                "reference_source": dataset["reference_source"],
                "selected_by": "compare-perturbation-iteration-models",
                "rank_score": best["rank_score"],
            },
        )
    comparison["candidate_model_paths"] = stored
    comparison["production_model_path"] = production_path
    comparison["top"] = [
        {
            "model_family": entry["model_family"],
            "rank_score": entry["rank_score"],
            "mean_relative_gain": entry["mean_relative_gain"],
            "beat_exact_accuracy": entry["beat_exact_accuracy"],
            "tier1_precision": entry["tier1_precision"],
            "path": stored.get(str(entry["model_family"]), ""),
        }
        for entry in comparison["models"][: max(int(top), 1)]
    ]
    report_dir = Path(model_dir or default_iteration_model_path().parent)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "iteration_model_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_rows(
        report_dir / "iteration_model_comparison.csv",
        [
            {
                "model_family": entry["model_family"],
                "rank_score": round(float(entry["rank_score"]), 4),
                "mean_relative_gain": round(float(entry["mean_relative_gain"]), 4),
                "mean_sign_agreement": round(float(entry["mean_sign_agreement"]), 4),
                "beat_exact_accuracy": round(float(entry["beat_exact_accuracy"]), 4),
                "beats_mae": round(float(entry["beats_mae"]), 4),
                "tier1_precision": round(float(entry["tier1_precision"]), 4),
                "tier1_recall": round(float(entry["tier1_recall"]), 4),
                "predicted_tier1_rows": entry["predicted_tier1_rows"],
                "actual_tier1_rows": entry["actual_tier1_rows"],
                "density_mae": round(float(entry["properties"]["margin_density"]["test_mae"]), 5),
                "heating_mae": round(float(entry["properties"]["margin_heating_value"]["test_mae"]), 5),
                "melting_mae": round(float(entry["properties"]["margin_melting_point"]["test_mae"]), 4),
                "fit_seconds": entry["fit_seconds"],
                "predict_seconds": entry["predict_seconds"],
                "candidate_path": stored.get(str(entry["model_family"]), ""),
            }
            for entry in comparison["models"]
        ],
    )
    return comparison


def _ensure_iteration_pool_consistency(root: Path) -> dict[str, int]:
    """Make sure every lead in the channel tables has a property row.

    A round writes the channel tables from the rows it carried over, so a lead
    that sits in a channel table but has no property row would silently drop out
    of the seed list. A folder straight from the pipeline always agrees; this
    repairs or reports the cases that do not (a folder assembled by hand, or one
    whose property table was removed).
    """

    pool_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for channel in CHANNEL_ORDER:
        for row in _read_rows(root / CHANNEL_FILENAMES[channel]):
            smiles = str(row.get("canonical_smiles") or "").strip()
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            pool_rows.append(row)
    property_path = root / "perturbation_property_predictions.csv"
    known = {
        str(row.get("canonical_smiles") or "").strip()
        for row in _iter_rows(property_path)
    }
    known.discard("")
    if not property_path.exists():
        if pool_rows:
            _write_rows(property_path, pool_rows)
        return {"restored_property_rows": len(pool_rows), "missing_property_rows": 0}
    missing = [smiles for smiles in seen if smiles not in known]
    if missing:
        raise ValueError(
            "%d lead(s) sit in the channel tables of %s but have no row in %s. "
            "Run organize-run-outputs (it re-derives both from the stored "
            "predictions) before an iterative run."
            % (len(missing), root, property_path.name)
        )
    return {"restored_property_rows": 0, "missing_property_rows": 0}


def run_iterative_perturbation(
    output_dir: str | Path,
    *,
    iterations: int = 3,
    pools: tuple[str, ...] = ITERATION_SEED_POOLS,
    iteration_model_path: str | Path | None = None,
    per_round_budget: int = 2000,
    max_seeds_per_round: int | None = 300,
    samples_per_seed_operation: int = 10,
    attempts_per_molecule: int | None = None,
    operator_attempts: Mapping[str, int] | None = None,
    operator_adaptation_rate: float = 0.5,
    ablation_arm: str | None = None,
    rounds_without_improvement: int = 0,
    **pipeline_kwargs: object,
) -> dict[str, object]:
    """Iterative perturbation towards Tier 1.

    Every round perturbs the leads that are still outside Tier 1, keeps the
    products the pre-trained property-change model expects to land closest to (or
    inside) Tier 1, screens and predicts those, and merges the outcome into the
    run folder's channel tables. The next round continues from what is still not
    Tier 1, so each round climbs from the previous one instead of repeating it.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    unknown = [channel for channel in pools if channel not in CHANNEL_ORDER]
    if unknown:
        raise ValueError("unknown iteration seed pools: %s" % ", ".join(unknown))
    consistency = _ensure_iteration_pool_consistency(root)
    model_path = Path(iteration_model_path or default_iteration_model_path())
    model_summary: dict[str, object] | None = None
    if not iteration_model_contract_is_current(model_path):
        model_summary = train_iteration_model_from_run(root, model_path=model_path)
    # Deduplication is only real when the two reference libraries travel with the
    # round; the summary records them so a missing library cannot go unnoticed.
    reference_libraries = {
        key: str(pipeline_kwargs.get(key) or "")
        for key in (
            "known_smiles_path",
            "temporary_smiles_path",
            "perturbation_temporary_smiles_path",
        )
    }
    reference_state = check_reference_libraries(
        known_smiles_path=reference_libraries["known_smiles_path"],
        temporary_smiles_path=reference_libraries["temporary_smiles_path"],
        perturbation_temporary_smiles_path=reference_libraries[
            "perturbation_temporary_smiles_path"
        ],
        products_already_present=(root / "perturbed_candidates.csv").exists(),
    )
    missing_reference = reference_state["missing"]
    # The model is static: a round never re-trains it, it only uses what it learnt
    # to shape where and how densely the next round samples.
    artifact = load_iteration_model(model_path)
    weights = artifact.get("operator_weights") or {}
    # The command line already carries the operator list; the arm narrows it
    # instead of passing the argument twice.
    cli_operations = pipeline_kwargs.pop("operations", None)
    generation_operations = tuple(cli_operations) if cli_operations else PERTURBATION_OPERATIONS
    ring_weight = 1.0
    if ablation_arm:
        arm = ABLATION_ARMS.get(str(ablation_arm).upper())
        if arm is None:
            raise ValueError(
                "unknown ablation arm %s (available: %s)"
                % (ablation_arm, ", ".join(sorted(ABLATION_ARMS)))
            )
        ring_weight = float(arm["ring_weight"])
        operator_adaptation_rate = float(arm["adaptation"])
        if not arm["ring_expand"]:
            generation_operations = tuple(
                op for op in generation_operations if op != "ring_expand"
            )
        if arm["split"]:
            operator_attempts = dict(arm["split"])  # type: ignore[arg-type]
        _panel_stage(
            "generate",
            "Ablation arm %s: ring weight %.1f, ring_expand %s, adaptation %.1f, split %s"
            % (
                ablation_arm.upper(),
                ring_weight,
                "on" if arm["ring_expand"] else "off",
                operator_adaptation_rate,
                operator_attempts or "model default",
            ),
        )
    if operator_attempts:
        # The panel/CLI can start from its own operator split; the rounds then
        # adapt it from what they measured.
        weights = {
            operation: max(float(operator_attempts.get(operation, 0) or 0), 0.0)
            for operation in PERTURBATION_OPERATIONS
        }
    attempts = attempts_per_operation(
        weights,
        attempts_per_seed_operation=int(samples_per_seed_operation),
        total_attempts_per_molecule=attempts_per_molecule,
    )
    _panel_stage(
        "generate",
        "Iteration plan: %s perturbation(s) per molecule, split per operator %s "
        "(model is static, rounds only re-sample)"
        % (
            sum(attempts.values()),
            ", ".join("%s=%d" % (key, attempts[key]) for key in sorted(attempts)),
        ),
    )
    baseline_tier1 = len(_read_rows(root / CHANNEL_FILENAMES[TIER1]))
    # Rounds continue across campaigns: a second campaign in the same folder
    # numbers its rounds after the first one and appends to the round table
    # instead of overwriting it, so per-round analysis stays unambiguous.
    previous_summaries = _read_rows(root / ITERATION_SUMMARY_FILE)
    used_rounds: list[int] = []
    for row in previous_summaries:
        try:
            used_rounds.append(int(row.get("round") or 0))
        except (TypeError, ValueError):
            continue
    first_round = (max(used_rounds) + 1) if used_rounds else 1
    summaries: list[dict[str, object]] = list(previous_summaries)
    rounds_without_gain = 0
    stop_reason = "iterations_completed"
    for round_index in range(first_round, first_round + max(int(iterations), 0)):
        # The manifest records every seed this folder has perturbed, so a round
        # can tell the fresh leads from the ones that already had their turn.
        already_perturbed = {
            str(row.get("canonical_smiles") or "").strip()
            for row in _read_rows(root / "perturbation_seeds.csv")
        }
        # The manifest can be truncated by an interrupted round, so the attempt
        # log is the reliable record: every row names the parent it perturbed.
        for row in _iter_rows(root / "perturbed_candidates.csv"):
            parent = str(row.get("parent_canonical_smiles") or "").strip()
            if parent:
                already_perturbed.add(parent)
        already_perturbed.discard("")
        seeds = _iteration_seed_rows(root, pools, already_perturbed=already_perturbed)
        seed_list_size = len(seeds)
        fresh_seeds = sum(
            1
            for row in seeds
            if str(row.get("canonical_smiles") or "").strip() not in already_perturbed
        )
        if max_seeds_per_round is not None:
            seeds = seeds[: max(int(max_seeds_per_round), 0)]
        if not seeds:
            stop_reason = "no_seeds_left"
            break
        _panel_stage(
            "generate",
            "Iteration %d: seed list holds %d leads (%d never perturbed); perturbing %d"
            % (round_index, seed_list_size, fresh_seeds, len(seeds)),
        )
        seed_table = _write_iteration_seed_table(root, seeds, round_index)
        try:
            result = run_perturbation_pipeline(
                finalized_hits_path=seed_table,
                output_dir=root,
                samples=int(samples_per_seed_operation),
                samples_per_seed_operation=True,
                merge_existing_products=True,
                iteration_model_path=model_path,
                iteration_budget=int(per_round_budget),
                iteration_round=round_index,
                iteration_ring_weight=ring_weight,
                operation_attempts=attempts,
                operations=generation_operations,
                **pipeline_kwargs,
            )
        finally:
            seed_table.unlink(missing_ok=True)
        statistics = _iteration_round_statistics(root, round_index)
        duplicate_rejections = _iteration_duplicate_rejections(root, round_index)
        gained = int(statistics["tier1_total"]) - baseline_tier1
        row = {
            "round": round_index,
            "seed_list_size": seed_list_size,
            "fresh_seeds_in_list": fresh_seeds,
            "seed_rows": len(seeds),
            "evaluated_rows": int(result.iteration_evaluated_rows),
            "selected_rows": int(result.iteration_selected_rows),
            "predicted_tier1_rows": int(result.iteration_predicted_tier1_rows),
            "duplicate_rejections": duplicate_rejections,
            "new_products": int(statistics["products"]),
            "new_tier1": int(statistics[TIER1]),
            "new_tier2": int(statistics[TIER2]),
            "new_tier3": int(statistics[TIER3]),
            "new_tier4": int(statistics[TIER4]),
            "new_potential1": int(statistics[POTENTIAL1]),
            "new_potential2": int(statistics[POTENTIAL2]),
            "new_potential3": int(statistics[POTENTIAL3]),
            "tier1_total": int(statistics["tier1_total"]),
            "tier1_gained_since_start": gained,
        }
        summaries.append(row)
        _write_rows(root / ITERATION_SUMMARY_FILE, summaries)
        # Adapt the sampling allocation from what the round actually produced and
        # carry it into the next round (the model itself stays static).
        realised = _iteration_operator_gain(root, round_index)
        if operator_adaptation_rate > 0.0:
            weights = _adapt_operator_weights(
                dict(weights), realised, rate=operator_adaptation_rate
            )
            attempts = (
                attempts_per_operation(
                    weights, total_attempts_per_molecule=attempts_per_molecule
                )
                if attempts_per_molecule
                else attempts_per_operation(
                    weights,
                    attempts_per_seed_operation=int(samples_per_seed_operation),
                )
            )
            row["next_attempts"] = ";".join(
                "%s=%d" % (operation, attempts[operation])
                for operation in PERTURBATION_OPERATIONS
            )
            _write_rows(root / ITERATION_SUMMARY_FILE, summaries)
            _panel_stage(
                "route",
                "Iteration %d allocation for the next round: %s"
                % (
                    round_index,
                    ", ".join(
                        "%s=%d" % (operation, attempts[operation])
                        for operation in PERTURBATION_OPERATIONS
                    ),
                ),
            )
        _panel_stage(
            "route",
            "Iteration %d done: %d new products, %d Tier 1 (folder total %d, %+d since start)"
            % (
                round_index,
                row["new_products"],
                row["new_tier1"],
                row["tier1_total"],
                gained,
            ),
        )
        if rounds_without_improvement and int(row["new_tier1"]) <= 0:
            rounds_without_gain += 1
            if rounds_without_gain >= int(rounds_without_improvement):
                stop_reason = "no_new_tier1_products"
                break
        else:
            rounds_without_gain = 0
    final_tier1 = len(_read_rows(root / CHANNEL_FILENAMES[TIER1]))
    summary = {
        "output_dir": str(root),
        "iterations_requested": int(iterations),
        "iterations_run": len(summaries),
        "seed_pools": list(pools),
        "absent_reference_libraries": reference_state["absent"],
        "iteration_model_path": str(model_path),
        "iteration_model_trained_here": model_summary is not None,
        "iteration_model_summary": model_summary,
        "tier1_before": baseline_tier1,
        "tier1_after": final_tier1,
        "tier1_gained": final_tier1 - baseline_tier1,
        "per_round_budget": int(per_round_budget),
        "max_seeds_per_round": max_seeds_per_round,
        "samples_per_seed_operation": int(samples_per_seed_operation),
        "attempts_per_molecule": attempts_per_molecule,
        "operator_adaptation_rate": operator_adaptation_rate,
        "operator_attempts_start": dict(operator_attempts or {}),
        "attempts_per_operation": attempts,
        "operator_weights": dict(weights),
        "iteration_model_static": True,
        "reference_libraries": reference_libraries,
        "missing_reference_libraries": missing_reference,
        "pool_consistency": consistency,
        "rounds": summaries,
        "stop_reason": stop_reason,
        "summary_path": str(root / ITERATION_SUMMARY_FILE),
    }
    (root / "iteration_pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # The per-round seed table is deleted as soon as its round is merged; drop
    # the scratch folder too, so the run folder only keeps tables that describe
    # the run. A scratch folder that still holds a table is left in place (it is
    # the seed list of a round that did not finish).
    scratch = root / "_iteration"
    try:
        if scratch.is_dir() and not any(scratch.iterdir()):
            scratch.rmdir()
    except OSError:
        pass
    return summary


def _perturb_smiles(
    smiles: str,
    operation: str,
    rng: random.Random,
) -> tuple[str | None, str]:
    if operation not in ALLOWED_PERTURBATION_OPERATIONS:
        raise ValueError(f"unknown perturbation operation: {operation}")
    generators = {
        "bond_rewire": _perturb_bond_rewire,
        "ring_adjust": _perturb_ring_adjust,
        "ring_expand": _perturb_ring_expand,
        "methyl_shift": _perturb_methyl_shift,
        "local_skeleton_rearrange": _perturb_local_skeleton_rearrange,
        "ch2_bridge_move": _perturb_ch2_bridge_move,
        "methylation": _perturb_methylation,
    }
    return generators[operation](smiles, rng)


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import rdmolops
    except ImportError as exc:
        raise RuntimeError("RDKit is required for perturbation generation.") from exc
    return Chem, rdmolops


def _parent_mol(smiles: str):
    Chem, _ = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    mol = Chem.RemoveHs(mol)
    _validate_carbon_single_bond_mol(mol)
    return mol


def _perturb_bond_rewire(smiles: str, rng: random.Random) -> tuple[str | None, str]:
    Chem, _ = _require_rdkit()
    parent = _parent_mol(smiles)
    bonds = _shuffled_bonds(parent, rng)
    for u, v in bonds:
        for a, b in _shuffled_pairs(parent.GetNumAtoms(), rng):
            if {a, b} == {u, v}:
                continue
            rw = Chem.RWMol(parent)
            rw.RemoveBond(u, v)
            mol = rw.GetMol()
            if _has_bond(mol, a, b):
                continue
            if _heavy_degree(mol, a) >= 4 or _heavy_degree(mol, b) >= 4:
                continue
            rw.AddBond(a, b, Chem.BondType.SINGLE)
            canonical = _canonical_if_valid(rw.GetMol(), smiles)
            if canonical:
                return canonical, f"removed {u}-{v}; added {a}-{b}"
    return None, "no_valid_bond_rewire"


def _perturb_ring_adjust(smiles: str, rng: random.Random) -> tuple[str | None, str]:
    Chem, rdmolops = _require_rdkit()
    parent = _parent_mol(smiles)
    distances = rdmolops.GetDistanceMatrix(parent)
    pairs = _shuffled_pairs(parent.GetNumAtoms(), rng)
    for a, b in pairs:
        if _has_bond(parent, a, b):
            continue
        if _heavy_degree(parent, a) >= 4 or _heavy_degree(parent, b) >= 4:
            continue
        distance = int(distances[a][b])
        if distance < 3 or distance > 6:
            continue
        rw = Chem.RWMol(parent)
        rw.AddBond(a, b, Chem.BondType.SINGLE)
        canonical = _canonical_if_valid(rw.GetMol(), smiles)
        if canonical:
            return canonical, f"added ring bond {a}-{b}"
    return _perturb_bond_rewire(smiles, rng)


def _perturb_methyl_shift(smiles: str, rng: random.Random) -> tuple[str | None, str]:
    Chem, _ = _require_rdkit()
    parent = _parent_mol(smiles)
    terminals = [atom.GetIdx() for atom in parent.GetAtoms() if _heavy_degree(parent, atom.GetIdx()) == 1]
    rng.shuffle(terminals)
    for terminal in terminals:
        neighbor = next(iter(parent.GetAtomWithIdx(terminal).GetNeighbors())).GetIdx()
        for target in _shuffled_atoms(parent.GetNumAtoms(), rng):
            if target in {terminal, neighbor}:
                continue
            rw = Chem.RWMol(parent)
            rw.RemoveBond(terminal, neighbor)
            mol = rw.GetMol()
            if _has_bond(mol, terminal, target):
                continue
            if _heavy_degree(mol, target) >= 4:
                continue
            rw.AddBond(terminal, target, Chem.BondType.SINGLE)
            canonical = _canonical_if_valid(rw.GetMol(), smiles)
            if canonical:
                return canonical, f"moved methyl {terminal} from {neighbor} to {target}"
    return None, "no_valid_methyl_shift"


def _perturb_ring_expand(smiles: str, rng: random.Random) -> tuple[str | None, str]:
    """Insert one CH2 into a small ring: the structural route towards JP-10.

    The ring bond is opened and a new carbon is bonded to both ends, so a
    three-membered ring becomes four-membered, four becomes five and five becomes
    six. Only single bonds inside three/four/five-membered rings are used, which
    keeps the product a saturated hydrocarbon.
    """

    Chem, _rdmolops = _require_rdkit()
    parent = _parent_mol(smiles)
    candidates: list[tuple[int, int]] = []
    for ring in parent.GetRingInfo().AtomRings():
        if len(ring) not in (3, 4, 5):
            continue
        for index, atom in enumerate(ring):
            other = ring[(index + 1) % len(ring)]
            bond = parent.GetBondBetweenAtoms(atom, other)
            if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                candidates.append((atom, other))
    rng.shuffle(candidates)
    for first, second in candidates:
        rw = Chem.RWMol(parent)
        if rw.GetBondBetweenAtoms(first, second) is None:
            continue
        rw.RemoveBond(first, second)
        new_index = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(first, new_index, Chem.BondType.SINGLE)
        rw.AddBond(second, new_index, Chem.BondType.SINGLE)
        canonical = _canonical_if_valid(rw.GetMol(), smiles)
        if canonical:
            return canonical, "inserted CH2 into the ring bond %d-%d" % (first, second)
    return None, "no_valid_ring_expand"


def _perturb_local_skeleton_rearrange(smiles: str, rng: random.Random) -> tuple[str | None, str]:
    Chem, rdmolops = _require_rdkit()
    parent = _parent_mol(smiles)
    distances = rdmolops.GetDistanceMatrix(parent)
    for u, v in _shuffled_bonds(parent, rng):
        for a, b in _shuffled_pairs(parent.GetNumAtoms(), rng):
            if {a, b} == {u, v}:
                continue
            if min(int(distances[u][a]), int(distances[u][b]), int(distances[v][a]), int(distances[v][b])) > 2:
                continue
            rw = Chem.RWMol(parent)
            rw.RemoveBond(u, v)
            mol = rw.GetMol()
            if _has_bond(mol, a, b):
                continue
            if _heavy_degree(mol, a) >= 4 or _heavy_degree(mol, b) >= 4:
                continue
            rw.AddBond(a, b, Chem.BondType.SINGLE)
            canonical = _canonical_if_valid(rw.GetMol(), smiles)
            if canonical:
                return canonical, f"local rearrange removed {u}-{v}; added {a}-{b}"
    return None, "no_valid_local_skeleton_rearrange"


def _perturb_ch2_bridge_move(smiles: str, rng: random.Random) -> tuple[str | None, str]:
    Chem, _ = _require_rdkit()
    parent = _parent_mol(smiles)
    bridges = [atom.GetIdx() for atom in parent.GetAtoms() if _heavy_degree(parent, atom.GetIdx()) == 2]
    rng.shuffle(bridges)
    for bridge in bridges:
        neighbors = [atom.GetIdx() for atom in parent.GetAtomWithIdx(bridge).GetNeighbors()]
        rng.shuffle(neighbors)
        for detached_neighbor in neighbors:
            kept_neighbor = next(n for n in neighbors if n != detached_neighbor)
            for target in _shuffled_atoms(parent.GetNumAtoms(), rng):
                if target in {bridge, kept_neighbor, detached_neighbor}:
                    continue
                rw = Chem.RWMol(parent)
                rw.RemoveBond(bridge, detached_neighbor)
                mol = rw.GetMol()
                if _has_bond(mol, bridge, target):
                    continue
                if _heavy_degree(mol, target) >= 4:
                    continue
                rw.AddBond(bridge, target, Chem.BondType.SINGLE)
                canonical = _canonical_if_valid(rw.GetMol(), smiles)
                if canonical:
                    return canonical, f"moved CH2 bridge {bridge} from {detached_neighbor} to {target}"
    return None, "no_valid_ch2_bridge_move"


def _perturb_methylation(smiles: str, rng: random.Random) -> tuple[str | None, str]:
    Chem, _ = _require_rdkit()
    parent = _parent_mol(smiles)
    atoms = _shuffled_atoms(parent.GetNumAtoms(), rng)
    for target in atoms:
        if _heavy_degree(parent, target) >= 4:
            continue
        rw = Chem.RWMol(parent)
        new_atom = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(target, new_atom, Chem.BondType.SINGLE)
        canonical = _canonical_if_valid(rw.GetMol(), smiles)
        if canonical:
            return canonical, f"added methyl {new_atom} to {target}"
    return None, "no_valid_methylation"


def _validate_carbon_single_bond_mol(mol) -> None:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            raise ValueError("perturbation currently supports hydrocarbons only")
    for bond in mol.GetBonds():
        if str(bond.GetBondType()) != "SINGLE":
            raise ValueError("perturbation currently supports single-bond carbon skeletons only")


def _canonical_if_valid(mol, parent_smiles: str) -> str | None:
    Chem, _ = _require_rdkit()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        _validate_carbon_single_bond_mol(mol)
    except ValueError:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    parent_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(parent_smiles), canonical=True)
    if canonical == parent_canonical:
        return None
    return canonical


def _heavy_degree(mol, atom_index: int) -> int:
    return int(mol.GetAtomWithIdx(atom_index).GetDegree())


def _has_bond(mol, a: int, b: int) -> bool:
    return mol.GetBondBetweenAtoms(int(a), int(b)) is not None


def _shuffled_atoms(n_atoms: int, rng: random.Random) -> list[int]:
    atoms = list(range(n_atoms))
    rng.shuffle(atoms)
    return atoms


def _shuffled_pairs(n_atoms: int, rng: random.Random) -> list[tuple[int, int]]:
    pairs = [(a, b) for a in range(n_atoms) for b in range(a + 1, n_atoms)]
    rng.shuffle(pairs)
    return pairs


def _shuffled_bonds(mol, rng: random.Random) -> list[tuple[int, int]]:
    bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
    ]
    rng.shuffle(bonds)
    return bonds


def _write_rows(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames is None and not rows:
            return
        if fieldnames is None:
            fieldnames = []
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames_of(rows: list[dict[str, object]]) -> list[str]:
    """Column order of a row set, as :func:`_write_rows` would derive it."""

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _merge_rows_into_table(
    path: Path,
    rows: list[dict[str, object]],
    *,
    key: str = "canonical_smiles",
) -> None:
    """Append ``rows`` to the table already stored at ``path``.

    Every perturbation campaign of a run folder contributes to the same set of
    tables: the rows written by an earlier campaign stay in place and the new
    rows are appended. A row is never stored twice - when the key repeats the
    new row wins. Rows are streamed, so merging a 40 MB product table does not
    pull the whole file into memory.
    """

    if not rows:
        return
    new_keys = {str(row.get(key) or "").strip() for row in rows}
    new_keys.discard("")
    if not path.exists() or not new_keys:
        _write_rows(path, rows)
        return
    incoming = path.with_name(path.name + ".incoming")
    _write_rows(incoming, rows)
    try:
        _merge_rows_on_top(path, incoming, path, key=key)
    finally:
        incoming.unlink(missing_ok=True)


def _write_stage_table(
    path: Path,
    rows: list[dict[str, object]],
    *,
    merge_existing: bool,
    key: str = "canonical_smiles",
    fieldnames: list[str] | None = None,
) -> None:
    """Store one run-level stage table, keeping earlier campaigns when asked."""

    if merge_existing:
        if not rows:
            # Never truncate a table that already holds an earlier campaign.
            if path.exists():
                return
        elif path.exists():
            _merge_rows_into_table(path, rows, key=key)
            return
    _write_rows(path, rows, fieldnames)


def _iter_rows(path: Path):
    """Yield the rows of a CSV table one at a time (streaming reader)."""

    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield row


def _table_header(path: Path) -> list[str]:
    """Column names of a CSV table, without reading its rows."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _merge_table_files(
    path: Path,
    prior_path: Path,
    *,
    key: str = "canonical_smiles",
) -> int:
    """Join the table a campaign just wrote with the table it replaced.

    Both files are streamed, so a six-figure attempt log is merged without
    holding it in memory. The freshly written rows win when a key repeats.
    """

    if not prior_path.exists() or not path.exists():
        return 0
    kept = _merge_rows_on_top(
        prior_path,
        path,
        path,
        key=key,
    )
    prior_path.unlink()
    return kept


def _id_suffix(identifier: object) -> int:
    """Trailing number of a ``..._000123`` candidate id (0 when absent)."""

    head, _, tail = str(identifier or "").rpartition("_")
    return int(tail) if head and tail.isdigit() else 0


def _with_id_number(identifier: object, number: int) -> object:
    """Same candidate id with a new trailing number."""

    head, _, tail = str(identifier or "").rpartition("_")
    if not head or not tail.isdigit():
        return identifier
    return "%s_%06d" % (head, number)


def _merge_rows_on_top(
    existing_path: Path,
    incoming_path: Path,
    output_path: Path,
    *,
    key: str = "canonical_smiles",
    extra_fieldnames: list[str] | None = None,
) -> int:
    """Write ``incoming_path`` on top of an existing table into ``output_path``.

    Rows of the existing table are kept unless an incoming row carries the same
    key. An incoming row whose candidate id is already taken gets the next free
    number, so two campaigns can share one table without repeating an id while
    every untouched row keeps its own. Everything is streamed. Returns the
    number of rows kept from the existing table.
    """

    incoming_keys = {
        str(row.get(key) or "").strip() for row in _iter_rows(incoming_path)
    }
    incoming_keys.discard("")
    fieldnames = list(extra_fieldnames or [])
    for column in _table_header(existing_path) + _table_header(incoming_path):
        if column not in fieldnames:
            fieldnames.append(column)
    temporary = output_path.with_name(output_path.name + ".merge")
    kept = 0
    max_suffix = 0
    used_ids: set[str] = set()
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=fieldnames,
            restval="",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in _iter_rows(existing_path):
            # The numbering floor comes from every stored row, including the ones
            # an incoming row replaces, so a renumbered id can never land back
            # inside the range the stored table already used.
            max_suffix = max(max_suffix, _id_suffix(row.get("candidate_id")))
            if str(row.get(key) or "").strip() in incoming_keys:
                continue
            identifier = str(row.get("candidate_id") or "")
            if identifier:
                used_ids.add(identifier)
            writer.writerow(row)
            kept += 1
        for row in _iter_rows(incoming_path):
            identifier = str(row.get("candidate_id") or "")
            if identifier and identifier in used_ids:
                while True:
                    max_suffix += 1
                    candidate = str(_with_id_number(identifier, max_suffix))
                    if candidate not in used_ids:
                        break
                row = dict(row)
                row["candidate_id"] = candidate
                identifier = candidate
            if identifier:
                used_ids.add(identifier)
                max_suffix = max(max_suffix, _id_suffix(identifier))
            writer.writerow(row)
    temporary.replace(output_path)
    return kept


#: Columns the screening pass adds to a candidate row. They are listed
#: explicitly so a streamed table always carries the same header, whichever row
#: happens to be written first.
SCREENING_COLUMNS = (
    "passes_screen",
    "screen_score",
    "failure_reasons",
    "is_connected",
    "is_novel",
    "max_known_similarity",
    "fingerprint_novelty",
    "high_novelty",
    "nearest_known_smiles",
    "mmff_embed_success",
    "mmff_optimize_success",
    "mmff_converged",
    "mmff_energy_kcal_mol",
    "mmff_energy_per_atom",
    "mmff_energy_per_heavy_atom",
    "mmff_stability_score",
    "mmff_failure_reason",
)


class _StageTableStream:
    """Write a run-level stage table while its rows are being produced.

    A perturbation campaign can produce six figures of candidate rows. Streaming
    them keeps the screening loop from holding every row in memory: only the
    rows that continue to property prediction stay alive. Rows go to a
    ``.partial`` sibling first and are moved into place when the stage is done,
    so an interrupted run never truncates a table that already holds an earlier
    campaign.
    """

    def __init__(
        self,
        path: Path,
        *,
        merge_existing: bool,
        key: str = "canonical_smiles",
    ) -> None:
        self.path = path
        self.merge_existing = merge_existing
        self.key = key
        self.count = 0
        self._partial = path.with_name(path.name + ".partial")
        self._handle = None
        self._writer = None
        self._fieldnames: list[str] = []
        self._keys: set[str] = set()

    def write(self, row: dict[str, object]) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._partial.open("w", encoding="utf-8", newline="")
            fieldnames = list(row)
            for column in SCREENING_COLUMNS:
                if column not in fieldnames:
                    fieldnames.append(column)
            self._fieldnames = fieldnames
            self._writer = csv.DictWriter(
                self._handle,
                fieldnames=fieldnames,
                restval="",
                extrasaction="ignore",
            )
            self._writer.writeheader()
        self._writer.writerow(row)
        value = str(row.get(self.key) or "").strip()
        if value:
            self._keys.add(value)
        self.count += 1

    def close(self) -> int:
        """Move the streamed rows into place and return the stored row count."""

        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None
        if self.count == 0:
            if self._partial.exists():
                self._partial.unlink()
            if not self.path.exists() and not self.merge_existing:
                _write_rows(self.path, [])
            return 0
        if self.merge_existing and self.path.exists() and self._keys:
            _merge_rows_on_top(
                self.path,
                self._partial,
                self.path,
                key=self.key,
                extra_fieldnames=self._fieldnames,
            )
            self._partial.unlink()
        else:
            self._partial.replace(self.path)
        return self.count


#: Columns that record which property models produced a stored row. Two
#: campaigns may only share one set of tables when they agree on these, so a
#: mismatched setting (for example a different fusion head) stops the run
#: instead of silently mixing two model versions in one table.
MODEL_IDENTITY_COLUMNS = (
    "external_melting_point_model_path",
    "external_boiling_point_model_path",
    "external_flash_point_model_path",
    "e3nn_geometry_model_checkpoint",
    "sta_property_model_checkpoint",
)


def _stored_model_identity(path: Path) -> dict[str, str]:
    """Model identifiers recorded in the first row of an existing product table."""

    identity: dict[str, str] = {}
    for row in _iter_rows(path):
        for column in MODEL_IDENTITY_COLUMNS:
            value = str(row.get(column) or "").strip()
            if value:
                identity[column] = value
        break
    return identity


def _max_stored_id_suffix(path: Path, prefix: str) -> int:
    """Highest ``candidate_id`` number already used with one campaign prefix."""

    maximum = 0
    for row in _iter_rows(path):
        identifier = str(row.get("candidate_id") or "")
        if identifier.startswith(prefix):
            maximum = max(maximum, _id_suffix(identifier))
    return maximum


def _assert_models_match_stored_products(
    path: Path,
    expected: dict[str, object],
) -> None:
    """Fail fast when stored products come from a different property model set."""

    stored = _stored_model_identity(path)
    mismatches = [
        "%s (stored: %s | this run: %s)"
        % (column, stored.get(column, "-"), str(expected[column]).strip())
        for column in MODEL_IDENTITY_COLUMNS
        if str(expected.get(column) or "").strip()
        and stored.get(column)
        and stored[column] != str(expected[column]).strip()
    ]
    if mismatches:
        raise ValueError(
            "The products already stored in this output folder were produced by "
            "a different property model set, so merging would mix two model "
            "versions in one table. Use the same model settings as the earlier "
            "campaign, or run into a new output folder. Mismatched settings: "
            + "; ".join(mismatches)
        )


def _carried_product_rows(
    path: Path,
    *,
    exclude_smiles: set[str],
) -> list[dict[str, object]]:
    """Load the products of earlier campaigns, dropping rows the new run replaces."""

    carried: list[dict[str, object]] = []
    for row in _iter_rows(path):
        smiles = str(row.get("canonical_smiles") or "").strip()
        if not smiles or smiles in exclude_smiles:
            continue
        carried.append(dict(row))
    return carried


def _rescore_carried_product_row(
    row: dict[str, object],
    *,
    uncertainty_scale: float,
    thresholds: FuelPropertyThresholds,
    ignore_density: bool,
    ignore_heating_value: bool,
    use_melting_point_screen: bool,
    use_boiling_point_screen: bool,
    use_flash_point_screen: bool,
    use_volumetric_heating_value_screen: bool,
    score_weights: CompositeScoreWeights,
    property_score_weights: PropertyScoreWeights,
    missing_property_penalty: float,
    benchmark: object | None,
) -> bool:
    """Re-apply this run's routing, scoring and tiering to a stored product.

    The merged tables must follow one single rule, so a product of an earlier
    campaign is re-scored from the property values stored in its row. Nothing is
    re-predicted: the tier channels are a pure function of those values.
    """

    try:
        _route_enriched_property_row(
            row,
            uncertainty_scale=uncertainty_scale,
            thresholds=thresholds,
            ignore_density=ignore_density,
            ignore_heating_value=ignore_heating_value,
            use_melting_point_screen=use_melting_point_screen,
            use_boiling_point_screen=use_boiling_point_screen,
            use_flash_point_screen=use_flash_point_screen,
            use_volumetric_heating_value_screen=use_volumetric_heating_value_screen,
        )
        _add_objective_scores(
            row,
            objective_mode="property_maximize",
            thresholds=thresholds,
            score_weights=score_weights,
            property_score_weights=property_score_weights,
            property_targets=PropertyTargetValues(),
            objective_relative_window=0.15,
            missing_property_penalty=missing_property_penalty,
        )
        if benchmark is not None:
            apply_tier_fields(row, benchmark)  # type: ignore[arg-type]
        return True
    except Exception:
        # Keep the row exactly as it was stored: re-using the stored screening
        # status keeps the rewritten table identical for this row.
        row["hard_threshold_verdict"] = str(row.get("screening_status") or "")
        return False


def _combine_manual_review_candidates(
    symmetric_seed_rows: list[dict[str, object]],
    perturbed_tier_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Merge the run's seeds with the perturbation products worth manual review.

    The pool follows the tier split: every Tier 1-3 product (plus the
    Tier 1-3 seeds themselves) is offered for a manual high-level calculation
    decision.
    """
    combined: list[dict[str, object]] = []
    seen: set[str] = set()
    sources = (
        (symmetric_seed_rows, "perfect_symmetric"),
        (perturbed_tier_rows, "perturbed_from_hits"),
    )
    for rows, origin in sources:
        for row in rows:
            smiles = str(row.get("canonical_smiles", "")).strip()
            tier = str(row.get("tier") or "").strip().lower()
            if not smiles or tier not in {"tier1", "tier2", "tier3"} or smiles in seen:
                continue
            seen.add(smiles)
            combined.append(
                {
                    **row,
                    "manual_review_id": f"manual_{len(combined) + 1:06d}",
                    "manual_review_status": "pending_manual_calculation",
                    "manual_review_tier": tier,
                    "candidate_origin": origin,
                }
            )
    combined.sort(
        key=lambda row: (
            str(row.get("manual_review_tier", "")),
            -float(row.get("composite_ranking_score") or 0.0),
            str(row.get("canonical_smiles", "")),
        )
    )
    return combined


def _load_smiles_from_paths(paths: tuple[str | Path, ...]) -> set[str]:
    smiles: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value:
                smiles.add(value)
    return smiles


def _write_combined_smiles_library(
    output_dir: Path,
    paths: tuple[str | Path, ...],
) -> Path | None:
    smiles = sorted(_load_smiles_from_paths(paths))
    if not smiles:
        return None
    path = output_dir / "known_plus_temporary_smiles.txt"
    path.write_text("\n".join(smiles) + "\n", encoding="utf-8")
    return path


def _load_heating_values(path_value: str | Path | None) -> dict[str, float]:
    if path_value is None:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    rows = _read_rows(path)
    values: dict[str, float] = {}
    for row in rows:
        smiles = str(
            row.get("canonical_smiles")
            or row.get("smiles")
            or row.get("SMILES")
            or ""
        ).strip()
        value = (
            row.get("heating_value_mj_kg")
            or row.get("external_heating_value_mj_kg")
            or row.get("mass_heating_value_mj_kg")
            or ""
        )
        if not smiles or value in ("", None):
            continue
        try:
            values[smiles] = float(value)
        except ValueError:
            continue
    return values

def _blank_if_none(value: object) -> object:
    return "" if value is None else value


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)




def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_dict(path: Path) -> dict[str, object]:
    """Small JSON object read that never raises (missing/broken file -> {})."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _score_iteration_candidates(
    candidates_path: Path,
    parents: dict[str, dict[str, object]],
    *,
    model_path: str | Path,
    budget: int,
    exploration_fraction: float,
    round_index: int,
    ring_weight: float = 1.0,
) -> dict[str, object]:
    """Rank this round's enumerated products with the iteration model.

    The candidate rows are streamed: only the small scoring tuple of a product is
    built, never the whole attempt log, and the log itself is rewritten in place
    with the iteration columns so the selection stays inspectable.
    """

    artifact = load_iteration_model(model_path)
    reference = JP10Reference.from_mapping(
        artifact.get("jp10_reference") or DOCUMENTED_JP10_REFERENCE
    )
    columns = list(artifact.get("feature_columns") or feature_columns())
    models = artifact.get("model")
    if not isinstance(models, Mapping):
        raise ValueError("iteration model artifact has no fitted models")

    evaluated: list[tuple[str, str, dict[str, float], dict[str, float]]] = []
    for row in _iter_rows(candidates_path):
        smiles = str(row.get("canonical_smiles") or "").strip()
        parent_smiles = str(row.get("parent_canonical_smiles") or "").strip()
        operation = str(row.get("operation") or "").strip()
        parent = parents.get(parent_smiles)
        if not smiles or parent is None or operation not in PERTURBATION_OPERATIONS:
            continue
        features = build_features(parent, row, operation, reference)
        parent_margins = margins_of_row(parent, reference)
        evaluated.append((smiles, parent_smiles, features, parent_margins))
    if not evaluated:
        return {
            "evaluated": 0,
            "ranked": 0,
            "selected": 0,
            "selected_smiles": [],
            "explored": 0,
            "predicted_tier1": 0,
            "budget": int(budget),
        }

    matrix = np.asarray(
        [
            [_iteration_number(features.get(column)) for column in columns]
            for _, _, features, _ in evaluated
        ],
        dtype=float,
    )
    deltas = {
        key: np.asarray(model.predict(matrix), dtype=float)
        for key, model in models.items()
    }
    scale = {
        "margin_density": max(abs(reference.density_g_cm3), 1e-6),
        "margin_heating_value": max(abs(reference.heating_value_mj_kg), 1e-6),
        "margin_melting_point": max(abs(reference.melting_point_k), 1e-6),
    }
    scored: list[dict[str, object]] = []
    for position, (smiles, parent_smiles, _features, parent_margins) in enumerate(evaluated):
        predicted = {
            key: parent_margins[key]
            + float(deltas.get("delta_%s" % key, np.zeros(len(evaluated)))[position])
            for key in MARGIN_KEYS
        }
        predicted_beats = beats_from_margins(predicted)
        ring_reward, ring_penalty, _strained, _preferred = ring_preference(smiles)
        promise = (
            10.0 * predicted_beats
            + sum(
                max(min(predicted[key] / scale[key], 0.5), -0.5)
                for key in MARGIN_KEYS
            )
            + ring_weight * RING_REWARD_FACTOR * ring_reward
            - ring_weight * RING_STRAIN_FACTOR * ring_penalty
        )
        scored.append(
            {
                "canonical_smiles": smiles,
                "parent_canonical_smiles": parent_smiles,
                "predicted_beats": predicted_beats,
                "ring_reward": ring_reward,
                "ring_strain_penalty": ring_penalty,
                "promise": promise,
            }
        )
    selection = select_iteration_candidates(
        scored,
        budget=budget,
        exploration_fraction=exploration_fraction,
        rng=random.Random(round_index * 7919 + 13),
    )
    selected_ids = {
        str(row["canonical_smiles"]): row for row in selection["selected"]  # type: ignore[union-attr]
    }
    predicted_tier1 = sum(
        1 for row in scored if int(row["predicted_beats"]) >= 3
    )

    # Rewrite the attempt log with the iteration annotations.
    temporary = candidates_path.with_name(candidates_path.name + ".iteration")
    with candidates_path.open("r", encoding="utf-8", newline="") as source, temporary.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        for column in (
            "iteration_round",
            "iteration_selected",
            "iteration_predicted_beats",
            "iteration_promise",
        ):
            if column not in fieldnames:
                fieldnames.append(column)
        writer = csv.DictWriter(
            target, fieldnames=fieldnames, restval="", extrasaction="ignore"
        )
        writer.writeheader()
        for row in reader:
            smiles = str(row.get("canonical_smiles") or "").strip()
            match = selected_ids.get(smiles)
            row["iteration_round"] = round_index
            row["iteration_selected"] = bool(match) or smiles in selected_ids
            row["iteration_predicted_beats"] = (
                match["predicted_beats"] if match else ""
            )
            row["iteration_promise"] = match["promise"] if match else ""
            writer.writerow(row)
    temporary.replace(candidates_path)
    return {
        "evaluated": len(evaluated),
        "ranked": len(scored),
        "selected": len(selected_ids),
        "selected_smiles": sorted(selected_ids),
        "explored": selection["explored"],
        "predicted_tier1": predicted_tier1,
        "budget": int(budget),
    }


def _add_folder_generation_totals(
    summary_path: Path,
    previous_summary: dict[str, object],
) -> None:
    """Record the whole-folder campaign counters next to this campaign's own.

    ``perturbation_generation_summary.json`` describes the campaign that just
    ran; the ``folder_*`` copies sum every campaign of the run folder, so the
    stage report can describe the folder without lying about this campaign.
    """

    current = _read_json_dict(summary_path)
    if not current:
        return

    def number(source: dict[str, object], key: str) -> int:
        try:
            return int(source.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    def previous_number(key: str) -> int:
        # A folder that already merged campaigns carries the running total in
        # ``folder_<key>``; the plain key only holds the campaign before this one.
        return number(previous_summary, "folder_" + key) or number(
            previous_summary, key
        )

    for key in (
        "attempted_samples",
        "generated_rows",
        "duplicate_rows",
        "failed_attempts",
    ):
        current["folder_" + key] = previous_number(key) + number(current, key)
    summary_path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
