from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from bond_model.predict import edge_feedback_key
from candidate_export import CandidateExportResult, export_candidate_inputs_from_csv
from chemistry import RING_COUNT_DEFINITION, molar_mass_from_formula
from evaluation import GenerationEvaluation, evaluate_generation
from generator import iter_count_partitions
from graph import HalfGraph
from phase_fusion import (
    DEFAULT_PHASE_FUSION_MODEL_PATHS,
    ReferenceOnlyFeatureModelError,
    predict_phase_fusion_property,
)
from property_model.e3nn_geometry_scorer import (
    geometry_score_from_smiles,
    predict_geometry_score,
    set_reproducibility_enabled,
)
from property_model.data import geometry_from_smiles
from property_model.e3nn_regressor import predict_property
from property_screening import (
    ThresholdVerdict,
    FuelPropertyThresholds,
    GROUP_SUMMARY_STATUSES,
    PropertyDecision,
    ThresholdDirection,
    classify_property_threshold_with_uncertainty,
    screening_status,
)
from property_consensus import (
    CompositeScoreWeights,
    PropertyScoreWeights,
    PropertyTargetValues,
    composite_ranking_score,
    property_quality_score,
    synthetic_accessibility_property_score,
    target_property_score,
)
from screening import (
    ScreenedCandidate,
    ScreeningConfig,
    ScreeningSummary,
    screen_generation,
)
from known_fuel_library import match_known_fuel
from parallel import resolve_worker_count
from property_cache import PropertyCache, config_fingerprint
from tier_routing import (
    CHANNEL_FILENAMES,
    CHANNEL_ORDER,
    POTENTIAL1,
    POTENTIAL2,
    POTENTIAL3,
    apply_tier_fields,
    benchmark_from_models,
)
from sta_model import (
    DEFAULT_SPECIALIST_MODEL_PATHS,
    DEFAULT_STA_MODEL_PATH,
    predict_sta_properties,
    predict_sta_properties_batch,
)
from sta_feedback import STA_FEEDBACK_COLUMNS, build_sta_weak_feedback_rows
from thesis_property_tools import (
    estimate_vdw_density_from_smiles,
    mass_heating_value_from_formation_enthalpy,
    mass_heating_value_from_qm9_h298_proxy,
    molar_volume_from_density,
    volumetric_heating_value_from_density,
)
from fuel_design.paths import CBH_BDC_DIR, EVAL_DATA_DIR, SFS_VR_DIR

# Models/data of non-STA methods are archived in addition; these constants are
# kept only as historical defaults and are used only when those methods are
# explicitly selected (modules are lazily imported at call sites).
DEFAULT_CBH_BDC_MODEL_PATH = CBH_BDC_DIR / "cbh_bdc_delta_ml_formation.joblib"
DEFAULT_CBH_BDC_FORMATION_FEATURE_PATH = (
    EVAL_DATA_DIR / "qm9_cbh_bdc_formation_features.csv"
)
DEFAULT_YANG_CBH_BDC_ROOT = CBH_BDC_DIR / "CBH-BDC-deltaML-main"
DEFAULT_SFS_VR_MP_MODEL_PATH = SFS_VR_DIR / "sfs_vr_melting_point.joblib"
DEFAULT_SFS_VR_MP_DATA_PATH = (
    SFS_VR_DIR / "SFS-VR-main" / "MP" / "MP-454-select-FEATURE - 15.xlsx"
)


def _panel_stage(stage: str, message: str) -> None:
    print(f"__FUEL_PANEL_STAGE__:{stage}:{message}", flush=True)


@dataclass(frozen=True)
class PipelineSummary:
    total_carbons: int
    samples: int
    iterations: int
    samples_per_iteration: int
    total_requested_samples: int
    feedback_enabled: bool
    feedback_strength: float
    top_k: int | None
    seed: int | None
    generated_bases: int
    valid_candidates: int
    passed_pre_screen_before_temporary_dedup: int
    passed_pre_screen: int
    temporary_smiles_count: int
    temporary_duplicate_rejections: int
    rejected_pre_property: int
    property_prediction_rows: int
    property_classified_rows: int
    tier1_count: int
    tier2_count: int
    tier3_count: int
    tier4_count: int
    potential1_count: int
    potential2_count: int
    potential3_count: int
    property_failed_count: int
    sta_weak_feedback_count: int
    novelty_threshold: float
    output_dir: str
    grouped_output_dir: str
    sta_weak_feedback_output: str
    dft_export: CandidateExportResult | None
    objective_mode: str = "property_maximize"
    score_weights: dict[str, float] | None = None
    property_score_weights: dict[str, float] | None = None
    property_targets: dict[str, float | None] | None = None
    online_property_feedback_enabled: bool = False
    online_property_feedback_rows: int = 0
    online_property_feedback_unique_molecules: int = 0
    online_property_feedback_output: str = ""
    property_reward_slope: float = 2.5
    partition_feedback_gain: float = 6.0
    edge_feedback_gain: float = 2.0
    feedback_exploration_fraction: float = 0.12
    missing_property_count: int = 0
    missing_property_output: str = ""
    missing_property_penalty: float = 0.15
    perturbation_seed_count: int = 0
    perturbation_seed_output: str = ""


@dataclass(frozen=True)
class GenerationFeedbackRound:
    iteration: int
    requested_samples: int
    threshold: float
    temperature: float
    top_k: int | None
    valid_candidates: int
    raw_generated_candidates: int
    raw_valid_candidates: int
    raw_connected_candidates: int
    raw_invalid_candidates: int
    raw_valid_rate: float
    passed_screen: int
    pass_rate: float
    mean_reward_score: float
    best_reward_score: float
    min_sa_score: float | None
    mean_sa_score: float | None
    mean_h_to_c_ratio: float | None
    mean_ring_count: float | None
    mean_fingerprint_novelty: float | None
    mean_mmff_stability_score: float | None
    new_unique_candidates: int
    repeated_candidates: int
    property_feedback_active: bool
    property_feedback_weight: float
    property_reward_slope: float
    property_evaluated_count: int
    property_pass_count: int
    property_review_count: int
    property_reject_count: int
    mean_property_quality_score: float | None
    best_property_quality_score: float | None
    mean_composite_ranking_score: float | None
    mean_property_objective_score: float | None
    best_online_reward_score: float | None
    favored_partition: str
    favored_partition_weight: float | None
    favored_edge_action: str
    favored_edge_bias: float | None
    exploration_fraction: float
    sa_pass_count: int
    h_to_c_pass_count: int
    mmff_pass_count: int
    decision: str
    next_threshold: float
    next_temperature: float
    next_top_k: int | None


@dataclass(frozen=True)
class IterativeGenerationResult:
    evaluation: GenerationEvaluation
    feedback_rounds: tuple[GenerationFeedbackRound, ...]
    screened_candidates: tuple[ScreenedCandidate, ...]
    screening_summary: ScreeningSummary
    online_property_rows: tuple[dict[str, object], ...]


PropertyFeedbackEvaluator = Callable[
    [tuple[ScreenedCandidate, ...], int],
    list[dict[str, object]],
]


def run_generation_pipeline(
    checkpoint_path: str | Path,
    known_smiles_path: str | Path,
    output_dir: str | Path,
    total_carbons: int,
    samples: int,
    semi_base_checkpoint_path: str | Path | None = None,
    semi_base_threshold: float = 0.73,
    semi_base_temperature: float = 1.0,
    semi_base_top_k: int | None = None,
    iterations: int = 1,
    threshold: float = 0.94,
    temperature: float = 1.0,
    top_k: int | None = 2,
    seed: int | None = None,
    feedback_enabled: bool = False,
    feedback_strength: float = 1.25,
    online_property_feedback: bool = True,
    property_feedback_warmup_iterations: int = 20,
    property_feedback_warmup_weight: float = 0.35,
    property_feedback_weight: float = 0.70,
    property_reward_slope: float = 2.5,
    partition_feedback_gain: float = 6.0,
    edge_feedback_gain: float = 2.0,
    feedback_exploration_fraction: float = 0.12,
    use_semi_base: bool = True,
    novelty_threshold: float = 0.75,
    max_sa_score: float = 4.5,
    preferred_ring_weight: float = 0.10,
    preferred_ring_target: int = 2,
    export_dft_inputs: bool = True,
    heating_values_csv: str | Path | None = None,
    uncertainty_scale: float = 1.0,
    thresholds: FuelPropertyThresholds = FuelPropertyThresholds(),
    ignore_density: bool = False,
    ignore_heating_value: bool = False,
    use_melting_point_screen: bool = False,
    use_boiling_point_screen: bool = False,
    use_flash_point_screen: bool = False,
    use_volumetric_heating_value_screen: bool = False,
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
    mp_model_path: str | Path = DEFAULT_SFS_VR_MP_MODEL_PATH,
    mp_data_path: str | Path = DEFAULT_SFS_VR_MP_DATA_PATH,
    cbh_bdc_model_path: str | Path = DEFAULT_CBH_BDC_MODEL_PATH,
    cbh_bdc_feature_csv: str | Path = DEFAULT_CBH_BDC_FORMATION_FEATURE_PATH,
    cbh_bdc_yang_root: str | Path = DEFAULT_YANG_CBH_BDC_ROOT,
    phase_fusion_melting_point_model_path: str | Path | None = None,
    phase_fusion_boiling_point_model_path: str | Path | None = None,
    phase_fusion_flash_point_model_path: str | Path | None = None,
    use_cbh_bdc_heating: bool = True,
    objective_mode: str = "property_maximize",
    score_weights: CompositeScoreWeights = CompositeScoreWeights(),
    property_score_weights: PropertyScoreWeights = PropertyScoreWeights(),
    property_targets: PropertyTargetValues = PropertyTargetValues(),
    objective_relative_window: float = 0.15,
    missing_property_penalty: float = 0.15,
    sta_weak_feedback_enabled: bool = True,
    sta_weak_feedback_weight: float = 0.35,
    temporary_smiles_path: str | Path | None = None,
    known_fuel_check: bool = False,
    screen_workers: int | None = None,
    property_workers: int | None = None,
    use_property_cache: bool = True,
    property_cache_path: str | Path | None = None,
    reproducibility_scope: str = "tier12",
) -> PipelineSummary:
    if not 1 <= total_carbons <= 17:
        raise ValueError("total_carbons must be between 1 and 17")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if property_feedback_warmup_iterations < 0:
        raise ValueError("property_feedback_warmup_iterations must be non-negative")
    for name, value in (
        ("property_feedback_warmup_weight", property_feedback_warmup_weight),
        ("property_feedback_weight", property_feedback_weight),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if property_reward_slope <= 0.0:
        raise ValueError("property_reward_slope must be positive")
    if partition_feedback_gain <= 0.0:
        raise ValueError("partition_feedback_gain must be positive")
    if edge_feedback_gain <= 0.0:
        raise ValueError("edge_feedback_gain must be positive")
    if not 0.0 <= feedback_exploration_fraction < 1.0:
        raise ValueError("feedback_exploration_fraction must be in [0, 1)")
    if not 0.0 <= missing_property_penalty <= 1.0:
        raise ValueError("missing_property_penalty must be between 0 and 1")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = ScreeningConfig(
        high_novelty_threshold=novelty_threshold,
        max_sa_score=max_sa_score,
        preferred_ring_weight=preferred_ring_weight,
        preferred_ring_target=preferred_ring_target,
    )
    heating_values = _load_heating_value_map(heating_values_csv)
    property_prediction_cache: dict[str, dict[str, object]] = {}
    active_property_cache_path: Path | None = None
    if use_property_cache:
        active_property_cache_path = (
            Path(property_cache_path)
            if property_cache_path
            else output_path / "property_prediction_cache.pkl"
        )
    online_feedback_active = (
        feedback_enabled
        and online_property_feedback
        and objective_mode != "novelty_exploration"
    )

    # JP-10 reference for the channel pools, produced by the same models
    # that score the candidates. Failure here must not stop a run.
    jp10_benchmark: object | None = None
    try:
        jp10_benchmark = benchmark_from_models(
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
        print("__FUEL_PANEL_STAGE__:property:JP-10 tier benchmark unavailable: %s" % exc,
              flush=True)

    def build_properties(
        candidate_rows: list[dict[str, object]],
        *,
        prediction_seed: int,
    ) -> list[dict[str, object]]:
        return _build_property_rows(
            candidate_rows,
            heating_values=heating_values,
            uncertainty_scale=uncertainty_scale,
            thresholds=thresholds,
            ignore_density=ignore_density or density_model == "none",
            ignore_heating_value=ignore_heating_value,
            use_melting_point_screen=use_melting_point_screen,
            use_boiling_point_screen=use_boiling_point_screen,
            use_flash_point_screen=use_flash_point_screen,
            use_volumetric_heating_value_screen=(
                use_volumetric_heating_value_screen
            ),
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
            objective_mode=objective_mode,
            score_weights=score_weights,
            property_score_weights=property_score_weights,
            property_targets=property_targets,
            objective_relative_window=objective_relative_window,
            missing_property_penalty=missing_property_penalty,
            seed=prediction_seed,
            property_prediction_cache=property_prediction_cache,
            property_workers=property_workers,
            property_cache_path=active_property_cache_path,
            jp10_benchmark=jp10_benchmark,
            reproducibility_scope=reproducibility_scope,
        )

    def evaluate_online_properties(
        candidates: tuple[ScreenedCandidate, ...],
        iteration: int,
    ) -> list[dict[str, object]]:
        input_rows: list[dict[str, object]] = []
        for index, candidate in enumerate(candidates, start=1):
            if not candidate.passes_screen:
                continue
            row = _candidate_pipeline_row(index, candidate, config=config)
            row["candidate_id"] = f"feedback_{iteration:04d}_{index:04d}"
            row["online_feedback_iteration"] = iteration
            input_rows.append(row)
        if not input_rows:
            return []
        _panel_stage(
            "property",
            (
        f"Round {iteration}/{iterations}: predicting properties for "
        f"{len(input_rows)} screened molecules and computing rewards"
            ),
        )
        return build_properties(
            input_rows,
            prediction_seed=(seed or 0) + iteration * max(samples, 1),
        )

    generation = _evaluate_generation_iterations(
        checkpoint_path=checkpoint_path,
        semi_base_checkpoint_path=semi_base_checkpoint_path,
        semi_base_threshold=semi_base_threshold,
        semi_base_temperature=semi_base_temperature,
        semi_base_top_k=semi_base_top_k,
        known_smiles_path=known_smiles_path,
        total_carbons=total_carbons,
        samples=samples,
        iterations=iterations,
        threshold=threshold,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        feedback_enabled=feedback_enabled,
        feedback_strength=feedback_strength,
        screening_config=config,
        use_semi_base=use_semi_base,
        property_feedback_evaluator=(
            evaluate_online_properties if online_feedback_active else None
        ),
        property_feedback_warmup_iterations=property_feedback_warmup_iterations,
        property_feedback_warmup_weight=property_feedback_warmup_weight,
        property_feedback_weight=property_feedback_weight,
        property_reward_slope=property_reward_slope,
        partition_feedback_gain=partition_feedback_gain,
        edge_feedback_gain=edge_feedback_gain,
        feedback_exploration_fraction=feedback_exploration_fraction,
        mmff_workers=screen_workers,
    )
    evaluation = generation.evaluation
    _panel_stage("screen", "Running dedup, structure rules, and screening")
    candidates = generation.screened_candidates
    screening_summary = generation.screening_summary
    rows = [
        _candidate_pipeline_row(index, candidate, config=config)
        for index, candidate in enumerate(candidates, start=1)
    ]
    external_temporary_smiles: set[str] = set()
    if temporary_smiles_path and Path(temporary_smiles_path).exists():
        external_temporary_smiles = {
            line.strip()
            for line in Path(temporary_smiles_path).read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
            if line.strip()
        }
    (
        temporary_smiles_accepted,
        temporary_duplicate_rejections,
        temporary_smiles_count,
    ) = _apply_temporary_smiles_database(
        rows, external_temporary_smiles, output_path=temporary_smiles_path
    )

    _panel_stage("property", "Computing density, melting point, e3nn scores, and classification")
    rejected_rows = [
        row
        for row in rows
        if row["pipeline_route"] == "reject_pre_property"
    ]

    property_rows = build_properties(rows, prediction_seed=seed or 0)
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
    _panel_stage("route", "Classifying candidates and marking data gaps")
    property_failed_rows = [
        row
        for row in property_rows
        if row.get("hard_threshold_verdict") == "property_failed"
    ]
    missing_property_rows = [
        row
        for row in property_rows
        if row.get("hard_threshold_verdict") == "missing_properties"
    ]

    candidates_path = output_path / "screened_candidates.csv"
    rejected_path = output_path / "rejected_pre_property.csv"
    property_path = output_path / "property_predictions.csv"
    property_failed_path = output_path / "property_failed_candidates.csv"
    missing_property_path = output_path / "missing_property_candidates.csv"
    feedback_path = output_path / "generation_feedback.csv"
    online_property_feedback_path = output_path / "online_property_feedback.csv"
    sta_weak_feedback_path = output_path / "sta_low_novelty_weak_feedback.csv"
    fieldnames = _fieldnames(rows)
    _write_csv(candidates_path, rows, fieldnames=fieldnames)
    _write_csv(rejected_path, rejected_rows, fieldnames=fieldnames)
    public_property_rows = public_rows(property_rows)
    property_fieldnames = _fieldnames(public_property_rows)
    _write_csv(property_path, public_property_rows, fieldnames=property_fieldnames)
    _write_csv(
        sta_weak_feedback_path,
        sta_weak_feedback_rows,
        fieldnames=STA_FEEDBACK_COLUMNS,
    )

    # ---- tier outputs (T1..T4) and the potential salvage pools --------------
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
    grouped: dict[str, list[dict[str, object]]] = {
        channel: [] for channel in CHANNEL_ORDER
    }
    if any("tier" in row for row in property_rows):
        for row in property_rows:
            # Only a molecule that carries a channel is pooled. A molecule that
            # matches no pool (a data gap, a failed prediction, or one that
            # misses thresholds and beats JP-10 on nothing) stays in the property
            # table and must not be counted as Tier 4.
            channel = str(row.get("tier") or "").strip().lower()
            if channel in CHANNEL_FILENAMES:
                grouped[channel].append(row)
        for channel in CHANNEL_ORDER:
            tier_paths[channel] = output_path / CHANNEL_FILENAMES[channel]
            _write_csv(
                tier_paths[channel],
                public_rows(sorted(grouped.get(channel, []), key=_tier_order)),
                fieldnames=property_fieldnames,
            )
        _panel_stage(
            "route",
            "Channels: T1 %d / T2 %d / T3 %d / T4 %d | Pot1 %d / Pot2 %d / Pot3 %d"
            % (
                len(grouped.get("tier1", [])),
                len(grouped.get("tier2", [])),
                len(grouped.get("tier3", [])),
                len(grouped.get("tier4", [])),
                len(grouped.get(POTENTIAL1, [])),
                len(grouped.get(POTENTIAL2, [])),
                len(grouped.get(POTENTIAL3, [])),
            ),
        )

    # Perturbation seeds are the Tier 1-3 pools; the run-level ``build-t123-seeds``
    # step writes that table, and the summary records its size.
    perturbation_seed_count = (
        len(grouped.get("tier1", []))
        + len(grouped.get("tier2", []))
        + len(grouped.get("tier3", []))
    )
    _write_csv(
        property_failed_path,
        public_rows(property_failed_rows),
        fieldnames=property_fieldnames,
    )
    _write_csv(
        missing_property_path,
        public_rows(missing_property_rows),
        fieldnames=property_fieldnames,
    )
    feedback_rows = [asdict(round_) for round_ in generation.feedback_rounds]
    _write_csv(feedback_path, feedback_rows)
    _write_csv(
        online_property_feedback_path,
        list(generation.online_property_rows),
    )
    # One compact partition table per carbon replaces the old per-partition
    # folders (each of those held a metadata file plus seven CSVs, i.e. ~110
    # files per carbon); the tier tables already live in the carbon folder.
    _write_group_summary(output_path, property_rows, total_carbons)

    _panel_stage("route", "Exporting Tier 1 DFT input files")
    dft_export = None
    if export_dft_inputs and tier_paths.get("tier1") and grouped.get("tier1"):
        dft_export = export_candidate_inputs_from_csv(
            tier_paths["tier1"],
            output_path / "dft_inputs",
            smiles_column="canonical_smiles",
            id_column="candidate_id",
            seed=seed or 0,
        )
    summary = PipelineSummary(
        total_carbons=total_carbons,
        samples=samples,
        iterations=iterations,
        samples_per_iteration=samples,
        total_requested_samples=samples * iterations,
        feedback_enabled=feedback_enabled,
        feedback_strength=feedback_strength,
        top_k=top_k,
        seed=seed,
        generated_bases=evaluation.generated_bases,
        valid_candidates=screening_summary.valid_candidates,
        passed_pre_screen_before_temporary_dedup=screening_summary.passed_screen,
        passed_pre_screen=temporary_smiles_accepted,
        temporary_smiles_count=temporary_smiles_count,
        temporary_duplicate_rejections=temporary_duplicate_rejections,
        rejected_pre_property=len(rejected_rows),
        property_prediction_rows=len(property_rows),
        property_classified_rows=sum(
            len(grouped.get(channel, [])) for channel in CHANNEL_ORDER
        ),
        tier1_count=len(grouped.get("tier1", [])),
        tier2_count=len(grouped.get("tier2", [])),
        tier3_count=len(grouped.get("tier3", [])),
        tier4_count=len(grouped.get("tier4", [])),
        potential1_count=len(grouped.get(POTENTIAL1, [])),
        potential2_count=len(grouped.get(POTENTIAL2, [])),
        potential3_count=len(grouped.get(POTENTIAL3, [])),
        property_failed_count=len(property_failed_rows),
        sta_weak_feedback_count=len(sta_weak_feedback_rows),
        novelty_threshold=novelty_threshold,
        output_dir=str(output_path),
        grouped_output_dir=str(output_path),
        sta_weak_feedback_output=str(sta_weak_feedback_path),
        dft_export=dft_export,
        objective_mode=objective_mode,
        score_weights=score_weights.normalized().as_dict(),
        property_score_weights=property_score_weights.as_dict(),
        property_targets=property_targets.as_dict(),
        online_property_feedback_enabled=online_feedback_active,
        online_property_feedback_rows=len(generation.online_property_rows),
        online_property_feedback_unique_molecules=len(
            {
                str(row.get("canonical_smiles") or "")
                for row in generation.online_property_rows
                if row.get("canonical_smiles")
            }
        ),
        online_property_feedback_output=str(online_property_feedback_path),
        property_reward_slope=property_reward_slope,
        partition_feedback_gain=partition_feedback_gain,
        edge_feedback_gain=edge_feedback_gain,
        feedback_exploration_fraction=feedback_exploration_fraction,
        missing_property_count=len(missing_property_rows),
        missing_property_output=str(missing_property_path),
        missing_property_penalty=missing_property_penalty,
        perturbation_seed_count=perturbation_seed_count,
        perturbation_seed_output="",
    )
    if known_fuel_check:
        from known_fuel_library import build_known_fuel_hit_table

        build_known_fuel_hit_table(rows, output_path / "known_fuel_hits.csv")

    (output_path / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _panel_stage("done", f"C{total_carbons} pipeline completed")
    return summary


def _evaluate_generation_iterations(
    checkpoint_path: str | Path,
    semi_base_checkpoint_path: str | Path | None,
    semi_base_threshold: float,
    semi_base_temperature: float,
    semi_base_top_k: int | None,
    known_smiles_path: str | Path,
    total_carbons: int,
    samples: int,
    iterations: int,
    threshold: float,
    temperature: float,
    top_k: int | None,
    seed: int | None,
    feedback_enabled: bool,
    feedback_strength: float,
    screening_config: ScreeningConfig,
    use_semi_base: bool,
    property_feedback_evaluator: PropertyFeedbackEvaluator | None = None,
    property_feedback_warmup_iterations: int = 20,
    property_feedback_warmup_weight: float = 0.35,
    property_feedback_weight: float = 0.70,
    property_reward_slope: float = 2.5,
    partition_feedback_gain: float = 6.0,
    edge_feedback_gain: float = 2.0,
    feedback_exploration_fraction: float = 0.12,
    mmff_workers: int | None = None,
) -> IterativeGenerationResult:
    evaluations: list[GenerationEvaluation] = []
    results = []
    feedback_rounds: list[GenerationFeedbackRound] = []
    all_screened: list[ScreenedCandidate] = []
    online_property_rows: list[dict[str, object]] = []
    seen_smiles: set[str] = set()
    descriptor_cache: dict[str, object] = {}
    mmff_cache: dict[str, object] = {}
    novelty_cache: dict[str, object] = {}
    partition_weights = _initial_partition_weights(total_carbons)
    edge_feedback_biases: dict[str, float] = {}
    current_threshold = threshold
    current_temperature = temperature
    current_top_k = top_k
    for iteration in range(1, iterations + 1):
        _panel_stage(
            "generate",
            (
        f"Generating C{total_carbons} round {iteration}/{iterations}, "
        f"{samples} requests per round"
            ),
        )
        iteration_seed = None if seed is None else seed + iteration - 1
        evaluation = evaluate_generation(
            checkpoint_path=checkpoint_path,
            semi_base_checkpoint_path=semi_base_checkpoint_path,
            semi_base_threshold=semi_base_threshold,
            semi_base_temperature=semi_base_temperature,
            semi_base_top_k=semi_base_top_k,
            known_smiles_path=known_smiles_path,
            total_carbons=total_carbons,
            samples=samples,
            threshold=current_threshold,
            temperature=current_temperature,
            top_k=current_top_k,
            sample=feedback_enabled,
            seed=iteration_seed,
            use_semi_base=use_semi_base,
            partition_weights=partition_weights if feedback_enabled else None,
            edge_feedback_biases=(
                edge_feedback_biases if feedback_enabled else None
            ),
        )
        evaluations.append(evaluation)
        results.extend(
            replace(result, generation_iteration=iteration)
            for result in evaluation.results
        )
        screened, _ = screen_generation(
            evaluation,
            screening_config,
            known_smiles_path=known_smiles_path,
            descriptor_cache=descriptor_cache,
            mmff_cache=mmff_cache,
            novelty_cache=novelty_cache,
            mmff_workers=mmff_workers,
        )
        screened = tuple(
            replace(candidate, generation_iteration=iteration)
            for candidate in screened
        )
        all_screened.extend(screened)
        round_property_rows = (
            property_feedback_evaluator(screened, iteration)
            if feedback_enabled and property_feedback_evaluator is not None
            else []
        )
        online_property_rows.extend(round_property_rows)
        active_property_weight = (
            property_feedback_warmup_weight
            if (
                round_property_rows
                and property_feedback_warmup_iterations > 0
                and iteration <= property_feedback_warmup_iterations
            )
            else property_feedback_weight if round_property_rows else 0.0
        )
        repeated_count = sum(
            1
            for candidate in screened
            if candidate.canonical_smiles in seen_smiles
        )
        repeated_rate = _safe_ratio(repeated_count, len(screened))
        dynamic_exploration_fraction = min(
            0.35,
            feedback_exploration_fraction + 0.35 * repeated_rate,
        )
        feedback = _generation_feedback_round(
            iteration=iteration,
            requested_samples=samples,
            threshold=current_threshold,
            temperature=current_temperature,
            top_k=current_top_k,
            candidates=screened,
            config=screening_config,
            feedback_enabled=feedback_enabled,
            feedback_strength=feedback_strength,
            property_rows=round_property_rows,
            property_feedback_weight=active_property_weight,
            property_reward_slope=property_reward_slope,
            seen_smiles=seen_smiles,
            raw_results=evaluation.results,
            exploration_fraction=dynamic_exploration_fraction,
        )
        partition_weights = _update_partition_weights(
            partition_weights,
            screened,
            property_rows=round_property_rows,
            property_feedback_weight=active_property_weight,
            seen_smiles=seen_smiles,
            config=screening_config,
            feedback_strength=feedback_strength,
            property_reward_slope=property_reward_slope,
            partition_feedback_gain=partition_feedback_gain,
            exploration_fraction=dynamic_exploration_fraction,
            generation_results=evaluation.results,
        )
        edge_feedback_biases = _update_edge_feedback_biases(
            edge_feedback_biases,
            screened,
            property_rows=round_property_rows,
            property_feedback_weight=active_property_weight,
            property_reward_slope=property_reward_slope,
            seen_smiles=seen_smiles,
            config=screening_config,
            feedback_strength=feedback_strength,
            edge_feedback_gain=edge_feedback_gain,
            generation_results=evaluation.results,
        )
        favored_partition, favored_weight = _favored_partition(partition_weights)
        favored_edge_action, favored_edge_bias = _favored_edge_action(
            edge_feedback_biases
        )
        feedback = replace(
            feedback,
            favored_partition=favored_partition,
            favored_partition_weight=favored_weight,
            favored_edge_action=favored_edge_action,
            favored_edge_bias=favored_edge_bias,
        )
        feedback_rounds.append(feedback)
        seen_smiles.update(
            candidate.canonical_smiles
            for candidate in screened
            if candidate.canonical_smiles
        )
        current_threshold = feedback.next_threshold
        current_temperature = feedback.next_temperature
        current_top_k = feedback.next_top_k

    valid = [result for result in results if result.is_valid]
    connected = [result for result in valid if result.is_connected]
    novel = [result for result in connected if result.is_novel]
    duplicate_known = [result for result in connected if result.is_novel is False]
    unique_smiles = {
        result.canonical_smiles for result in valid if result.canonical_smiles
    }
    screening_summary = _aggregate_screening_summary(
        all_screened,
        total_generated=sum(
            evaluation.generated_bases for evaluation in evaluations
        ),
    )
    return IterativeGenerationResult(
        evaluation=GenerationEvaluation(
            total_carbons=total_carbons,
            requested_samples=samples * iterations,
            generated_bases=sum(evaluation.generated_bases for evaluation in evaluations),
            predicted_with_candidates=sum(
                evaluation.predicted_with_candidates for evaluation in evaluations
            ),
            predicted_with_edges=sum(
                evaluation.predicted_with_edges for evaluation in evaluations
            ),
            valid_molecules=len(valid),
            connected_molecules=len(connected),
            novel_molecules=len(novel),
            duplicate_known_molecules=len(duplicate_known),
            unique_valid_smiles=len(unique_smiles),
            failures=sum(evaluation.failures for evaluation in evaluations),
            use_semi_base=use_semi_base,
            results=tuple(results),
        ),
        feedback_rounds=tuple(feedback_rounds),
        screened_candidates=tuple(all_screened),
        screening_summary=screening_summary,
        online_property_rows=tuple(online_property_rows),
    )


def _generation_feedback_round(
    iteration: int,
    requested_samples: int,
    threshold: float,
    temperature: float,
    top_k: int | None,
    candidates,
    config: ScreeningConfig,
    feedback_enabled: bool,
    feedback_strength: float,
    property_rows: list[dict[str, object]] | None = None,
    property_feedback_weight: float = 0.0,
    property_reward_slope: float = 2.5,
    seen_smiles: set[str] | None = None,
    raw_results: tuple[object, ...] = (),
    exploration_fraction: float = 0.12,
) -> GenerationFeedbackRound:
    property_rows = [] if property_rows is None else property_rows
    seen_smiles = set() if seen_smiles is None else seen_smiles
    property_by_smiles = {
        str(row.get("canonical_smiles") or ""): row
        for row in property_rows
        if row.get("canonical_smiles")
    }
    rewards = []
    for candidate in candidates:
        structural_reward = _generation_reward_score(candidate, config)
        property_row = property_by_smiles.get(candidate.canonical_smiles)
        reward = _combined_online_reward(
            structural_reward,
            property_row,
            property_feedback_weight=property_feedback_weight,
            property_reward_slope=property_reward_slope,
        )
        if candidate.canonical_smiles in seen_smiles:
            reward *= 0.50
        rewards.append(round(_clamp(reward, 0.0, 1.0), 6))
    valid_count = len(candidates)
    passed_count = sum(1 for candidate in candidates if candidate.passes_screen)
    sa_values = [float(candidate.sa_score) for candidate in candidates]
    h_values = [float(candidate.h_to_c_ratio) for candidate in candidates]
    ring_values = [float(candidate.ring_count) for candidate in candidates]
    novelty_values = [
        float(candidate.fingerprint_novelty)
        for candidate in candidates
        if candidate.fingerprint_novelty is not None
    ]
    mmff_values = [float(candidate.mmff_stability_score) for candidate in candidates]
    property_quality_values = [
        float(row["property_quality_score"])
        for row in property_rows
        if row.get("property_quality_score") not in (None, "")
    ]
    composite_values = [
        float(row["composite_ranking_score"])
        for row in property_rows
        if row.get("composite_ranking_score") not in (None, "")
    ]
    property_objective_values = [
        objective
        for row in property_rows
        if (objective := _property_online_objective(
            row,
            property_reward_slope=property_reward_slope,
        ))
        is not None
    ]
    repeated_candidates = sum(
        1 for candidate in candidates if candidate.canonical_smiles in seen_smiles
    )
    raw_generated_count = len(raw_results) if raw_results else len(candidates)
    raw_valid_count = (
        sum(1 for result in raw_results if getattr(result, "is_valid", False))
        if raw_results
        else len(candidates)
    )
    raw_connected_count = (
        sum(1 for result in raw_results if getattr(result, "is_connected", False))
        if raw_results
        else sum(
            1 for candidate in candidates if getattr(candidate, "is_connected", True)
        )
    )
    sa_pass_count = sum(
        1 for candidate in candidates if candidate.sa_score <= config.max_sa_score
    )
    h_to_c_pass_count = sum(
        1
        for candidate in candidates
        if config.min_h_to_c <= candidate.h_to_c_ratio <= config.max_h_to_c
    )
    mmff_pass_count = sum(1 for candidate in candidates if candidate.mmff_embed_success)

    next_threshold, next_temperature, next_top_k, decision = _next_feedback_controls(
        threshold=threshold,
        temperature=temperature,
        top_k=top_k,
        candidates=candidates,
        rewards=rewards,
        config=config,
        feedback_enabled=feedback_enabled,
        feedback_strength=feedback_strength,
        property_rows=property_rows,
        property_feedback_weight=property_feedback_weight,
        property_reward_slope=property_reward_slope,
        duplicate_rate=_safe_ratio(repeated_candidates, valid_count),
        raw_valid_rate=_safe_ratio(raw_valid_count, raw_generated_count),
    )
    return GenerationFeedbackRound(
        iteration=iteration,
        requested_samples=requested_samples,
        threshold=round(float(threshold), 6),
        temperature=round(float(temperature), 6),
        top_k=top_k,
        valid_candidates=valid_count,
        raw_generated_candidates=raw_generated_count,
        raw_valid_candidates=raw_valid_count,
        raw_connected_candidates=raw_connected_count,
        raw_invalid_candidates=raw_generated_count - raw_valid_count,
        raw_valid_rate=_safe_ratio(raw_valid_count, raw_generated_count),
        passed_screen=passed_count,
        pass_rate=_safe_ratio(passed_count, valid_count),
        mean_reward_score=_mean(rewards),
        best_reward_score=max(rewards) if rewards else 0.0,
        min_sa_score=min(sa_values) if sa_values else None,
        mean_sa_score=_mean(sa_values) if sa_values else None,
        mean_h_to_c_ratio=_mean(h_values) if h_values else None,
        mean_ring_count=_mean(ring_values) if ring_values else None,
        mean_fingerprint_novelty=_mean(novelty_values) if novelty_values else None,
        mean_mmff_stability_score=_mean(mmff_values) if mmff_values else None,
        new_unique_candidates=valid_count - repeated_candidates,
        repeated_candidates=repeated_candidates,
        property_feedback_active=bool(
            property_rows and property_feedback_weight > 0.0
        ),
        property_feedback_weight=round(float(property_feedback_weight), 6),
        property_reward_slope=round(float(property_reward_slope), 6),
        property_evaluated_count=len(property_rows),
        property_pass_count=sum(
            1 for row in property_rows if row.get("hard_threshold_verdict") == "pass"
        ),
        property_review_count=sum(
            1 for row in property_rows if row.get("hard_threshold_verdict") == "review"
        ),
        property_reject_count=sum(
            1 for row in property_rows if row.get("hard_threshold_verdict") == "reject"
        ),
        mean_property_quality_score=(
            _mean(property_quality_values) if property_quality_values else None
        ),
        best_property_quality_score=(
            max(property_quality_values) if property_quality_values else None
        ),
        mean_composite_ranking_score=(
            _mean(composite_values) if composite_values else None
        ),
        mean_property_objective_score=(
            _mean(property_objective_values)
            if property_objective_values
            else None
        ),
        best_online_reward_score=(
            max(rewards)
            if rewards and property_rows and property_feedback_weight > 0.0
            else None
        ),
        favored_partition="",
        favored_partition_weight=None,
        favored_edge_action="",
        favored_edge_bias=None,
        exploration_fraction=round(float(exploration_fraction), 6),
        sa_pass_count=sa_pass_count,
        h_to_c_pass_count=h_to_c_pass_count,
        mmff_pass_count=mmff_pass_count,
        decision=decision,
        next_threshold=next_threshold,
        next_temperature=next_temperature,
        next_top_k=next_top_k,
    )


def _next_feedback_controls(
    threshold: float,
    temperature: float,
    top_k: int | None,
    candidates,
    rewards: list[float],
    config: ScreeningConfig,
    feedback_enabled: bool,
    feedback_strength: float,
    property_rows: list[dict[str, object]] | None = None,
    property_feedback_weight: float = 0.0,
    property_reward_slope: float = 2.5,
    duplicate_rate: float = 0.0,
    raw_valid_rate: float = 1.0,
) -> tuple[float, float, int | None, str]:
    if not feedback_enabled:
        return (
            round(float(threshold), 6),
            round(float(temperature), 6),
            top_k,
            "feedback_disabled",
        )
    if not candidates:
        return _control_tuple(
            threshold - 0.02 * feedback_strength,
            temperature + 0.05 * feedback_strength,
            _shift_top_k(top_k, 1),
            "explore_no_valid_candidates",
        )
    if raw_valid_rate < 0.80:
        return _control_tuple(
            threshold + 0.03 * feedback_strength,
            temperature + 0.03 * feedback_strength,
            _shift_top_k(top_k, -1),
            "penalize_invalid_structures",
        )
    if duplicate_rate >= 0.25:
        return _control_tuple(
            threshold - 0.02 * feedback_strength,
            temperature + 0.08 * feedback_strength,
            _shift_top_k(top_k, 1),
            "expand_search_on_repetition",
        )

    total = len(candidates)
    passed = sum(1 for candidate in candidates if candidate.passes_screen)
    low_h_rate = _safe_ratio(
        sum(1 for candidate in candidates if candidate.h_to_c_ratio < config.min_h_to_c),
        total,
    )
    high_sa_rate = _safe_ratio(
        sum(1 for candidate in candidates if candidate.sa_score > config.max_sa_score),
        total,
    )
    mmff_fail_rate = _safe_ratio(
        sum(1 for candidate in candidates if not candidate.mmff_embed_success),
        total,
    )
    best_reward = max(rewards) if rewards else 0.0
    mean_reward = _mean(rewards)
    property_rows = [] if property_rows is None else property_rows
    if property_rows and property_feedback_weight > 0.0:
        property_step = feedback_strength * max(0.25, property_feedback_weight)
        elite_top_k = _elite_selected_count(property_rows, top_k)
        pass_count = sum(
            1 for row in property_rows if row.get("hard_threshold_verdict") == "pass"
        )
        review_count = sum(
            1
            for row in property_rows
            if row.get("hard_threshold_verdict") == "review"
        )
        objective_values = [
            objective
            for row in property_rows
            if (objective := _property_online_objective(
                row,
                property_reward_slope=property_reward_slope,
            ))
            is not None
        ]
        mean_objective = _mean(objective_values)
        if pass_count:
            return _control_tuple(
                threshold + 0.02 * property_step,
                temperature - 0.08 * property_step,
                elite_top_k,
                "exploit_property_pass_elites",
            )
        if review_count and mean_objective >= 0.45:
            return _control_tuple(
                threshold + 0.005 * property_step,
                temperature - 0.03 * property_step,
                elite_top_k,
                "refine_property_review_candidates",
            )
        return _control_tuple(
            threshold - 0.04 * property_step,
            temperature + 0.10 * property_step,
            _shift_top_k(elite_top_k, 1),
            "explore_property_rejected_round",
        )

    if passed:
        return _control_tuple(
            threshold + 0.01 * feedback_strength,
            temperature - 0.02 * feedback_strength,
            top_k,
            "exploit_passed_candidates",
        )
    if low_h_rate >= 0.20:
        return _control_tuple(
            threshold + 0.05 * feedback_strength,
            temperature - 0.05 * feedback_strength,
            _shift_top_k(top_k, -1),
            "reward_h_to_c_guard_reduce_bonds",
        )
    if high_sa_rate >= 0.80:
        return _control_tuple(
            threshold + 0.04 * feedback_strength,
            temperature - 0.04 * feedback_strength,
            _shift_top_k(top_k, -1),
            "reward_sa_guard_reduce_complexity",
        )
    if mmff_fail_rate >= 0.35:
        return _control_tuple(
            threshold + 0.03 * feedback_strength,
            temperature - 0.03 * feedback_strength,
            _shift_top_k(top_k, -1),
            "reward_mmff_guard_reduce_complexity",
        )
    if best_reward >= 0.80:
        return _control_tuple(
            threshold + 0.01 * feedback_strength,
            temperature - 0.01 * feedback_strength,
            top_k,
            "exploit_high_reward_candidates",
        )
    if mean_reward < 0.55:
        return _control_tuple(
            threshold - 0.02 * feedback_strength,
            temperature + 0.04 * feedback_strength,
            _shift_top_k(top_k, 1),
            "explore_low_reward_round",
        )
    return _control_tuple(
        threshold,
        temperature + 0.02 * feedback_strength,
        top_k,
        "balanced_small_explore",
    )


def _control_tuple(
    threshold: float,
    temperature: float,
    top_k: int | None,
    decision: str,
) -> tuple[float, float, int | None, str]:
    return (
        round(_clamp(threshold, 0.35, 0.75), 6),
        round(_clamp(temperature, 0.65, 1.25), 6),
        top_k,
        decision,
    )


def _shift_top_k(top_k: int | None, delta: int) -> int | None:
    if top_k is None:
        return None
    return int(_clamp(top_k + delta, 1, 4))


def _generation_reward_score(candidate, config: ScreeningConfig) -> float:
    sa_score = synthetic_accessibility_property_score(
        _candidate_float(candidate, "sa_score", config.max_sa_score)
    )
    h_score = _range_score(
        _candidate_float(candidate, "h_to_c_ratio", config.target_h_to_c),
        config.min_h_to_c,
        config.target_h_to_c,
        config.max_h_to_c,
    )
    ring_score = _range_score(
        _candidate_float(candidate, "ring_count", float(config.target_ring_count)),
        float(config.min_ring_count),
        float(config.target_ring_count),
        float(config.max_ring_count),
    )
    mmff_score = (
        min(max(_candidate_float(candidate, "mmff_stability_score", 0.5), 0.0), 1.0)
        if bool(getattr(candidate, "mmff_embed_success", True))
        else 0.0
    )
    novelty_score = (
        min(max(_candidate_float(candidate, "fingerprint_novelty", 0.0), 0.0), 1.0)
        if getattr(candidate, "fingerprint_novelty", None) is not None
        else 0.0
    )
    valid_bonus = 1.0 if bool(getattr(candidate, "is_valid", True)) else 0.0
    connected_bonus = (
        1.0 if bool(getattr(candidate, "is_connected", True)) else 0.0
    )
    mmff_embed_bonus = (
        1.0 if bool(getattr(candidate, "mmff_embed_success", False)) else 0.0
    )
    pass_bonus = 1.0 if getattr(candidate, "passes_screen", False) else 0.0
    reward = (
        0.10 * valid_bonus
        + 0.10 * connected_bonus
        + 0.15 * sa_score
        + 0.10 * h_score
        + 0.10 * ring_score
        + 0.10 * mmff_score
        + 0.05 * mmff_embed_bonus
        + 0.10 * novelty_score
        + 0.20 * pass_bonus
    )
        # Stronger penalty: acyclic molecules (very low ring score) and
        # library-known molecules (very low novelty) waste sampling budget;
        # extra penalty discourages the generator from reproducing them.
    ring_count = _candidate_float(candidate, "ring_count", 0.0)
    if ring_count < float(config.min_ring_count):
        reward -= 0.15
    is_novel = getattr(candidate, "is_novel", None)
    if is_novel is False:
        reward -= 0.15
    return round(float(_clamp(reward, 0.0, 1.0)), 6)


def _combined_online_reward(
    structural_reward: float,
    property_row: dict[str, object] | None,
    *,
    property_feedback_weight: float,
    property_reward_slope: float = 2.5,
) -> float:
    if property_row is None or property_feedback_weight <= 0.0:
        return structural_reward
    online_objective = _property_online_objective(
        property_row,
        property_reward_slope=property_reward_slope,
    )
    if online_objective is None:
        return structural_reward
    weight = _clamp(property_feedback_weight, 0.0, 1.0)
    return (1.0 - weight) * structural_reward + weight * online_objective


def _property_online_objective(
    property_row: dict[str, object],
    *,
    property_reward_slope: float,
) -> float | None:
    composite_score = _optional_float(
        property_row.get("composite_ranking_score")
    )
    if composite_score is None:
        return None
    shaped_score = _shape_property_reward(
        composite_score,
        slope=property_reward_slope,
    )
    route_score = {
        "pass": 1.0,
        "review": 0.45,
        "reject": 0.0,
        "missing_properties": 0.10,
        "property_failed": 0.0,
    }.get(str(property_row.get("hard_threshold_verdict") or ""), 0.20)
    return 0.65 * shaped_score + 0.35 * route_score


def _shape_property_reward(
    score: float,
    *,
    slope: float,
    center: float = 0.65,
) -> float:
    """Expand the narrow score band around the useful-candidate boundary."""
    bounded_score = _clamp(score, 0.0, 1.0)
    return _clamp(0.5 + slope * (bounded_score - center), 0.0, 1.0)


def _elite_selected_count(
    property_rows: list[dict[str, object]],
    fallback_top_k: int | None,
) -> int | None:
    scored_rows = [
        row
        for row in property_rows
        if row.get("composite_ranking_score") not in (None, "")
        and row.get("selected_count") not in (None, "")
    ]
    if not scored_rows:
        return fallback_top_k
    scored_rows.sort(
        key=lambda row: float(row["composite_ranking_score"]),
        reverse=True,
    )
    elite_count = max(1, math.ceil(len(scored_rows) * 0.25))
    selected_counts = sorted(
        max(1, int(float(row["selected_count"])))
        for row in scored_rows[:elite_count]
    )
    middle = len(selected_counts) // 2
    if len(selected_counts) % 2:
        selected_count = selected_counts[middle]
    else:
        selected_count = round(
            (selected_counts[middle - 1] + selected_counts[middle]) / 2
        )
    return int(_clamp(selected_count, 1, 4))


def _initial_partition_weights(
    total_carbons: int,
) -> dict[tuple[int, int, int], float]:
    partitions = [
        counts
        for counts in iter_count_partitions(total_carbons)
        if counts[0] + counts[1] // 2 > 0
    ]
    if not partitions:
        return {}
    uniform = 1.0 / len(partitions)
    return {partition: uniform for partition in partitions}


def _update_partition_weights(
    current_weights: dict[tuple[int, int, int], float],
    candidates: tuple[ScreenedCandidate, ...],
    *,
    property_rows: list[dict[str, object]],
    property_feedback_weight: float,
    seen_smiles: set[str],
    config: ScreeningConfig,
    feedback_strength: float,
    property_reward_slope: float = 2.5,
    partition_feedback_gain: float = 6.0,
    exploration_fraction: float = 0.12,
    generation_results: tuple[object, ...] = (),
) -> dict[tuple[int, int, int], float]:
    if not current_weights or (not candidates and not generation_results):
        return current_weights
    property_by_smiles = {
        str(row.get("canonical_smiles") or ""): row
        for row in property_rows
        if row.get("canonical_smiles")
    }
    grouped_rewards: dict[tuple[int, int, int], list[float]] = {}
    all_rewards: list[float] = []
    feedback_items: list[tuple[tuple[int, int, int], float]] = []
    if generation_results:
        candidate_by_index = {candidate.index: candidate for candidate in candidates}
        for index, result in enumerate(generation_results, start=1):
            candidate = candidate_by_index.get(index)
            if candidate is None:
                reward = 0.0
            else:
                reward = _combined_online_reward(
                    _generation_reward_score(candidate, config),
                    property_by_smiles.get(candidate.canonical_smiles),
                    property_feedback_weight=property_feedback_weight,
                    property_reward_slope=property_reward_slope,
                )
                if candidate.canonical_smiles in seen_smiles:
                    reward *= 0.50
            feedback_items.append(
                ((result.n1, result.n2, result.n3), reward)
            )
    else:
        for candidate in candidates:
            reward = _combined_online_reward(
                _generation_reward_score(candidate, config),
                property_by_smiles.get(candidate.canonical_smiles),
                property_feedback_weight=property_feedback_weight,
                property_reward_slope=property_reward_slope,
            )
            if candidate.canonical_smiles in seen_smiles:
                reward *= 0.50
            feedback_items.append(((candidate.n1, candidate.n2, candidate.n3), reward))

    for partition, reward in feedback_items:
        grouped_rewards.setdefault(partition, []).append(reward)
        all_rewards.append(reward)
    if not all_rewards:
        return current_weights

    baseline = sum(all_rewards) / len(all_rewards)
    learning_rate = _clamp(
        0.20 + 0.20 * feedback_strength + 0.20 * property_feedback_weight,
        0.10,
        0.65,
    )
    updated: dict[tuple[int, int, int], float] = {}
    for partition, old_weight in current_weights.items():
        partition_rewards = grouped_rewards.get(partition)
        if partition_rewards:
            partition_mean = sum(partition_rewards) / len(partition_rewards)
            multiplier = math.exp(
                partition_feedback_gain
                * learning_rate
                * (partition_mean - baseline)
            )
        else:
            multiplier = 1.0
        updated[partition] = max(1e-12, old_weight * multiplier)

    total = sum(updated.values())
    uniform = 1.0 / len(updated)
    exploration_fraction = _clamp(exploration_fraction, 0.0, 0.95)
    return {
        partition: (
            exploration_fraction * uniform
            + (1.0 - exploration_fraction) * weight / total
        )
        for partition, weight in updated.items()
    }


def _update_edge_feedback_biases(
    current_biases: dict[str, float],
    candidates: tuple[ScreenedCandidate, ...],
    *,
    property_rows: list[dict[str, object]],
    property_feedback_weight: float,
    property_reward_slope: float,
    seen_smiles: set[str],
    config: ScreeningConfig,
    feedback_strength: float,
    edge_feedback_gain: float,
    generation_results: tuple[object, ...] = (),
) -> dict[str, float]:
    if not candidates and not generation_results:
        return {
            key: value * 0.98
            for key, value in current_biases.items()
            if abs(value * 0.98) >= 1e-4
        }
    property_by_smiles = {
        str(row.get("canonical_smiles") or ""): row
        for row in property_rows
        if row.get("canonical_smiles")
    }
    action_rewards: dict[str, list[float]] = {}
    feedback_items: list[
        tuple[int, int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], float]
    ] = []
    if generation_results:
        candidate_by_index = {candidate.index: candidate for candidate in candidates}
        for index, result in enumerate(generation_results, start=1):
            candidate = candidate_by_index.get(index)
            if candidate is None:
                reward = 0.0
            else:
                reward = _combined_online_reward(
                    _generation_reward_score(candidate, config),
                    property_by_smiles.get(candidate.canonical_smiles),
                    property_feedback_weight=property_feedback_weight,
                    property_reward_slope=property_reward_slope,
                )
                if candidate.canonical_smiles in seen_smiles:
                    reward *= 0.50
            feedback_items.append(
                (
                    result.n1,
                    result.n2,
                    result.n3,
                    result.edges_base,
                    result.predicted_edges,
                    reward,
                )
            )
    else:
        for candidate in candidates:
            reward = _combined_online_reward(
                _generation_reward_score(candidate, config),
                property_by_smiles.get(candidate.canonical_smiles),
                property_feedback_weight=property_feedback_weight,
                property_reward_slope=property_reward_slope,
            )
            if candidate.canonical_smiles in seen_smiles:
                reward *= 0.50
            feedback_items.append(
                (
                    candidate.n1,
                    candidate.n2,
                    candidate.n3,
                    candidate.edges_base,
                    candidate.predicted_edges,
                    reward,
                )
            )

    for n1, n2, n3, edges_base, predicted_edges, reward in feedback_items:
        if not predicted_edges:
            continue
        graph = HalfGraph(
            n1,
            n2,
            n3,
            edges_base,
        )
        for edge in predicted_edges:
            key = edge_feedback_key(graph, edge)
            action_rewards.setdefault(key, []).append(reward)

    updated = {
        key: value * 0.98
        for key, value in current_biases.items()
        if abs(value * 0.98) >= 1e-4
    }
    step = 0.15 * feedback_strength * edge_feedback_gain
    reward_center = 0.55
    for key, rewards in action_rewards.items():
        action_mean = sum(rewards) / len(rewards)
        updated[key] = _clamp(
            updated.get(key, 0.0) + step * (action_mean - reward_center),
            -1.5,
            1.5,
        )
    return updated


def _favored_edge_action(
    edge_feedback_biases: dict[str, float],
) -> tuple[str, float | None]:
    if not edge_feedback_biases:
        return "", None
    action, bias = max(edge_feedback_biases.items(), key=lambda item: item[1])
    return action, round(float(bias), 6)


def _favored_partition(
    partition_weights: dict[tuple[int, int, int], float],
) -> tuple[str, float | None]:
    if not partition_weights:
        return "", None
    partition, weight = max(partition_weights.items(), key=lambda item: item[1])
    return (
        f"n1={partition[0]},n2={partition[1]},n3={partition[2]}",
        round(float(weight), 6),
    )


def _aggregate_screening_summary(
    candidates: list[ScreenedCandidate],
    *,
    total_generated: int,
) -> ScreeningSummary:
    passed = [candidate for candidate in candidates if candidate.passes_screen]
    high_novelty = [
        candidate for candidate in candidates if candidate.high_novelty is True
    ]
    return ScreeningSummary(
        total_generated=total_generated,
        valid_candidates=len(candidates),
        screened_candidates=len(candidates),
        passed_screen=len(passed),
        connected_candidates=sum(
            1 for candidate in candidates if candidate.is_connected
        ),
        novel_candidates=sum(1 for candidate in candidates if candidate.is_novel),
        high_novelty_candidates=len(high_novelty),
        high_novelty_passed_screen=sum(
            1 for candidate in high_novelty if candidate.passes_screen
        ),
        unique_screened_smiles=len(
            {candidate.canonical_smiles for candidate in candidates}
        ),
        mean_screen_score=(
            round(
                sum(candidate.screen_score for candidate in candidates)
                / len(candidates),
                6,
            )
            if candidates
            else 0.0
        ),
        ring_count_definition=RING_COUNT_DEFINITION,
    )


def _candidate_float(candidate, field: str, default: float) -> float:
    try:
        value = getattr(candidate, field)
    except AttributeError:
        return float(default)
    if value is None or value == "":
        return float(default)
    return float(value)


def _range_score(value: float, low: float, target: float, high: float) -> float:
    if high <= low:
        return 0.0
    if value < low or value > high:
        return 0.0
    if value <= target:
        return _safe_ratio(value - low, target - low)
    return _safe_ratio(high - value, high - target)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(sum(values) / len(values)), 6)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


# Below this many pending molecules the process-pool start-up cost dominates.
PARALLEL_PROPERTY_MIN_ITEMS = 16

# Per-worker context for the property stage. Populated in the parent for the
# serial path and by the pool initializer for the parallel path, so both paths
# execute exactly the same function with exactly the same arguments.
_PROPERTY_WORKER_CONTEXT: dict[str, object] = {}


def _property_worker_init(context: dict[str, object]) -> None:
    _PROPERTY_WORKER_CONTEXT.clear()
    _PROPERTY_WORKER_CONTEXT.update(context)
    if int(context.get("__workers__") or 1) > 1:
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass
        # Worker processes already run in parallel, so keep RDKit's internal
        # symmetry-aware RMSD search single-threaded inside each of them.
        try:
            from property_model.e3nn_geometry_scorer import set_align_threads

            set_align_threads(1)
        except Exception:
            pass


def _property_delta_worker(
    payload: tuple[
        dict[str, object],
        int,
        dict[str, object] | None,
        dict[str, tuple[GeometryScore, float | None]] | None,
    ],
) -> tuple[str, object, dict[str, object]]:
    """Predict one molecule and return only the fields it added.

    Returns ``("ok", delta, {})`` on success and
    ``("error", message, partial_delta)`` on failure. The partial delta keeps the
    fields that were already computed before the failure, matching the serial
    behaviour of the original loop.
    """

    row, index, sta_override, geometry_override = payload
    context = dict(_PROPERTY_WORKER_CONTEXT)
    base_seed = int(context.pop("__seed__", 0))
    context.pop("__workers__", None)
    enriched = dict(row)
    base_keys = set(enriched)

    def _delta() -> dict[str, object]:
        return {key: value for key, value in enriched.items() if key not in base_keys}

    try:
        _add_external_property_predictions(
            enriched,
            seed=base_seed + index,
            sta_predictions_override=sta_override,
            geometry_scores_override=geometry_override,
            **context,  # type: ignore[arg-type]
        )
    except Exception as exc:  # pragma: no cover - delegated to the caller
        set_reproducibility_enabled(True)
        return ("error", str(exc), _delta())
    return ("ok", _delta(), {})


def _parallel_property_deltas(
    payloads: list[
        tuple[
            dict[str, object],
            int,
            dict[str, object] | None,
            dict[str, tuple[GeometryScore, float | None]] | None,
        ]
    ],
    context: dict[str, object],
    workers: int,
) -> list[tuple[str, object, dict[str, object]]]:
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_property_worker_init,
        initargs=(context,),
    ) as pool:
        return list(
            pool.map(
                _property_delta_worker,
                payloads,
                chunksize=max(1, len(payloads) // (workers * 8)),
            )
        )


def _build_property_rows(
    rows: list[dict[str, object]],
    heating_values: dict[str, float],
    uncertainty_scale: float,
    thresholds: FuelPropertyThresholds,
    mp_model_path: str | Path,
    mp_data_path: str | Path,
    e3nn_geometry_checkpoint: str | Path | None,
    e3nn_property_checkpoint: str | Path | None,
    e3nn_property_name: str,
    sta_property_model_path: str | Path | None,
    sta_density_model_path: str | Path | None,
    sta_heating_value_model_path: str | Path | None,
    sta_melting_point_model_path: str | Path | None,
    sta_boiling_point_model_path: str | Path | None,
    sta_flash_point_model_path: str | Path | None,
    sta_formation_enthalpy_model_path: str | Path | None,
    sta_molar_volume_model_path: str | Path | None,
    sta_isp_model_path: str | Path | None,
    density_model: str,
    heating_value_model: str,
    melting_point_model: str,
    boiling_point_model: str,
    flash_point_model: str,
    formation_enthalpy_model: str,
    molar_volume_model: str,
    isp_model: str,
    cbh_bdc_model_path: str | Path,
    cbh_bdc_feature_csv: str | Path,
    cbh_bdc_yang_root: str | Path,
    phase_fusion_melting_point_model_path: str | Path | None,
    phase_fusion_boiling_point_model_path: str | Path | None,
    phase_fusion_flash_point_model_path: str | Path | None,
    use_cbh_bdc_heating: bool,
    objective_mode: str,
    score_weights: CompositeScoreWeights,
    property_score_weights: PropertyScoreWeights,
    property_targets: PropertyTargetValues,
    objective_relative_window: float,
    seed: int,
    ignore_density: bool = False,
    ignore_heating_value: bool = False,
    use_melting_point_screen: bool = False,
    use_boiling_point_screen: bool = False,
    use_flash_point_screen: bool = False,
    use_volumetric_heating_value_screen: bool = False,
    missing_property_penalty: float = 0.15,
    property_prediction_cache: dict[str, dict[str, object]] | None = None,
    property_workers: int | None = None,
    property_cache_path: str | Path | None = None,
    jp10_benchmark: object | None = None,
    reproducibility_scope: str = "tier12",
) -> list[dict[str, object]]:
    predict_kwargs: dict[str, object] = dict(
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
        jp10_benchmark=jp10_benchmark,
        reproducibility_scope=reproducibility_scope,
    )

    # ---- 1. plan the work: which molecules still need a prediction ---------
    prepared: list[tuple[dict[str, object], str, int]] = []
    first_index: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        if row.get("passes_screen") is not True:
            continue
        smiles = str(row.get("canonical_smiles") or "").strip()
        prepared.append((row, smiles, index))
        if smiles and smiles not in first_index:
            first_index[smiles] = index

    cache_path = Path(property_cache_path) if property_cache_path else None
    fingerprint = config_fingerprint(predict_kwargs, seed=seed)
    disk_cache = PropertyCache(cache_path, fingerprint)
    memory_cache = property_prediction_cache

    deltas: dict[str, dict[str, object]] = {}
    errors: dict[str, tuple[str, dict[str, object]]] = {}
    pending: list[tuple[str, int]] = []
    for smiles, index in first_index.items():
        if memory_cache is not None and smiles in memory_cache:
            continue
        cached = disk_cache.get(smiles, seed + index)
        if cached is not None:
            deltas[smiles] = cached
            continue
        pending.append((smiles, index))
    if disk_cache.entries:
        _panel_stage(
            "property",
            "Property cache: %d molecules reused (%s), %d to compute"
            % (len(deltas), disk_cache.describe(), len(pending)),
        )

    # ---- 2. batched STA pre-pass (one forward pass per property head) ------
    sta_overrides: dict[str, dict[str, object]] = {}
    sta_model_paths = _selected_sta_model_paths(
        fallback_model_path=sta_property_model_path,
        density_model=density_model,
        heating_value_model=heating_value_model,
        melting_point_model=melting_point_model,
        boiling_point_model=boiling_point_model,
        flash_point_model=flash_point_model,
        formation_enthalpy_model=formation_enthalpy_model,
        molar_volume_model=molar_volume_model,
        isp_model=isp_model,
        sta_density_model_path=sta_density_model_path,
        sta_heating_value_model_path=sta_heating_value_model_path,
        sta_melting_point_model_path=sta_melting_point_model_path,
        sta_boiling_point_model_path=sta_boiling_point_model_path,
        sta_flash_point_model_path=sta_flash_point_model_path,
        sta_formation_enthalpy_model_path=sta_formation_enthalpy_model_path,
        sta_molar_volume_model_path=sta_molar_volume_model_path,
        sta_isp_model_path=sta_isp_model_path,
    )
    if pending and sta_model_paths:
        try:
            batch = _batch_sta_predictions_for_smiles(
                [smiles for smiles, _ in pending],
                sta_model_paths,
            )
        except Exception:
            batch = {}
        # The single-model path does not emit these two provenance columns; drop
        # them so the output schema stays exactly as before.
        model_label = ";".join(dict.fromkeys(str(path) for path in sta_model_paths))
        for smiles, values in batch.items():
            values.pop("sta_inference_device", None)
            values.pop("sta_batch_size", None)
            values["sta_status"] = "predicted"
            values["sta_model"] = model_label
            sta_overrides[smiles] = values

    # ---- 3. per-molecule prediction (parallel when asked) ------------------
    if pending:
        payloads = [
            (dict(rows[index - 1]), index, sta_overrides.get(smiles), None)
            for smiles, index in pending
        ]
        workers = resolve_worker_count(property_workers, memory_per_worker_gb=1.3)
        if len(payloads) < PARALLEL_PROPERTY_MIN_ITEMS:
            workers = 1
        context = dict(predict_kwargs)
        context["__seed__"] = seed
        context["__workers__"] = workers
        if workers > 1:
            _panel_stage(
                "property",
                "Predicting properties for %d molecules on %d worker processes"
                % (len(payloads), workers),
            )
            outcomes = _parallel_property_deltas(payloads, context, workers)
        else:
            _property_worker_init(context)
            outcomes = [_property_delta_worker(payload) for payload in payloads]
        for (smiles, index), (status, payload, partial) in zip(pending, outcomes):
            if status == "ok":
                deltas[smiles] = payload  # type: ignore[assignment]
                disk_cache.put(smiles, seed + index, payload)  # type: ignore[arg-type]
            else:
                errors[smiles] = (str(payload), partial)
        disk_cache.save(force=True)

    # ---- 4. assemble rows in the original order ---------------------------
    property_rows: list[dict[str, object]] = []
    for row, smiles, _index in prepared:
        enriched = dict(row)
        cache_hit = bool(
            memory_cache is not None
            and smiles
            and smiles in memory_cache
        )
        try:
            if cache_hit:
                enriched.update(memory_cache[smiles])  # type: ignore[index]
            else:
                delta = deltas.get(smiles)
                if delta is None:
                    message, partial = errors.get(smiles, ("unknown property failure", {}))
                    enriched.update(partial)
                    raise RuntimeError(message)
                enriched.update(delta)
                if memory_cache is not None and smiles:
                    memory_cache[smiles] = delta
            enriched["property_prediction_cache_hit"] = cache_hit
            _route_enriched_property_row(
                enriched,
                uncertainty_scale=uncertainty_scale,
                thresholds=thresholds,
                ignore_density=ignore_density,
                ignore_heating_value=ignore_heating_value or heating_value_model == "none",
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
                objective_mode=objective_mode,
                thresholds=thresholds,
                score_weights=score_weights,
                property_score_weights=property_score_weights,
                property_targets=property_targets,
                objective_relative_window=objective_relative_window,
                missing_property_penalty=missing_property_penalty,
            )
            if jp10_benchmark is not None:
                apply_tier_fields(enriched, jp10_benchmark)  # type: ignore[arg-type]
        except Exception as exc:
            enriched["property_stage_status"] = "failed"
            enriched["hard_threshold_verdict"] = "property_failed"
            enriched["property_failure_reason"] = str(exc)
            _add_objective_metadata(
                enriched,
                objective_mode=objective_mode,
                property_targets=property_targets,
            )
        property_rows.append(enriched)
    return property_rows


def _add_objective_scores(
    row: dict[str, object],
    objective_mode: str,
    thresholds: FuelPropertyThresholds,
    score_weights: CompositeScoreWeights,
    property_score_weights: PropertyScoreWeights,
    property_targets: PropertyTargetValues,
    objective_relative_window: float,
    missing_property_penalty: float = 0.15,
) -> None:
    density = _optional_float(row.get("density_g_cm3")) or _optional_float(
        row.get("external_density_g_cm3")
    )
    heating_value = _optional_float(row.get("heating_value_mj_kg")) or _optional_float(
        row.get("external_heating_value_mj_kg")
    )
    volumetric_heating_value = _optional_float(
        row.get("volumetric_heating_value_mj_l")
    )
    melting_point = _optional_float(row.get("melting_point_k")) or _optional_float(
        row.get("external_melting_point_k")
    )
    boiling_point = _optional_float(row.get("boiling_point_k")) or _optional_float(
        row.get("external_boiling_point_k")
    )
    flash_point = _optional_float(row.get("flash_point_k")) or _optional_float(
        row.get("external_flash_point_k")
    )
    formation_enthalpy = _optional_float(
        row.get("formation_enthalpy_kj_mol")
    ) or _optional_float(row.get("external_formation_enthalpy_kj_mol"))
    molar_volume = _optional_float(row.get("molar_volume_cm3_mol")) or _optional_float(
        row.get("external_molar_volume_cm3_mol")
    )
    isp = _optional_float(row.get("isp")) or _optional_float(row.get("external_isp"))
    sa_score = _optional_float(row.get("sa_score"))
    structure_score = _structure_validity_score(row)
    row["structure_validity_score"] = structure_score

    volumetric_mode = (
        property_score_weights.volumetric_heating_value > 0.0
        or property_targets.volumetric_heating_value is not None
    )
    effective_weights = (
        replace(
            property_score_weights,
            density=0.0,
            heating_value=0.0,
        )
        if volumetric_mode
        else property_score_weights
    )
    if objective_mode == "target_property":
        property_score = target_property_score(
            density_g_cm3=density,
            heating_value_mj_kg=heating_value,
            melting_point_k=melting_point,
            volumetric_heating_value_mj_l=volumetric_heating_value,
            boiling_point_k=boiling_point,
            flash_point_k=flash_point,
            sa_score=sa_score,
            structure_validity_score=structure_score,
            formation_enthalpy_kj_mol=formation_enthalpy,
            molar_volume_cm3_mol=molar_volume,
            isp=isp,
            targets=property_targets,
            weights=effective_weights,
            relative_window=objective_relative_window,
        )
    else:
        property_score = property_quality_score(
            density_g_cm3=density,
            heating_value_mj_kg=heating_value,
            melting_point_k=melting_point,
            volumetric_heating_value_mj_l=volumetric_heating_value,
            boiling_point_k=boiling_point,
            flash_point_k=flash_point,
            sa_score=sa_score,
            structure_validity_score=structure_score,
            formation_enthalpy_kj_mol=formation_enthalpy,
            molar_volume_cm3_mol=molar_volume,
            isp=isp,
            thresholds=thresholds,
            weights=effective_weights,
            relative_window=objective_relative_window,
        )
    missing_properties = _selected_missing_properties(
        objective_mode=objective_mode,
        weights=effective_weights,
        targets=property_targets,
        values={
            "density": density,
            "heating_value": heating_value,
            "volumetric_heating_value": volumetric_heating_value,
            "melting_point": melting_point,
            "boiling_point": boiling_point,
            "flash_point": flash_point,
            "synthetic_accessibility": sa_score,
            "structure_validity": structure_score,
        },
    )
    penalty_factor = max(
        0.0,
        1.0 - missing_property_penalty * len(missing_properties),
    )
    property_score = round(property_score * penalty_factor, 6)
    _add_objective_metadata(
        row,
        objective_mode=objective_mode,
        property_targets=property_targets,
    )
    row["objective_relative_window"] = objective_relative_window
    row["volumetric_heating_value_mode"] = volumetric_mode
    row["missing_property_names"] = ";".join(missing_properties)
    row["missing_property_count"] = len(missing_properties)
    row["missing_property_penalty_per_property"] = missing_property_penalty
    row["missing_property_penalty_factor"] = round(penalty_factor, 6)
    row["property_quality_score"] = property_score
    row["composite_ranking_score"] = composite_ranking_score(
        property_quality_score=property_score,
        novelty_score=float(row.get("fingerprint_novelty") or 0.0),
        geometry_score=_optional_float(row.get("e3nn_geometry_score")),
        symmetry_score=_optional_float(row.get("symmetry_consistency")),
        sa_score=_optional_float(row.get("sa_score")),
        weights=score_weights,
    )
    if missing_properties:
        row["property_stage_status"] = "missing_properties"
        row["hard_threshold_verdict"] = "missing_properties"
        row["final_route_reason"] = "selected_property_prediction_missing"


def _structure_validity_score(row: dict[str, object]) -> float:
    rdkit_valid = 1.0 if str(row.get("canonical_smiles") or "").strip() else 0.0
    connected = 1.0 if row.get("is_connected") is True else 0.0
    mmff = _optional_float(
        row.get("mmff_stability_score")
    ) or _optional_float(row.get("mmff_stability"))
    mmff_score = 0.0 if mmff is None else min(max(mmff, 0.0), 1.0)
    return round((rdkit_valid + connected + mmff_score) / 3.0, 6)


def _selected_missing_properties(
    *,
    objective_mode: str,
    weights: PropertyScoreWeights,
    targets: PropertyTargetValues,
    values: dict[str, float | None],
) -> list[str]:
    weighted_names = {
        "density": weights.density,
        "heating_value": weights.heating_value,
        "volumetric_heating_value": weights.volumetric_heating_value,
        "melting_point": weights.melting_point,
        "boiling_point": weights.boiling_point,
        "flash_point": weights.flash_point,
        "synthetic_accessibility": weights.synthetic_accessibility,
        "structure_validity": weights.structure_validity,
    }
    target_names = {
        "density": targets.density,
        "heating_value": targets.heating_value,
        "volumetric_heating_value": targets.volumetric_heating_value,
        "melting_point": targets.melting_point,
        "boiling_point": targets.boiling_point,
        "flash_point": targets.flash_point,
    }
    selected = {
        name
        for name, weight in weighted_names.items()
        if weight > 0.0
    }
    if objective_mode == "target_property":
        selected = {
            name for name, target in target_names.items() if target is not None
        }
        selected.update(
            name
            for name in ("synthetic_accessibility", "structure_validity")
            if weighted_names[name] > 0.0
        )
    return sorted(name for name in selected if values.get(name) is None)


def _add_objective_metadata(
    row: dict[str, object],
    objective_mode: str,
    property_targets: PropertyTargetValues,
) -> None:
    row["objective_mode"] = objective_mode
    row["property_target_density_g_cm3"] = (
        "" if property_targets.density is None else property_targets.density
    )
    row["property_target_heating_value_mj_kg"] = (
        "" if property_targets.heating_value is None else property_targets.heating_value
    )
    row["property_target_volumetric_heating_value_mj_l"] = (
        ""
        if property_targets.volumetric_heating_value is None
        else property_targets.volumetric_heating_value
    )
    row["property_target_melting_point_k"] = (
        "" if property_targets.melting_point is None else property_targets.melting_point
    )
    row["property_target_boiling_point_k"] = (
        "" if property_targets.boiling_point is None else property_targets.boiling_point
    )
    row["property_target_flash_point_k"] = (
        "" if property_targets.flash_point is None else property_targets.flash_point
    )


def _add_external_property_predictions(
    row: dict[str, object],
    heating_values: dict[str, float],
    mp_model_path: str | Path,
    mp_data_path: str | Path,
    e3nn_geometry_checkpoint: str | Path | None,
    seed: int,
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
    sta_predictions_override: dict[str, object] | None = None,
    geometry_scores_override: dict[str, tuple[GeometryScore, float | None]] | None = None,
    jp10_benchmark: object | None = None,
    reproducibility_scope: str = "tier12",
) -> None:
    smiles = str(row.get("canonical_smiles") or "").strip()
    if not smiles:
        raise ValueError("missing canonical_smiles")
    novelty = _optional_float(row.get("fingerprint_novelty"))
    high_novelty = row.get("high_novelty") is True

    density = None
    if density_model != "none":
        density = estimate_vdw_density_from_smiles(
            smiles,
            seed=seed,
            novelty_score=novelty,
            high_novelty=high_novelty,
        )
    melting = None
    if melting_point_model == "sfs_vr_mp":
        try:
            from melting_point_sfs_vr import predict_sfs_vr_melting_point
        except ModuleNotFoundError as exc:  # pragma: no cover - legacy backend
            raise RuntimeError(
                "melting_point_model='sfs_vr_mp' requires the historical "
                "'melting_point_sfs_vr' module, which is not part of the "
                "released package. Use --melting-point-model phase_fusion "
                "(the production head) or 'sta'."
            ) from exc

        melting = predict_sfs_vr_melting_point(
            smiles,
            model_path=mp_model_path,
            data_path=mp_data_path,
            novelty_score=novelty,
            high_novelty=high_novelty,
        )
    # Confirmatory geometry diagnostics (the reproducibility term) are computed
    # only for the candidates that stay in the running (Tier 1 or Tier 2). The
    # three compared properties are already known here, so the decision is a
    # plain inline check; every other molecule reuses a neutral value.
    confirm_geometry = True
    if reproducibility_scope and reproducibility_scope != "all" and jp10_benchmark is not None:
        from tier_routing import classify_tier

        confirm_geometry = classify_tier(
            {
                "density": _optional_float(row.get("external_density_g_cm3")),
                "mass_heating_value": _optional_float(row.get("external_heating_value_mj_kg")),
                "melting_point": _optional_float(row.get("external_melting_point_k")),
            },
            jp10_benchmark,  # type: ignore[arg-type]
            hard_criteria_pass=False,
        ).tier in ("tier1", "tier2")
    set_reproducibility_enabled(confirm_geometry)
    geometry_payload: dict[str, object] | None = None
    geometry_override = (
        geometry_scores_override.get(smiles)
        if geometry_scores_override is not None
        else None
    )
    if geometry_override is not None:
        geometry, geometry_model_score = geometry_override
        if e3nn_geometry_checkpoint is not None and geometry_model_score is not None:
            geometry_payload = {"e3nn_geometry_model_score": geometry_model_score}
    else:
        if e3nn_geometry_checkpoint is not None:
            geometry_payload = predict_geometry_score(
                e3nn_geometry_checkpoint,
                smiles,
                seed=seed,
            )
        geometry = geometry_score_from_smiles(smiles, seed=seed)
    e3nn_geometry_score = (
        geometry_payload.get("e3nn_geometry_model_score")
        if geometry_payload is not None
        else geometry.e3nn_geometry_score
    )
    set_reproducibility_enabled(True)
    e3nn_property_prediction: float | str = ""
    e3nn_property_status = "not_requested"
    e3nn_property_failure_reason = ""
    if e3nn_property_checkpoint is not None:
        try:
            e3nn_property_geometry = geometry_from_smiles(smiles, seed=seed)
            e3nn_property_prediction = predict_property(
                e3nn_property_checkpoint,
                e3nn_property_geometry,
            )
            e3nn_property_status = "predicted"
        except Exception as exc:
            e3nn_property_status = "failed"
            e3nn_property_failure_reason = str(exc)
    sta_status = "not_requested"
    sta_failure_reason = ""
    sta_predictions: dict[str, object] = dict(sta_predictions_override or {})
    sta_model_paths = _selected_sta_model_paths(
        fallback_model_path=sta_property_model_path,
        density_model=density_model,
        heating_value_model=heating_value_model,
        melting_point_model=melting_point_model,
        boiling_point_model=boiling_point_model,
        flash_point_model=flash_point_model,
        formation_enthalpy_model=formation_enthalpy_model,
        molar_volume_model=molar_volume_model,
        isp_model=isp_model,
        sta_density_model_path=sta_density_model_path,
        sta_heating_value_model_path=sta_heating_value_model_path,
        sta_melting_point_model_path=sta_melting_point_model_path,
        sta_boiling_point_model_path=sta_boiling_point_model_path,
        sta_flash_point_model_path=sta_flash_point_model_path,
        sta_formation_enthalpy_model_path=sta_formation_enthalpy_model_path,
        sta_molar_volume_model_path=sta_molar_volume_model_path,
        sta_isp_model_path=sta_isp_model_path,
    )
    if sta_predictions_override is not None:
        sta_status = str(sta_predictions.get("sta_status") or "predicted")
    elif sta_model_paths:
        statuses = []
        failures = []
        unique_sta_model_paths = list(dict.fromkeys(str(path) for path in sta_model_paths))
        for model_path in unique_sta_model_paths:
            try:
                prediction = predict_sta_properties(
                    smiles,
                    model_path=model_path,
                )
                statuses.append(str(prediction.get("sta_status") or "predicted"))
                sta_predictions.update(prediction)
            except Exception as exc:
                statuses.append("failed")
                failures.append(f"{model_path}: {exc}")
        if failures:
            sta_status = "failed" if not sta_predictions else "partial_failed"
            sta_failure_reason = "; ".join(failures)
        else:
            sta_status = "predicted" if statuses else "not_requested"
        sta_predictions["sta_model"] = ";".join(unique_sta_model_paths)

    density_fields = (
        {
            "molecular_weight_g_mol": density.molecular_weight_g_mol,
            "external_density_g_cm3": density.density_g_cm3,
            "external_density_uncertainty": density.uncertainty_g_cm3,
            "external_density_uncertainty_source": density.uncertainty_source,
            "external_density_model": "thesis_vdw_density",
        }
        if density is not None
        else {
            "external_density_g_cm3": "",
            "external_density_uncertainty": "",
            "external_density_uncertainty_source": "",
            "external_density_model": "not_predicted",
        }
    )
    melting_fields = (
        {
            "external_melting_point_k": melting.melting_point_k,
            "external_melting_point_c": melting.melting_point_c,
            "external_melting_point_uncertainty": melting.uncertainty_k,
            "external_melting_point_interval_low_k": round(
                melting.melting_point_k - melting.uncertainty_k,
                6,
            ),
            "external_melting_point_interval_high_k": round(
                melting.melting_point_k + melting.uncertainty_k,
                6,
            ),
            "external_melting_point_interval_low_c": round(
                melting.melting_point_c - melting.uncertainty_k,
                6,
            ),
            "external_melting_point_interval_high_c": round(
                melting.melting_point_c + melting.uncertainty_k,
                6,
            ),
            "external_melting_point_uncertainty_source": melting.uncertainty_source,
            "external_melting_point_model": "sfs_vr_mp",
            "external_melting_point_usage": "reference_only",
        }
        if melting is not None
        else {
            "external_melting_point_k": "",
            "external_melting_point_c": "",
            "external_melting_point_uncertainty": "",
            "external_melting_point_interval_low_k": "",
            "external_melting_point_interval_high_k": "",
            "external_melting_point_interval_low_c": "",
            "external_melting_point_interval_high_c": "",
            "external_melting_point_uncertainty_source": "",
            "external_melting_point_model": "not_predicted",
            "external_melting_point_usage": "ignored",
        }
    )
    boiling_fields = {
        "external_boiling_point_k": "",
        "external_boiling_point_uncertainty": "",
        "external_boiling_point_uncertainty_source": "",
        "external_boiling_point_model": "not_predicted",
        "external_boiling_point_model_path": "",
    }

    row.update(
        {
            **density_fields,
            **melting_fields,
            **boiling_fields,
            "e3nn_geometry_score": e3nn_geometry_score,
            "e3nn_geometry_label_score": geometry.e3nn_geometry_score,
            "e3nn_geometry_model_checkpoint": (
                "" if e3nn_geometry_checkpoint is None else str(e3nn_geometry_checkpoint)
            ),
            "e3nn_property_name": e3nn_property_name,
            "e3nn_property_prediction": e3nn_property_prediction,
            "e3nn_property_model_checkpoint": (
                "" if e3nn_property_checkpoint is None else str(e3nn_property_checkpoint)
            ),
            "e3nn_property_prediction_status": e3nn_property_status,
            "e3nn_property_failure_reason": e3nn_property_failure_reason,
            "sta_property_model_checkpoint": (
                ";".join(str(path) for path in sta_model_paths)
            ),
            "sta_property_prediction_status": sta_status,
            "sta_property_failure_reason": sta_failure_reason,
            "symmetry_consistency": geometry.symmetry_consistency,
            "mmff_stability": geometry.mmff_stability,
            "compactness": geometry.compactness,
            "strain_reasonableness": geometry.strain_reasonableness,
            "conformer_reproducibility": geometry.conformer_reproducibility,
        }
    )
    row.update(sta_predictions)
    _apply_sta_optional_property_models(
        row,
        density_model=density_model,
        melting_point_model=melting_point_model,
        boiling_point_model=boiling_point_model,
        flash_point_model=flash_point_model,
        molar_volume_model=molar_volume_model,
        isp_model=isp_model,
    )
    _apply_derived_molar_volume(
        row,
        enabled=True,
    )
    heating_value = heating_values.get(geometry.canonical_smiles)
    e3nn_heating = _e3nn_heating_proxy_from_prediction(
        smiles,
        e3nn_property_prediction,
        e3nn_property_status,
        e3nn_property_name,
    )
    cbh_heating = None
    cbh_failure_reason = ""
    if use_cbh_bdc_heating and (
        heating_value_model == "yang_cbh_bdc"
        or formation_enthalpy_model == "yang_cbh_bdc"
    ):
        try:
            from cbh_bdc_model import predict_cbh_bdc_formation_enthalpy

            cbh_heating = predict_cbh_bdc_formation_enthalpy(
                smiles,
                model_path=cbh_bdc_model_path,
                feature_csv=cbh_bdc_feature_csv,
                yang_root=cbh_bdc_yang_root,
            )
        except Exception as exc:
            cbh_failure_reason = str(exc)

    if cbh_heating is not None:
        row["cbh_bdc_formation_enthalpy_kj_mol"] = (
            cbh_heating.formation_enthalpy_kj_mol
        )
        row["cbh_bdc_heating_value_mj_kg"] = cbh_heating.heating_value_mj_kg
        row["cbh_bdc_heating_value_uncertainty"] = (
            cbh_heating.heating_value_uncertainty_mj_kg
        )
        row["cbh_bdc_training_target"] = cbh_heating.predicted_training_target
        row["cbh_bdc_target_uncertainty"] = cbh_heating.target_uncertainty
        row["cbh_bdc_uncertainty_source"] = cbh_heating.uncertainty_source
        row["cbh_bdc_ignored_fragment_count"] = cbh_heating.ignored_fragment_count
        row["cbh_bdc_ignored_fragments_json"] = json.dumps(
            cbh_heating.ignored_fragments,
            ensure_ascii=False,
        )
        row["cbh_bdc_status"] = "predicted"
        row["cbh_bdc_failure_reason"] = ""
    else:
        row["cbh_bdc_formation_enthalpy_kj_mol"] = ""
        row["cbh_bdc_heating_value_mj_kg"] = ""
        row["cbh_bdc_heating_value_uncertainty"] = ""
        row["cbh_bdc_training_target"] = ""
        row["cbh_bdc_target_uncertainty"] = ""
        row["cbh_bdc_uncertainty_source"] = ""
        row["cbh_bdc_ignored_fragment_count"] = ""
        row["cbh_bdc_ignored_fragments_json"] = ""
        row["cbh_bdc_status"] = "not_requested" if not use_cbh_bdc_heating else "failed"
        row["cbh_bdc_failure_reason"] = cbh_failure_reason

    if e3nn_heating is not None:
        row["e3nn_heating_value_mj_kg"] = e3nn_heating.heating_value_mj_kg
        row["e3nn_formation_enthalpy_kj_mol_proxy"] = (
            e3nn_heating.formation_enthalpy_kj_mol
        )
        row["e3nn_combustion_enthalpy_kj_mol_proxy"] = (
            e3nn_heating.combustion_enthalpy_kj_mol
        )
        row["e3nn_heating_value_basis"] = "QM9_h298_proxy_LHV_gas_water"
    else:
        row["e3nn_heating_value_mj_kg"] = ""
        row["e3nn_formation_enthalpy_kj_mol_proxy"] = ""
        row["e3nn_combustion_enthalpy_kj_mol_proxy"] = ""
        row["e3nn_heating_value_basis"] = ""

    if heating_value is None:
        if heating_value_model == "none":
            row["external_heating_value_mj_kg"] = ""
            row["external_heating_value_uncertainty"] = ""
            row["external_heating_value_model"] = "not_predicted"
        elif heating_value_model == "sta":
            _apply_sta_formation_derived_heating_value(row, smiles)
        elif heating_value_model == "yang_cbh_bdc" and cbh_heating is not None:
            row["external_heating_value_mj_kg"] = cbh_heating.heating_value_mj_kg
            row["external_heating_value_uncertainty"] = (
                cbh_heating.heating_value_uncertainty_mj_kg
            )
            row["external_heating_value_model"] = "yang_cbh_bdc_delta_ml"
        else:
            row["external_heating_value_mj_kg"] = ""
            row["external_heating_value_uncertainty"] = ""
            row["external_heating_value_model"] = (
                "yang_cbh_bdc_failed" if use_cbh_bdc_heating else "heating_value_pending"
            )
    else:
        row["external_heating_value_mj_kg"] = heating_value
        row["external_heating_value_uncertainty"] = 0.0
        row["external_formation_enthalpy_kj_mol"] = ""
        row["external_formation_enthalpy_uncertainty_kj_mol"] = ""
        row["external_heating_value_model"] = "external_heating_values_csv"
    _apply_formation_enthalpy_model(
        row,
        formation_enthalpy_model=formation_enthalpy_model,
        cbh_heating=cbh_heating,
    )
    _apply_derived_volumetric_heating_value(row)
    _apply_phase_fusion_optional_property_models(
        row,
        smiles=smiles,
        melting_point_model=melting_point_model,
        boiling_point_model=boiling_point_model,
        flash_point_model=flash_point_model,
        phase_fusion_melting_point_model_path=phase_fusion_melting_point_model_path,
        phase_fusion_boiling_point_model_path=phase_fusion_boiling_point_model_path,
        phase_fusion_flash_point_model_path=phase_fusion_flash_point_model_path,
    )


def _selected_sta_model_paths(
    fallback_model_path: str | Path | None,
    density_model: str,
    heating_value_model: str,
    melting_point_model: str,
    boiling_point_model: str,
    flash_point_model: str,
    formation_enthalpy_model: str,
    molar_volume_model: str,
    isp_model: str,
    sta_density_model_path: str | Path | None,
    sta_heating_value_model_path: str | Path | None,
    sta_melting_point_model_path: str | Path | None,
    sta_boiling_point_model_path: str | Path | None,
    sta_flash_point_model_path: str | Path | None,
    sta_formation_enthalpy_model_path: str | Path | None,
    sta_molar_volume_model_path: str | Path | None,
    sta_isp_model_path: str | Path | None,
) -> list[str | Path]:
    selected: list[str | Path] = []

    def choose_path(target: str, explicit: str | Path | None) -> str | Path:
        return explicit or fallback_model_path or DEFAULT_SPECIALIST_MODEL_PATHS.get(
            target,
            DEFAULT_STA_MODEL_PATH,
        )

    if density_model == "sta":
        selected.append(choose_path("density_g_cm3", sta_density_model_path))
    if heating_value_model == "sta":
        selected.append(
            choose_path(
                "formation_enthalpy_kj_mol",
                sta_heating_value_model_path or sta_formation_enthalpy_model_path,
            )
        )
    if melting_point_model in ("sta", "phase_fusion"):
        selected.append(choose_path("melting_point_k", sta_melting_point_model_path))
    if boiling_point_model in ("sta", "phase_fusion"):
        selected.append(choose_path("boiling_point_k", sta_boiling_point_model_path))
    if flash_point_model in ("sta", "phase_fusion"):
        selected.append(choose_path("flash_point_k", sta_flash_point_model_path))
    if formation_enthalpy_model == "sta":
        selected.append(
            choose_path(
                "formation_enthalpy_kj_mol",
                sta_formation_enthalpy_model_path,
            )
        )
    if isp_model == "sta":
        selected.append(choose_path("isp", sta_isp_model_path))
    return selected


def _batch_sta_predictions_for_smiles(
    smiles_values: list[str],
    model_paths: list[str | Path],
    batch_size: int = 256,
) -> dict[str, dict[str, object]]:
    unique_smiles = list(dict.fromkeys(smiles_values))
    merged = {smiles: {} for smiles in unique_smiles}
    for model_path in dict.fromkeys(str(path) for path in model_paths):
        predictions = predict_sta_properties_batch(
            unique_smiles,
            model_path=model_path,
            batch_size=batch_size,
        )
        for smiles, prediction in zip(unique_smiles, predictions):
            merged[smiles].update(prediction)
    return merged


def _apply_sta_optional_property_models(
    row: dict[str, object],
    density_model: str,
    melting_point_model: str,
    boiling_point_model: str,
    flash_point_model: str,
    molar_volume_model: str,
    isp_model: str,
) -> None:
    if density_model == "sta":
        _copy_sta_value(
            row,
            sta_value_column="sta_density_g_cm3",
            sta_uncertainty_column="sta_density_g_cm3_uncertainty",
            external_value_column="external_density_g_cm3",
            external_uncertainty_column="external_density_uncertainty",
            external_model_column="external_density_model",
            model_name="sta_gdb_density_paper",
        )
        row["external_density_uncertainty_source"] = "sta_validation_mae"
    if melting_point_model == "sta":
        _copy_sta_value(
            row,
            sta_value_column="sta_melting_point_k",
            sta_uncertainty_column="sta_melting_point_k_uncertainty",
            external_value_column="external_melting_point_k",
            external_uncertainty_column="external_melting_point_uncertainty",
            external_model_column="external_melting_point_model",
            model_name="sta_gdb_melting_point_paper",
        )
        melting = _optional_float(row.get("external_melting_point_k"))
        uncertainty = _optional_float(row.get("external_melting_point_uncertainty")) or 0.0
        if melting is not None:
            row["external_melting_point_c"] = round(melting - 273.15, 6)
            row["external_melting_point_interval_low_k"] = round(melting - uncertainty, 6)
            row["external_melting_point_interval_high_k"] = round(melting + uncertainty, 6)
            row["external_melting_point_interval_low_c"] = round(
                melting - 273.15 - uncertainty,
                6,
            )
            row["external_melting_point_interval_high_c"] = round(
                melting - 273.15 + uncertainty,
                6,
            )
            row["external_melting_point_uncertainty_source"] = "sta_validation_mae"
    if boiling_point_model == "sta":
        _copy_sta_value(
            row,
            sta_value_column="sta_boiling_point_k",
            sta_uncertainty_column="sta_boiling_point_k_uncertainty",
            external_value_column="external_boiling_point_k",
            external_uncertainty_column="external_boiling_point_uncertainty",
            external_model_column="external_boiling_point_model",
            model_name="sta_gdb_boiling_point_paper",
        )
        row["external_boiling_point_uncertainty_source"] = "sta_validation_mae"
    if flash_point_model == "sta":
        _copy_sta_value(
            row,
            sta_value_column="sta_flash_point_k",
            sta_uncertainty_column="sta_flash_point_k_uncertainty",
            external_value_column="external_flash_point_k",
            external_uncertainty_column="external_flash_point_uncertainty",
            external_model_column="external_flash_point_model",
            model_name="sta_gdb_flash_point_paper",
        )
    if isp_model == "sta":
        _copy_sta_value(
            row,
            sta_value_column="sta_isp",
            sta_uncertainty_column="sta_isp_uncertainty",
            external_value_column="external_isp",
            external_uncertainty_column="external_isp_uncertainty",
            external_model_column="external_isp_model",
            model_name="sta_gdb_isp_paper",
        )


def _apply_derived_molar_volume(
    row: dict[str, object],
    *,
    enabled: bool,
) -> None:
    formula = str(row.get("formula") or "").strip()
    molecular_weight = None
    if formula:
        molecular_weight = molar_mass_from_formula(formula)
        row["molecular_weight"] = molecular_weight
        row["molecular_weight_g_mol"] = molecular_weight
        row["molecular_weight_source"] = "calculated_from_molecular_formula"
    else:
        molecular_weight = _optional_float(row.get("molecular_weight_g_mol"))
        if molecular_weight is None:
            molecular_weight = _optional_float(row.get("molecular_weight"))
        row["molecular_weight_source"] = (
            "existing_value_without_formula"
            if molecular_weight is not None
            else "not_available_without_formula"
        )

    if not enabled:
        row["molar_volume_cm3_mol"] = ""
        row["external_molar_volume_cm3_mol"] = ""
        row["external_molar_volume_uncertainty"] = ""
        row["external_molar_volume_uncertainty_source"] = ""
        row["external_molar_volume_model"] = "not_requested"
        return

    density = _optional_float(row.get("external_density_g_cm3"))
    if density is None:
        density = _optional_float(row.get("density_g_cm3"))
    density_uncertainty = _optional_float(row.get("external_density_uncertainty"))
    if molecular_weight is None or density is None:
        row["molar_volume_cm3_mol"] = ""
        row["external_molar_volume_cm3_mol"] = ""
        row["external_molar_volume_uncertainty"] = ""
        row["external_molar_volume_uncertainty_source"] = ""
        row["external_molar_volume_model"] = "not_available_without_mass_and_density"
        return

    estimate = molar_volume_from_density(
        molecular_weight,
        density,
        density_uncertainty,
    )
    row["molecular_weight_g_mol"] = estimate.molecular_weight_g_mol
    row["molar_volume_cm3_mol"] = estimate.molar_volume_cm3_mol
    row["external_molar_volume_cm3_mol"] = estimate.molar_volume_cm3_mol
    row["external_molar_volume_uncertainty"] = (
        "" if estimate.uncertainty_cm3_mol is None else estimate.uncertainty_cm3_mol
    )
    row["external_molar_volume_uncertainty_source"] = (
        "propagated_from_density_uncertainty"
        if estimate.uncertainty_cm3_mol is not None
        else ""
    )
    row["external_molar_volume_model"] = estimate.source


def _apply_derived_volumetric_heating_value(row: dict[str, object]) -> None:
    density = _first_float(
        row,
        "external_density_g_cm3",
        "density_g_cm3",
    )
    heating_value = _first_float(
        row,
        "external_heating_value_mj_kg",
        "heating_value_mj_kg",
    )
    if density is None or heating_value is None:
        row["volumetric_heating_value_mj_l"] = ""
        row["volumetric_heating_value_uncertainty"] = ""
        row["volumetric_heating_value_source"] = (
            "not_available_without_density_and_mass_heating_value"
        )
        return
    estimate = volumetric_heating_value_from_density(
        density_g_cm3=density,
        heating_value_mj_kg=heating_value,
        density_uncertainty_g_cm3=_first_float(
            row,
            "external_density_uncertainty",
            "density_uncertainty",
        ),
        heating_value_uncertainty_mj_kg=_first_float(
            row,
            "external_heating_value_uncertainty",
            "heating_value_uncertainty",
        ),
    )
    row["volumetric_heating_value_mj_l"] = (
        estimate.volumetric_heating_value_mj_l
    )
    row["volumetric_heating_value_uncertainty"] = (
        ""
        if estimate.uncertainty_mj_l is None
        else estimate.uncertainty_mj_l
    )
    row["volumetric_heating_value_source"] = estimate.source


def _apply_phase_fusion_optional_property_models(
    row: dict[str, object],
    smiles: str,
    melting_point_model: str,
    boiling_point_model: str,
    flash_point_model: str,
    phase_fusion_melting_point_model_path: str | Path | None,
    phase_fusion_boiling_point_model_path: str | Path | None,
    phase_fusion_flash_point_model_path: str | Path | None,
) -> None:
    requests = (
        (
            melting_point_model,
            "melting_point_k",
            phase_fusion_melting_point_model_path,
            "external_melting_point_k",
            "external_melting_point_uncertainty",
            "external_melting_point_model",
            "external_melting_point_uncertainty_source",
            "external_melting_point_model_path",
        ),
        (
            boiling_point_model,
            "boiling_point_k",
            phase_fusion_boiling_point_model_path,
            "external_boiling_point_k",
            "external_boiling_point_uncertainty",
            "external_boiling_point_model",
            "external_boiling_point_uncertainty_source",
            "external_boiling_point_model_path",
        ),
        (
            flash_point_model,
            "flash_point_k",
            phase_fusion_flash_point_model_path,
            "external_flash_point_k",
            "external_flash_point_uncertainty",
            "external_flash_point_model",
            "external_flash_point_uncertainty_source",
            "external_flash_point_model_path",
        ),
    )
    for (
        selected_model,
        target,
        model_path,
        value_column,
        uncertainty_column,
        model_column,
        uncertainty_source_column,
        model_path_column,
    ) in requests:
        if selected_model != "phase_fusion":
            continue
        try:
            prediction = predict_phase_fusion_property(
                smiles,
                target=target,
                model_path=model_path or DEFAULT_PHASE_FUSION_MODEL_PATHS[target],
                aux_values=_phase_fusion_aux_values(row),
            )
        except ReferenceOnlyFeatureModelError:
            sta_value = _optional_float(row.get(f"sta_{target}"))
            if sta_value is None:
                raise
            sta_uncertainty = _optional_float(row.get(f"sta_{target}_uncertainty"))
            prediction = {
                "value": sta_value,
                "uncertainty": 20.0 if sta_uncertainty is None else sta_uncertainty,
                "uncertainty_source": "sta_fallback_without_reference_only_features",
                "model": "sta_fallback",
                "model_path": str(row.get("sta_model") or ""),
                "feature_set": "sta_fallback_until_molar_volume_fusion_retraining",
            }
        value = float(prediction["value"])
        uncertainty = float(prediction["uncertainty"])
        row[value_column] = value
        row[uncertainty_column] = uncertainty
        row[model_column] = prediction["model"]
        row[uncertainty_source_column] = prediction["uncertainty_source"]
        row[model_path_column] = prediction["model_path"]
        row[f"{target}_phase_fusion_feature_set"] = prediction["feature_set"]
        if target == "melting_point_k":
            row["external_melting_point_c"] = round(value - 273.15, 6)
            row["external_melting_point_interval_low_k"] = round(value - uncertainty, 6)
            row["external_melting_point_interval_high_k"] = round(value + uncertainty, 6)
            row["external_melting_point_interval_low_c"] = round(
                value - 273.15 - uncertainty,
                6,
            )
            row["external_melting_point_interval_high_c"] = round(
                value - 273.15 + uncertainty,
                6,
            )
            row["external_melting_point_usage"] = "reference_only"


def _phase_fusion_aux_values(row: dict[str, object]) -> dict[str, object]:
    return {
        "density_g_cm3": _first_present(
            row,
            "external_density_g_cm3",
            "density_g_cm3",
            "sta_density_g_cm3",
        ),
        "molar_volume_cm3_mol": _first_present(
            row,
            "molar_volume_cm3_mol",
            "external_molar_volume_cm3_mol",
        ),
        "formation_enthalpy_kj_mol": _first_present(
            row,
            "external_formation_enthalpy_kj_mol",
            "cbh_bdc_formation_enthalpy_kj_mol",
            "sta_formation_enthalpy_kj_mol",
        ),
        "isp": _first_present(row, "external_isp", "sta_isp"),
        "melting_point_k": _first_present(
            row,
            "external_melting_point_k",
            "sta_melting_point_k",
        ),
        "boiling_point_k": _first_present(
            row,
            "external_boiling_point_k",
            "sta_boiling_point_k",
        ),
        "flash_point_k": _first_present(
            row,
            "external_flash_point_k",
            "sta_flash_point_k",
        ),
    }


def _first_present(row: dict[str, object], *columns: str) -> object:
    for column in columns:
        value = row.get(column)
        if _optional_float(value) is not None:
            return value
    return ""


def _first_float(row: dict[str, object], *columns: str) -> float | None:
    for column in columns:
        value = _optional_float(row.get(column))
        if value is not None:
            return value
    return None


def _apply_formation_enthalpy_model(
    row: dict[str, object],
    formation_enthalpy_model: str,
    cbh_heating,
) -> None:
    if formation_enthalpy_model == "sta":
        _copy_sta_value(
            row,
            sta_value_column="sta_formation_enthalpy_kj_mol",
            sta_uncertainty_column="sta_formation_enthalpy_kj_mol_uncertainty",
            external_value_column="external_formation_enthalpy_kj_mol",
            external_uncertainty_column="external_formation_enthalpy_uncertainty_kj_mol",
            external_model_column="external_formation_enthalpy_model",
            model_name="sta_gdb_formation_enthalpy_paper",
        )
        return
    if formation_enthalpy_model == "yang_cbh_bdc" and cbh_heating is not None:
        row["external_formation_enthalpy_kj_mol"] = (
            cbh_heating.formation_enthalpy_kj_mol
        )
        row["external_formation_enthalpy_uncertainty_kj_mol"] = (
            cbh_heating.formation_enthalpy_uncertainty_kj_mol
        )
        row["external_formation_enthalpy_model"] = "yang_cbh_bdc_delta_ml"
        return
    row.setdefault("external_formation_enthalpy_kj_mol", "")
    row.setdefault("external_formation_enthalpy_uncertainty_kj_mol", "")
    row.setdefault("external_formation_enthalpy_model", "none")


def _apply_sta_formation_derived_heating_value(
    row: dict[str, object], smiles: str
) -> None:
    formation = _optional_float(row.get("sta_formation_enthalpy_kj_mol"))
    formation_uncertainty = _optional_float(
        row.get("sta_formation_enthalpy_kj_mol_uncertainty")
    )
    if formation is None:
        row["external_heating_value_mj_kg"] = ""
        row["external_heating_value_uncertainty"] = ""
        row["external_heating_value_model"] = "sta_formation_enthalpy_derived_failed"
        return

    try:
        estimate = mass_heating_value_from_formation_enthalpy(
            smiles,
            formation,
            water_phase="gas",
        )
    except Exception:
        row["external_heating_value_mj_kg"] = ""
        row["external_heating_value_uncertainty"] = ""
        row["external_heating_value_model"] = "sta_formation_enthalpy_derived_failed"
        return

    uncertainty = (
        formation_uncertainty / estimate.molecular_weight_g_mol
        if formation_uncertainty is not None
        else 0.0
    )
    row["sta_combustion_enthalpy_kj_mol"] = estimate.combustion_enthalpy_kj_mol
    row["sta_mass_calorific_value_mj_kg"] = estimate.heating_value_mj_kg
    row["sta_mass_calorific_value_mj_kg_uncertainty"] = round(uncertainty, 6)
    row["external_heating_value_mj_kg"] = estimate.heating_value_mj_kg
    row["external_heating_value_uncertainty"] = round(uncertainty, 6)
    row["external_heating_value_model"] = "sta_formation_enthalpy_hess_law"


def _copy_sta_value(
    row: dict[str, object],
    sta_value_column: str,
    sta_uncertainty_column: str,
    external_value_column: str,
    external_uncertainty_column: str,
    external_model_column: str,
    model_name: str,
) -> None:
    value = _optional_float(row.get(sta_value_column))
    if value is None:
        row[external_value_column] = ""
        row[external_uncertainty_column] = ""
        row[external_model_column] = f"{model_name}_failed"
        return
    row[external_value_column] = value
    row[external_uncertainty_column] = _optional_float(row.get(sta_uncertainty_column)) or 0.0
    row[external_model_column] = model_name


def _e3nn_heating_proxy_from_prediction(
    smiles: str,
    prediction: float | str,
    status: str,
    property_name: str,
):
    if status != "predicted" or prediction in ("", None):
        return None
    normalized_name = property_name.lower()
    if "h298" not in normalized_name or "hartree" not in normalized_name:
        return None
    return mass_heating_value_from_qm9_h298_proxy(smiles, float(prediction))


def _route_enriched_property_row(
    row: dict[str, object],
    uncertainty_scale: float,
    thresholds: FuelPropertyThresholds,
    ignore_density: bool = False,
    ignore_heating_value: bool = False,
    use_melting_point_screen: bool = False,
    use_boiling_point_screen: bool = False,
    use_flash_point_screen: bool = False,
    use_volumetric_heating_value_screen: bool = False,
) -> None:
    if (
        not ignore_heating_value
        and row.get("external_heating_value_mj_kg") in ("", None)
    ):
        # The heating value is a required hard property, so a row without it is a
        # data gap: it is reported in the missing-property table and carries no
        # channel, exactly like a row missing any other selected property.
        row["property_stage_status"] = "missing_properties"
        row["hard_threshold_verdict"] = "missing_properties"
        row["final_route_reason"] = "missing_heating_value"
        return

    decisions: list[PropertyDecision] = []
    density_value = _optional_float(row.get("external_density_g_cm3"))
    density_uncertainty = _optional_float(row.get("external_density_uncertainty"))
    heating_value = _optional_float(row.get("external_heating_value_mj_kg"))
    heating_uncertainty = _optional_float(row.get("external_heating_value_uncertainty"))
    melting_value = _optional_float(row.get("external_melting_point_k"))
    melting_uncertainty = _optional_float(row.get("external_melting_point_uncertainty"))
    boiling_value = _optional_float(row.get("external_boiling_point_k"))
    boiling_uncertainty = _optional_float(row.get("external_boiling_point_uncertainty"))
    flash_value = _optional_float(row.get("external_flash_point_k"))
    flash_uncertainty = _optional_float(row.get("external_flash_point_uncertainty"))
    volumetric_value = _optional_float(row.get("volumetric_heating_value_mj_l"))
    volumetric_uncertainty = _optional_float(
        row.get("volumetric_heating_value_uncertainty")
    )

    if use_volumetric_heating_value_screen:
        density_decision = "replaced_by_volumetric_heating_value"
    elif ignore_density or density_value is None:
        density_decision = "ignored"
    else:
        density = classify_property_threshold_with_uncertainty(
            "density_g_cm3",
            density_value,
            density_uncertainty or 0.0,
            thresholds.density_min_g_cm3,
            ThresholdDirection.MINIMUM,
            uncertainty_scale,
        )
        density_decision = density.decision.value
        decisions.append(density.decision)

    if use_volumetric_heating_value_screen:
        heating_decision = "replaced_by_volumetric_heating_value"
    elif ignore_heating_value or heating_value is None:
        heating_decision = "ignored"
    else:
        heating = classify_property_threshold_with_uncertainty(
            "heating_value_mj_kg",
            heating_value,
            heating_uncertainty or 0.0,
            thresholds.heating_value_min_mj_kg,
            ThresholdDirection.MINIMUM,
            uncertainty_scale,
        )
        heating_decision = heating.decision.value
        decisions.append(heating.decision)

    if not use_melting_point_screen or melting_value is None:
        melting_decision = "reference_only" if melting_value is not None else "ignored"
    else:
        melting = classify_property_threshold_with_uncertainty(
            "melting_point_k",
            melting_value,
            melting_uncertainty or 0.0,
            thresholds.melting_point_max_k,
            ThresholdDirection.MAXIMUM,
            uncertainty_scale,
        )
        melting_decision = melting.decision.value
        decisions.append(melting.decision)

    if not use_boiling_point_screen or boiling_value is None:
        boiling_decision = "reference_only" if boiling_value is not None else "ignored"
    else:
        boiling = classify_property_threshold_with_uncertainty(
            "boiling_point_k",
            boiling_value,
            boiling_uncertainty or 0.0,
            thresholds.boiling_point_min_k,
            ThresholdDirection.MINIMUM,
            uncertainty_scale,
        )
        boiling_decision = boiling.decision.value
        decisions.append(boiling.decision)

    if not use_flash_point_screen or flash_value is None:
        flash_decision = "reference_only" if flash_value is not None else "ignored"
    else:
        flash = classify_property_threshold_with_uncertainty(
            "flash_point_k",
            flash_value,
            flash_uncertainty or 0.0,
            thresholds.flash_point_min_k,
            ThresholdDirection.MINIMUM,
            uncertainty_scale,
        )
        flash_decision = flash.decision.value
        decisions.append(flash.decision)

    if use_volumetric_heating_value_screen and volumetric_value is not None:
        volumetric = classify_property_threshold_with_uncertainty(
            "volumetric_heating_value_mj_l",
            volumetric_value,
            volumetric_uncertainty or 0.0,
            thresholds.volumetric_heating_value_min_mj_l,
            ThresholdDirection.MINIMUM,
            uncertainty_scale,
        )
        volumetric_decision = volumetric.decision.value
        decisions.append(volumetric.decision)
    elif use_volumetric_heating_value_screen:
        volumetric_decision = "missing"
    else:
        volumetric_decision = (
            "reference_only" if volumetric_value is not None else "ignored"
        )

    if PropertyDecision.REJECT in decisions:
        overall = PropertyDecision.REJECT
        route = ThresholdVerdict.REJECT
    elif PropertyDecision.THRESHOLD_REVIEW in decisions:
        overall = PropertyDecision.THRESHOLD_REVIEW
        route = ThresholdVerdict.REVIEW
    else:
        overall = PropertyDecision.PASS
        route = ThresholdVerdict.PASS

    row.update(
        {
            "property_decision": overall.value,
            "density_decision": density_decision,
            "heating_value_decision": heating_decision,
            "melting_point_decision": melting_decision,
            "boiling_point_decision": boiling_decision,
            "flash_point_decision": flash_decision,
            "volumetric_heating_value_decision": volumetric_decision,
            "density_g_cm3": "" if density_value is None else density_value,
            "heating_value_mj_kg": "" if heating_value is None else heating_value,
            "melting_point_k": "" if melting_value is None else melting_value,
            "boiling_point_k": "" if boiling_value is None else boiling_value,
            "flash_point_k": "" if flash_value is None else flash_value,
            "density_uncertainty": "" if density_uncertainty is None else density_uncertainty,
            "heating_value_uncertainty": (
                "" if heating_uncertainty is None else heating_uncertainty
            ),
            "melting_point_uncertainty": (
                "" if melting_uncertainty is None else melting_uncertainty
            ),
            "boiling_point_uncertainty": (
                "" if boiling_uncertainty is None else boiling_uncertainty
            ),
            "flash_point_uncertainty": (
                "" if flash_uncertainty is None else flash_uncertainty
            ),
            "relative_error_margin": 0.0,
            "uncertainty_scale": uncertainty_scale,
            "property_stage_status": "classified",
            "hard_threshold_verdict": route.value,
            "final_route_reason": "enabled_property_thresholds_only",
        }
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = fieldnames or _fieldnames(rows)
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_heating_value_map(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    values: dict[str, float] = {}
    for record in records:
        smiles = str(record.get("canonical_smiles") or "").strip()
        value = record.get("heating_value_mj_kg") or record.get(
            "external_heating_value_mj_kg"
        )
        if not smiles or value in (None, ""):
            continue
        values[smiles] = float(value)
    return values


def public_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Rows as they go to disk: tier split plus a neutral screening status.

    The internal ``hard_threshold_verdict`` (pass / review / reject and the
    data-gap markers) is an implementation detail of the screening step and is
    replaced by the neutral ``screening_status`` here, so no written table
    carries the internal column.
    """
    public: list[dict[str, object]] = []
    for row in rows:
        record = dict(row)
        record.pop("hard_threshold_verdict", None)
        record["screening_status"] = screening_status(row.get("hard_threshold_verdict"))
        public.append(record)
    return public


def _write_group_summary(
    root: str | Path,
    rows: list[dict[str, object]],
    total_carbons: int,
) -> Path:
    """Write one ``group_summary.csv`` for the carbon: counts per mirror split.

    The full per-partition dumps (a folder with ``metadata.json`` and seven
    CSVs for every ``n1/n2/n3`` combination) were removed on 2026-09-22: the
    per-carbon tables already carry those columns, so only the counts - the
    part that is not readable from a single flat table scale - are kept. The
    counts follow the channel split (Tier 1-4, then Potential 1-3).
    """
    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for row in rows:
        try:
            key = (int(row["n1"]), int(row["n2"]), int(row["n3"]))
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (n1, n2, n3), group_rows in sorted(grouped.items()):
        entry: dict[str, object] = {
            "total_carbons": total_carbons,
            "n1": n1,
            "n2": n2,
            "n3": n3,
            "total": len(group_rows),
        }
        for channel in CHANNEL_ORDER:
            entry[channel] = sum(
                1 for row in group_rows if row.get("tier") == channel
            )
        for status in GROUP_SUMMARY_STATUSES:
            entry[status] = sum(
                1
                for row in group_rows
                if screening_status(row.get("hard_threshold_verdict")) == status
            )
        summary_rows.append(entry)

    target = Path(root) / "group_summary.csv"
    fieldnames = [
        "total_carbons",
        "n1",
        "n2",
        "n3",
        "total",
        *CHANNEL_ORDER,
        *GROUP_SUMMARY_STATUSES,
    ]
    _write_csv(target, summary_rows, fieldnames=fieldnames)
    return target


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _candidate_pipeline_row(
    index: int,
    candidate,
    config: ScreeningConfig = ScreeningConfig(),
) -> dict[str, object]:
    if candidate.passes_screen is not True:
        route = "reject_pre_property"
        reason = candidate.failure_reasons or "failed_pre_property_screen"
    elif candidate.high_novelty is True:
        route = "exact_new_fingerprint_high"
        reason = (
            f"exact-new structure; fingerprint_novelty={candidate.fingerprint_novelty} "
            f">= {config.high_novelty_threshold}"
        )
    else:
        # the route value names what actually happened: the molecule passed the
        # pre-screen and sits below the high-novelty threshold (it still goes to
        # property prediction, with a weak-feedback label)
        route = "exact_new_fingerprint_low"
        reason = "exact-new structure; fingerprint novelty below high-novelty threshold"

    known_fuel = match_known_fuel(str(candidate.canonical_smiles or ""))
    row = {
        "candidate_id": f"candidate_{index:06d}",
        "pipeline_route": route,
        "route_reason": reason,
        "generation_reward_score": _generation_reward_score(candidate, config),
        **candidate.__dict__,
    }
    row["known_fuel"] = bool(known_fuel)
    row["known_fuel_names"] = (
        ";".join(known_fuel["known_fuel_names"]) if known_fuel else ""
    )
    return row


def _apply_temporary_smiles_database(
    rows: list[dict[str, object]],
    external_smiles: set[str] | None = None,
    output_path: str | Path | None = None,
) -> tuple[int, int, int]:
    """Return ``(accepted, duplicate_rejections, temporary_library_size)``.

    ``accepted`` counts the candidates of this group that were newly stored in the
    temporary library; ``temporary_library_size`` is the size of the library
    itself, which may be pre-seeded from an earlier group. Reporting the library
    size as "unique novel candidates" double counts those carried-over entries
    when groups are processed sequentially, so the two numbers stay separate.
    """
    temporary_smiles: set[str] = set(external_smiles or ())
    duplicate_rejections = 0
    accepted = 0

    for row in rows:
        if row.get("passes_screen") is not True:
            row["temporary_smiles_status"] = "not_usable"
            continue

        smiles = str(row.get("canonical_smiles") or "").strip()
        if not smiles:
            row["temporary_smiles_status"] = "missing_smiles"
            continue

        if smiles in temporary_smiles:
            duplicate_rejections += 1
            row["pipeline_route"] = "reject_pre_property"
            row["route_reason"] = "duplicate_temporary_smiles"
            row["passes_screen"] = False
            row["temporary_smiles_status"] = "duplicate"
            row["failure_reasons"] = _append_failure_reason(
                row.get("failure_reasons"),
                "duplicate_temporary_smiles",
            )
            continue

        temporary_smiles.add(smiles)
        row["temporary_smiles_status"] = "accepted"
        accepted += 1

    if output_path is not None and temporary_smiles:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            "\n".join(sorted(temporary_smiles)) + "\n",
            encoding="utf-8",
        )

    return accepted, duplicate_rejections, len(temporary_smiles)


def _append_failure_reason(existing: object, reason: str) -> str:
    existing_text = str(existing or "").strip()
    if not existing_text:
        return reason
    if reason in {part.strip() for part in existing_text.split(";")}:
        return existing_text
    return f"{existing_text};{reason}"


def _final_sort_key(row: dict[str, object]) -> tuple[float, float, float, float, int]:
    return (
        float(row.get("composite_ranking_score") or row.get("generation_reward_score") or 0.0),
        float(row.get("property_quality_score") or 0.0),
        float(row.get("screen_score") or 0.0),
        float(row.get("fingerprint_novelty") or 0.0),
        int(row.get("preferred_ring_score") or 0),
    )


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
