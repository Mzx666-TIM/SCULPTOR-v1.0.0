from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

from fuel_design.paths import SCULPTOR_ROOT

from bond_model.dataset import export_semi_base_training_csv, export_training_csv
from ani1x import process_ani1x_hydrocarbons
from bond_model.predict import (
    load_bond_model,
    predict_bonds,
    predict_bonds_with_model,
    prediction_to_json,
)
from bond_model.train import result_to_json, train_bond_model
from bond_model.semi_base import predict_semi_base_with_model
from chemistry import descriptors_from_smiles
from gdb import build_known_smiles_library, process_gdb_archive
from qm9 import (
    DEFAULT_QM9_E3NN_PROPERTY_PATH,
    DEFAULT_QM9_HYDROCARBON_PATH,
    prepare_qm9_e3nn_property_csv,
)
from evaluation import evaluate_generation, evaluation_to_json
from generator import (
    generate_base_for_counts,
    iter_count_partitions,
    iter_generated_bases_for_total,
    random_base_graph,
)
from graph import HalfGraph
from schedule import (
    adjust_schedule_with_feedback,
    generation_schedule,
    schedule_to_json,
)
from screening import ScreeningConfig, screen_generation
from symmetry import mirror_complete
from threshold_calibration import (
    calibrate_novelty_threshold_from_csv,
    threshold_calibration_to_json,
)
from property_model.data import geometry_from_smiles
from property_model.e3nn_regressor import (
    predict_property,
    train_property_model,
)
from property_model.e3nn_geometry_scorer import (
    geometry_score_from_smiles,
    predict_geometry_score,
    train_geometry_scorer,
    write_geometry_scores_csv,
)
from property_screening import FuelPropertyThresholds
from property_consensus import (
    CompositeScoreWeights,
    PropertyScoreWeights,
    PropertyTargetValues,
    rescore_candidate_csv,
)
from phase_fusion import (
    DEFAULT_PHASE_FUSION_MODEL_PATHS,
    DEFAULT_PHASE_TARGETS,
    train_phase_fusion_models,
)
from candidate_export import export_candidate_inputs_from_csv
from campaign_layout import ALLOWED_PERTURBATION_OPERATIONS
from finalization import (
    POTENTIAL_SEED_TOP_FRACTION,
    build_t123_perturbation_seeds,
    build_potential_perturbation_seeds,
    organize_run_outputs,
    retier_run_outputs,
)
from pipeline import run_generation_pipeline
from perturbation import (
    ITERATION_SEED_POOLS,
    compare_iteration_models_from_run,
    prepare_perturbation_campaign,
    train_iteration_model_from_run,
    run_iterative_perturbation,
    run_perturbation_generation,
    run_perturbation_pipeline,
)
from thesis_property_tools import (
    estimate_vdw_density_from_smiles,
    mass_heating_value_from_formation_enthalpy,
)
from sta_model import (
    DEFAULT_STA_GDB13_DATA_PATH,
    DEFAULT_STA_DATA_PATH,
    DEFAULT_STA_MODEL_PATH,
    DEFAULT_SPECIALIST_MODEL_PATHS,
    DEFAULT_TARGETS as DEFAULT_STA_TARGETS,
    predict_sta_property_csv,
    prepare_sta_gdb_dataset,
    train_sta_optimized_property_models,
    train_sta_specialist_property_models,
    train_sta_property_model,
)
from sta_feedback import (
    export_sta_weak_feedback_csv,
    sta_feedback_summary_to_dict,
)
from visualization import draw_symmetric_grid_from_csv
from fuel_design.paths import CBH_BDC_DIR, EVAL_DATA_DIR, RESULT_ROOT, SFS_VR_DIR

# Default paths of non-STA methods archived to addition (kept only as
# historical defaults; modules are lazily imported in command handlers and
# only fail when these methods are explicitly used).
DEFAULT_CBH_BDC_FEATURE_PATH = EVAL_DATA_DIR / "qm9_cbh_bdc_features.csv"
DEFAULT_CBH_BDC_FORMATION_FEATURE_PATH = (
    EVAL_DATA_DIR / "qm9_cbh_bdc_formation_features.csv"
)
DEFAULT_CBH_BDC_MODEL_PATH = CBH_BDC_DIR / "cbh_bdc_delta_ml_formation.joblib"
DEFAULT_CBH_BDC_PREDICTION_PATH = (
    RESULT_ROOT / "cbh_bdc_delta_ml_formation_predictions.csv"
)
DEFAULT_CBH_BDC_METRICS_PATH = (
    RESULT_ROOT / "cbh_bdc_delta_ml_formation_metrics.json"
)
DEFAULT_SFS_VR_MP_DATA_PATH = (
    SFS_VR_DIR / "SFS-VR-main" / "MP" / "MP-454-select-FEATURE - 15.xlsx"
)
DEFAULT_SFS_VR_MP_MODEL_PATH = SFS_VR_DIR / "sfs_vr_melting_point.joblib"



def _perturbation_pipeline_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Arguments shared by the single and the iterative perturbation run."""

    return dict(
            known_smiles_path=args.known_smiles,
            temporary_smiles_path=args.temporary_smiles,
            perturbation_temporary_smiles_path=args.perturbation_temporary_smiles,
            merge_existing_products=not args.replace_existing_products,
            known_fuel_check=args.known_fuel_check,
            seed=args.seed,
            operations=tuple(args.operation)
            if args.operation
            else tuple(sorted(ALLOWED_PERTURBATION_OPERATIONS)),
            max_seeds=args.max_seeds,
            novelty_threshold=args.novelty_threshold,
            uncertainty_scale=args.uncertainty_scale,
            ignore_density=args.ignore_density,
            ignore_heating_value=not args.use_heating_value,
            heating_values_csv=args.heating_values_csv,
            mp_model_path=args.mp_model,
            mp_data_path=args.mp_data,
            e3nn_geometry_checkpoint=args.e3nn_geometry_checkpoint,
            e3nn_property_checkpoint=args.e3nn_property_checkpoint,
            e3nn_property_name=args.e3nn_property_name,
            sta_property_model_path=args.sta_property_model,
            sta_density_model_path=args.sta_density_model,
            sta_heating_value_model_path=args.sta_heating_value_model,
            sta_melting_point_model_path=args.sta_melting_point_model,
            sta_boiling_point_model_path=args.sta_boiling_point_model,
            sta_flash_point_model_path=args.sta_flash_point_model,
            sta_formation_enthalpy_model_path=args.sta_formation_enthalpy_model,
            sta_molar_volume_model_path=args.sta_molar_volume_model,
            sta_isp_model_path=args.sta_isp_model,
            density_model=args.density_model,
            heating_value_model=args.heating_value_model,
            melting_point_model=args.melting_point_model,
            boiling_point_model=args.boiling_point_model,
            flash_point_model=args.flash_point_model,
            formation_enthalpy_model=args.formation_enthalpy_model,
            molar_volume_model=args.molar_volume_model,
            isp_model=args.isp_model,
            cbh_bdc_model_path=args.cbh_bdc_model,
            cbh_bdc_feature_csv=args.cbh_bdc_feature_csv,
            cbh_bdc_yang_root=args.cbh_bdc_yang_root,
            phase_fusion_melting_point_model_path=args.phase_fusion_melting_point_model,
            phase_fusion_boiling_point_model_path=args.phase_fusion_boiling_point_model,
            phase_fusion_flash_point_model_path=args.phase_fusion_flash_point_model,
            screen_workers=args.screen_workers,
            property_workers=args.property_workers,
            use_property_cache=not args.no_property_cache,
            reproducibility_scope=args.reproducibility_scope,
            use_cbh_bdc_heating=(
                not args.no_cbh_bdc_heating
                and (
                    args.heating_value_model == "yang_cbh_bdc"
                    or args.formation_enthalpy_model == "yang_cbh_bdc"
                )
            ),
            sta_weak_feedback_enabled=not args.disable_sta_weak_feedback,
            sta_weak_feedback_weight=args.sta_weak_feedback_weight,
            screening_config=ScreeningConfig(
                min_h_to_c=args.min_h_to_c,
                max_h_to_c=args.max_h_to_c,
                min_ring_count=args.min_ring_count,
                max_ring_count=args.max_ring_count,
                max_rotatable_bonds=args.max_rotatable_bonds,
                min_fraction_sp3=args.min_fraction_sp3,
                max_sa_score=args.max_sa_score,
                target_h_to_c=args.target_h_to_c,
                target_ring_count=args.target_ring_count,
                preferred_ring_weight=args.preferred_ring_weight,
                preferred_ring_target=args.preferred_ring_target,
                require_mmff_geometry=not args.skip_mmff_geometry,
                min_mmff_stability_score=args.min_mmff_stability_score,
                max_mmff_energy_per_heavy_atom=args.max_mmff_energy_per_heavy_atom,
                high_novelty_threshold=args.novelty_threshold,
            ),
            score_weights=CompositeScoreWeights(
                property_quality=args.property_score_weight,
                e3nn_geometry=args.e3nn_geometry_score_weight,
                e3nn_symmetry=args.e3nn_symmetry_score_weight,
                synthetic_accessibility=0.0,
                novelty=args.novelty_score_weight,
            ),
            property_score_weights=PropertyScoreWeights(
                density=args.density_score_weight,
                heating_value=args.heating_value_score_weight,
                volumetric_heating_value=(
                    args.volumetric_heating_value_score_weight
                ),
                melting_point=args.melting_point_score_weight,
                boiling_point=args.boiling_point_score_weight,
                flash_point=args.flash_point_score_weight,
                synthetic_accessibility=args.property_sa_score_weight,
                structure_validity=args.structure_validity_score_weight,
                formation_enthalpy=args.formation_enthalpy_score_weight,
                molar_volume=args.molar_volume_score_weight,
                isp=args.isp_score_weight,
            ),
            thresholds=FuelPropertyThresholds(
                density_min_g_cm3=args.density_min,
                heating_value_min_mj_kg=args.heating_value_min,
                heating_value_possible_min_mj_kg=args.heating_value_possible_min,
                melting_point_max_k=args.melting_point_max,
                boiling_point_min_k=args.boiling_point_min,
                flash_point_min_k=args.flash_point_min,
                volumetric_heating_value_min_mj_l=(
                    args.volumetric_heating_value_min
                ),
            ),
            use_melting_point_screen=args.use_melting_point_screen,
            use_boiling_point_screen=args.use_boiling_point_screen,
            use_flash_point_screen=args.use_flash_point_screen,
            use_volumetric_heating_value_screen=(
                args.use_volumetric_heating_value_screen
            ),
            missing_property_penalty=args.missing_property_penalty,
    )




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fuel-design")
    parser.add_argument(
        "--device-mode", choices=("auto", "cpu", "gpu"), default=None,
        help="auto (probe CUDA, fall back to CPU), cpu (never use CUDA) or "
             "gpu (require CUDA; fail if it cannot be used). May be given "
             "before or after the sub-command.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-base")
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--min-carbons", type=int, default=8)
    gen.add_argument("--max-carbons", type=int, default=17)

    partitions = subparsers.add_parser("count-partitions")
    partitions.add_argument("--total-carbons", type=int, required=True)

    gen_counts = subparsers.add_parser("generate-base-by-counts")
    gen_counts.add_argument("--n1", type=int, required=True)
    gen_counts.add_argument("--n2", type=int, required=True)
    gen_counts.add_argument("--n3", type=int, required=True)
    gen_counts.add_argument("--seed", type=int, default=None)

    semi_counts = subparsers.add_parser("generate-semi-base-by-counts")
    semi_counts.add_argument("--n1", type=int, required=True)
    semi_counts.add_argument("--n2", type=int, required=True)
    semi_counts.add_argument("--n3", type=int, required=True)
    semi_counts.add_argument("--seed", type=int, default=None)
    semi_counts.add_argument("--checkpoint", required=True)
    semi_counts.add_argument("--threshold", type=float, default=0.67)
    semi_counts.add_argument("--temperature", type=float, default=1.0)
    semi_counts.add_argument("--sample", action="store_true")

    gen_total = subparsers.add_parser("generate-bases-for-total")
    gen_total.add_argument("--total-carbons", type=int, required=True)
    gen_total.add_argument("--per-partition", type=int, default=1)
    gen_total.add_argument("--seed", type=int, default=None)

    mirror = subparsers.add_parser("mirror")
    mirror.add_argument("--n1", type=int, required=True)
    mirror.add_argument("--n2", type=int, required=True)
    mirror.add_argument("--n3", type=int, required=True)
    mirror.add_argument(
        "--edge",
        action="append",
        default=[],
        help="Half-edge as u,v. Can be repeated.",
    )

    gdb = subparsers.add_parser("preprocess-gdb")
    gdb.add_argument(
        "--archive",
        default=str(SCULPTOR_ROOT / "dataset" / "GDB" / "GDB-13" / "gdb13.g.tgz"),
        help="Path to the GDB saturated hydrocarbon .tgz archive.",
    )
    gdb.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "Generation" / "bond_completion_2" / "data" / "gdb13_g_bond_samples.csv"),
    )
    gdb.add_argument(
        "--known-output",
        default=str(SCULPTOR_ROOT / "Generation" / "screening" / "data" / "known_smiles_gdb13_g.txt"),
    )
    gdb.add_argument(
        "--skips-output",
        default=str(SCULPTOR_ROOT / "Generation" / "bond_completion_2" / "data" / "gdb13_g_skips.csv"),
    )
    gdb.add_argument("--member", action="append", default=None)
    gdb.add_argument("--max-molecules", type=int, default=None)
    gdb.add_argument("--max-molecules-per-member", type=int, default=None)
    gdb.add_argument("--max-lines-per-member", type=int, default=None)
    gdb.add_argument("--variants", type=int, default=3)
    gdb.add_argument("--variant-attempts", type=int, default=50)
    gdb.add_argument("--min-carbons", type=int, default=6)
    gdb.add_argument("--semi-base-samples", action="store_true")
    gdb.add_argument("--seed", type=int, default=0)

    known = subparsers.add_parser("build-known-smiles")
    known.add_argument(
        "--source",
        default=str(SCULPTOR_ROOT / "dataset" / "GDB" / "GDB-13" / "13-saturated_hydrocarbons"),
        help="Directory or .tgz archive containing GDB .smi files.",
    )
    known.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "Generation" / "screening" / "data" / "known_smiles_gdb13_g_full.txt"),
    )
    known.add_argument(
        "--skips-output",
        default=str(SCULPTOR_ROOT / "Generation" / "screening" / "data" / "known_smiles_gdb13_g_full_skips.csv"),
    )
    known.add_argument("--member", action="append", default=None)
    known.add_argument("--min-carbons", type=int, default=None)
    known.add_argument("--max-carbons", type=int, default=None)

    ani = subparsers.add_parser("preprocess-ani1x")
    ani.add_argument("--input", required=True)
    ani.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "evaluation" / "data" / "g_ANI-1x.csv"),
    )
    ani.add_argument(
        "--skips-output",
        default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "evaluation" / "data" / "g_ANI-1x_skips.csv"),
    )
    ani.add_argument("--max-molecules", type=int, default=None)
    ani.add_argument("--max-conformers", type=int, default=None)
    ani.add_argument("--min-carbons", type=int, default=1)
    ani.add_argument("--max-carbons", type=int, default=None)
    ani.add_argument("--omit-coordinates", action="store_true")

    train_bond = subparsers.add_parser("train-bond-model")
    train_bond.add_argument("--csv", required=True)
    train_bond.add_argument(
        "--checkpoint",
        default=str(SCULPTOR_ROOT / "Generation" / "bond_completion_2" / "model" / "bond_completion.pt"),
    )
    train_bond.add_argument("--epochs", type=int, default=20)
    train_bond.add_argument("--batch-size", type=int, default=16)
    train_bond.add_argument("--hidden-dim", type=int, default=64)
    train_bond.add_argument("--num-layers", type=int, default=3)
    train_bond.add_argument("--learning-rate", type=float, default=1e-3)
    train_bond.add_argument("--val-fraction", type=float, default=0.2)
    train_bond.add_argument("--seed", type=int, default=0)
    train_bond.add_argument("--resume", action="store_true")
    train_bond.add_argument("--checkpoint-every", type=int, default=1)
    train_bond.add_argument("--num-workers", type=int, default=0)
    train_bond.add_argument("--disable-amp", action="store_true")

    train_semi = subparsers.add_parser("train-semi-base-model")
    train_semi.add_argument("--csv", required=True)
    train_semi.add_argument(
        "--checkpoint",
        default=str(SCULPTOR_ROOT / "Generation" / "bond_completion_1" / "model" / "semi_base_completion.pt"),
    )
    train_semi.add_argument("--epochs", type=int, default=20)
    train_semi.add_argument("--batch-size", type=int, default=16)
    train_semi.add_argument("--hidden-dim", type=int, default=64)
    train_semi.add_argument("--num-layers", type=int, default=3)
    train_semi.add_argument("--learning-rate", type=float, default=1e-3)
    train_semi.add_argument("--val-fraction", type=float, default=0.2)
    train_semi.add_argument("--seed", type=int, default=0)
    train_semi.add_argument("--resume", action="store_true")
    train_semi.add_argument("--checkpoint-every", type=int, default=1)
    train_semi.add_argument("--num-workers", type=int, default=0)
    train_semi.add_argument("--disable-amp", action="store_true")

    export_train = subparsers.add_parser("export-training-csv")
    export_train.add_argument("--input", required=True)
    export_train.add_argument("--output", required=True)

    export_semi_train = subparsers.add_parser("export-semi-base-training-csv")
    export_semi_train.add_argument("--input", required=True)
    export_semi_train.add_argument("--output", required=True)

    predict_bond = subparsers.add_parser("predict-bonds")
    predict_bond.add_argument("--checkpoint", required=True)
    predict_bond.add_argument("--n1", type=int, required=True)
    predict_bond.add_argument("--n2", type=int, required=True)
    predict_bond.add_argument("--n3", type=int, required=True)
    predict_bond.add_argument("--edge", action="append", default=[])
    predict_bond.add_argument("--threshold", type=float, default=0.5)
    predict_bond.add_argument("--temperature", type=float, default=1.0)
    predict_bond.add_argument("--top-k", type=int, default=None)
    predict_bond.add_argument("--sample", action="store_true")
    predict_bond.add_argument("--seed", type=int, default=None)

    gen_predict = subparsers.add_parser("generate-and-predict")
    gen_predict.add_argument("--checkpoint", required=True)
    gen_predict.add_argument("--total-carbons", type=int, required=True)
    gen_predict.add_argument("--per-partition", type=int, default=1)
    gen_predict.add_argument("--limit", type=int, default=None)
    gen_predict.add_argument("--threshold", type=float, default=0.5)
    gen_predict.add_argument("--temperature", type=float, default=1.0)
    gen_predict.add_argument("--top-k", type=int, default=None)
    gen_predict.add_argument("--sample", action="store_true")
    gen_predict.add_argument("--seed", type=int, default=None)
    gen_predict.add_argument("--semi-base", action="store_true")
    gen_predict.add_argument("--semi-base-checkpoint", default=None)

    eval_gen = subparsers.add_parser("evaluate-generation")
    eval_gen.add_argument("--checkpoint", required=True)
    eval_gen.add_argument("--known-smiles", default=None)
    eval_gen.add_argument("--total-carbons", type=int, required=True)
    eval_gen.add_argument("--samples", type=int, default=100)
    eval_gen.add_argument("--threshold", type=float, default=0.5)
    eval_gen.add_argument("--temperature", type=float, default=1.0)
    eval_gen.add_argument("--top-k", type=int, default=None)
    eval_gen.add_argument("--sample", action="store_true")
    eval_gen.add_argument("--seed", type=int, default=None)
    eval_gen.add_argument("--semi-base", action="store_true")
    eval_gen.add_argument("--semi-base-checkpoint", default=None)
    eval_gen.add_argument("--semi-base-threshold", type=float, default=0.67)

    sweep_topk = subparsers.add_parser("sweep-top-k")
    sweep_topk.add_argument("--checkpoint", required=True)
    sweep_topk.add_argument("--known-smiles", default=None)
    sweep_topk.add_argument("--total-carbons", type=int, required=True)
    sweep_topk.add_argument("--samples", type=int, default=100)
    sweep_topk.add_argument("--threshold", type=float, default=0.5)
    sweep_topk.add_argument("--temperature", type=float, default=1.0)
    sweep_topk.add_argument("--top-k", action="append", type=int, default=None)
    sweep_topk.add_argument("--sample", action="store_true")
    sweep_topk.add_argument("--seed", type=int, default=None)
    sweep_topk.add_argument("--semi-base", action="store_true")
    sweep_topk.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "result" / "topk_sweep.csv"),
    )

    sweep_grid = subparsers.add_parser("sweep-generation-grid")
    sweep_grid.add_argument("--checkpoint", required=True)
    sweep_grid.add_argument("--known-smiles", default=None)
    sweep_grid.add_argument("--total-carbons", action="append", type=int, required=True)
    sweep_grid.add_argument("--seed", action="append", type=int, required=True)
    sweep_grid.add_argument("--top-k", action="append", type=int, required=True)
    sweep_grid.add_argument("--samples", type=int, default=100)
    sweep_grid.add_argument("--threshold", type=float, default=0.5)
    sweep_grid.add_argument("--temperature", type=float, default=1.0)
    sweep_grid.add_argument(
        "--mode",
        choices=("base", "semi-base", "both"),
        default="semi-base",
    )
    sweep_grid.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "result" / "generation_grid.csv"),
    )

    props = subparsers.add_parser("export-generation-properties")
    props.add_argument("--checkpoint", required=True)
    props.add_argument("--known-smiles", default=None)
    props.add_argument("--total-carbons", type=int, required=True)
    props.add_argument("--samples", type=int, default=100)
    props.add_argument("--threshold", type=float, default=0.5)
    props.add_argument("--temperature", type=float, default=1.0)
    props.add_argument("--top-k", type=int, default=2)
    props.add_argument("--seed", type=int, default=None)
    props.add_argument("--semi-base", action="store_true")
    props.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "result" / "generation_properties.csv"),
    )

    screen = subparsers.add_parser("screen-generation")
    screen.add_argument("--checkpoint", required=True)
    screen.add_argument("--known-smiles", default=None)
    screen.add_argument("--total-carbons", type=int, required=True)
    screen.add_argument("--samples", type=int, default=100)
    screen.add_argument("--threshold", type=float, default=0.5)
    screen.add_argument("--temperature", type=float, default=1.0)
    screen.add_argument("--top-k", type=int, default=2)
    screen.add_argument("--seed", type=int, default=None)
    screen.add_argument("--semi-base", action="store_true")
    screen.add_argument("--allow-known", action="store_true")
    screen.add_argument("--allow-disconnected", action="store_true")
    screen.add_argument("--min-h-to-c", type=float, default=0.7)
    screen.add_argument("--max-h-to-c", type=float, default=2.2)
    screen.add_argument("--min-ring-count", type=int, default=1)
    screen.add_argument("--max-ring-count", type=int, default=6)
    screen.add_argument("--max-rotatable-bonds", type=int, default=4)
    screen.add_argument("--min-fraction-sp3", type=float, default=0.95)
    screen.add_argument("--max-sa-score", type=float, default=4.5)
    screen.add_argument("--target-h-to-c", type=float, default=1.35)
    screen.add_argument("--target-ring-count", type=int, default=4)
    screen.add_argument("--preferred-ring-weight", type=float, default=0.10)
    screen.add_argument("--preferred-ring-target", type=int, default=2)
    screen.add_argument("--novelty-threshold", type=float, default=0.75)
    screen.add_argument("--skip-mmff-geometry", action="store_true")
    screen.add_argument("--min-mmff-stability-score", type=float, default=0.05)
    screen.add_argument("--max-mmff-energy-per-heavy-atom", type=float, default=120.0)
    screen.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "result" / "screened_generation.csv"),
    )
    screen.add_argument(
        "--high-novelty-output",
        default=str(SCULPTOR_ROOT / "result" / "high_novelty_generation.csv"),
    )
    screen.add_argument(
        "--summary-output",
        default=str(SCULPTOR_ROOT / "result" / "screened_generation_summary.json"),
    )

    prop_train = subparsers.add_parser("train-property-model")
    prop_train.add_argument("--csv", required=True)
    prop_train.add_argument("--smiles-column", default="smiles")
    prop_train.add_argument("--target-column", default="target")
    prop_train.add_argument(
        "--checkpoint",
        default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "evaluation" / "model" / "e3nn" / "e3nn_property_model.pt"),
    )
    prop_train.add_argument("--max-rows", type=int, default=None)
    prop_train.add_argument("--epochs", type=int, default=50)
    prop_train.add_argument("--hidden-dim", type=int, default=64)
    prop_train.add_argument("--learning-rate", type=float, default=1e-3)
    prop_train.add_argument("--val-fraction", type=float, default=0.2)
    prop_train.add_argument("--seed", type=int, default=0)
    prop_train.add_argument(
        "--baseline-mode",
        choices=("none", "composition", "composition-ring"),
        default="none",
    )

    qm9_prop = subparsers.add_parser("prepare-qm9-e3nn-property")
    qm9_prop.add_argument("--input", default=str(DEFAULT_QM9_HYDROCARBON_PATH))
    qm9_prop.add_argument("--output", default=str(DEFAULT_QM9_E3NN_PROPERTY_PATH))
    qm9_prop.add_argument("--source-target-column", default="h298")
    qm9_prop.add_argument("--target-column", default="target")
    qm9_prop.add_argument("--property-name", default="qm9_h298_hartree")
    qm9_prop.add_argument("--target-units", default="hartree")
    qm9_prop.add_argument("--min-carbons", type=int, default=None)
    qm9_prop.add_argument("--max-carbons", type=int, default=None)
    qm9_prop.add_argument("--max-rows", type=int, default=None)

    qm9_cbh = subparsers.add_parser("prepare-qm9-cbh-bdc-features")
    qm9_cbh.add_argument("--input", default=str(DEFAULT_QM9_HYDROCARBON_PATH))
    qm9_cbh.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "evaluation" / "data" / "qm9_cbh_bdc_features.csv"),
    )
    qm9_cbh.add_argument(
        "--yang-root",
        default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "CBH-BDC" / "CBH-BDC-deltaML-main"),
    )
    qm9_cbh.add_argument("--rung", type=int, default=2)
    qm9_cbh.add_argument("--smiles-column", default="canonical_smiles")
    qm9_cbh.add_argument("--target-column", default="h298")
    qm9_cbh.add_argument("--min-carbons", type=int, default=None)
    qm9_cbh.add_argument("--max-carbons", type=int, default=None)
    qm9_cbh.add_argument("--max-rows", type=int, default=None)

    cbh_targets = subparsers.add_parser("prepare-cbh-bdc-formation-targets")
    cbh_targets.add_argument("--input", default=str(DEFAULT_CBH_BDC_FEATURE_PATH))
    cbh_targets.add_argument(
        "--output",
        default=str(DEFAULT_CBH_BDC_FORMATION_FEATURE_PATH),
    )
    cbh_targets.add_argument("--smiles-column", default="canonical_smiles")
    cbh_targets.add_argument("--h298-column", default="h298")
    cbh_targets.add_argument("--target-column", default="formation_enthalpy_kj_mol")

    cbh_train = subparsers.add_parser("train-cbh-bdc-delta-ml")
    cbh_train.add_argument(
        "--csv",
        default=str(DEFAULT_CBH_BDC_FORMATION_FEATURE_PATH),
    )
    cbh_train.add_argument(
        "--model-output",
        default=str(DEFAULT_CBH_BDC_MODEL_PATH),
    )
    cbh_train.add_argument(
        "--prediction-output",
        default=str(DEFAULT_CBH_BDC_PREDICTION_PATH),
    )
    cbh_train.add_argument(
        "--metrics-output",
        default=str(DEFAULT_CBH_BDC_METRICS_PATH),
    )
    cbh_train.add_argument(
        "--yang-root",
        default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "CBH-BDC" / "CBH-BDC-deltaML-main"),
    )
    cbh_train.add_argument("--target-column", default="formation_enthalpy_kj_mol")
    cbh_train.add_argument(
        "--baseline-column",
        default=None,
        help="Optional baseline column. If provided, the model learns target - baseline.",
    )
    cbh_train.add_argument("--smiles-column", default="canonical_smiles")
    cbh_train.add_argument("--max-rows", type=int, default=None)
    cbh_train.add_argument("--test-fraction", type=float, default=0.2)
    cbh_train.add_argument("--seed", type=int, default=0)

    cbh_predict = subparsers.add_parser("predict-cbh-bdc-formation")
    cbh_predict.add_argument("--smiles", default=None)
    cbh_predict.add_argument("--csv", default=None)
    cbh_predict.add_argument("--smiles-column", default="canonical_smiles")
    cbh_predict.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "result" / "cbh_bdc_formation_predictions.csv"),
    )
    cbh_predict.add_argument("--model", default=str(DEFAULT_CBH_BDC_MODEL_PATH))
    cbh_predict.add_argument("--feature-csv", default=str(DEFAULT_CBH_BDC_FORMATION_FEATURE_PATH))
    cbh_predict.add_argument(
        "--yang-root",
        default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "CBH-BDC" / "CBH-BDC-deltaML-main"),
    )
    cbh_predict.add_argument("--target-column", default="formation_enthalpy_kj_mol")
    cbh_predict.add_argument(
        "--target-kind",
        choices=("qm9_h298_hartree", "formation_enthalpy_kj_mol"),
        default="formation_enthalpy_kj_mol",
    )
    cbh_predict.add_argument(
        "--water-phase",
        choices=("gas", "liquid"),
        default="gas",
    )
    cbh_predict.add_argument("--no-auto-train", action="store_true")

    qm9_train = subparsers.add_parser("train-qm9-e3nn-property")
    qm9_train.add_argument("--input", default=str(DEFAULT_QM9_HYDROCARBON_PATH))
    qm9_train.add_argument("--prepared-output", default=str(DEFAULT_QM9_E3NN_PROPERTY_PATH))
    qm9_train.add_argument("--source-target-column", default="h298")
    qm9_train.add_argument("--target-column", default="target")
    qm9_train.add_argument("--property-name", default="qm9_h298_hartree")
    qm9_train.add_argument("--target-units", default="hartree")
    qm9_train.add_argument("--min-carbons", type=int, default=None)
    qm9_train.add_argument("--max-carbons", type=int, default=None)
    qm9_train.add_argument("--max-rows", type=int, default=None)
    qm9_train.add_argument(
        "--checkpoint",
        default=None,
    )
    qm9_train.add_argument("--epochs", type=int, default=50)
    qm9_train.add_argument("--hidden-dim", type=int, default=64)
    qm9_train.add_argument("--learning-rate", type=float, default=1e-3)
    qm9_train.add_argument("--val-fraction", type=float, default=0.2)
    qm9_train.add_argument("--seed", type=int, default=0)
    qm9_train.add_argument(
        "--baseline-mode",
        choices=("none", "composition", "composition-ring"),
        default="composition-ring",
    )

    prop_predict = subparsers.add_parser("predict-property")
    prop_predict.add_argument("--checkpoint", required=True)
    prop_predict.add_argument("--smiles", required=True)
    prop_predict.add_argument("--seed", type=int, default=0)

    geom_score = subparsers.add_parser("score-geometry")
    geom_score.add_argument("--smiles", required=True)
    geom_score.add_argument("--seed", type=int, default=0)

    geom_score_csv = subparsers.add_parser("score-geometry-csv")
    geom_score_csv.add_argument("--csv", required=True)
    geom_score_csv.add_argument("--output", required=True)
    geom_score_csv.add_argument("--smiles-column", default="canonical_smiles")
    geom_score_csv.add_argument("--seed", type=int, default=0)

    geom_train = subparsers.add_parser("train-geometry-scorer")
    geom_train.add_argument("--csv", required=True)
    geom_train.add_argument("--smiles-column", default="canonical_smiles")
    geom_train.add_argument("--target-column", default="e3nn_geometry_score")
    geom_train.add_argument(
        "--checkpoint",
        default=str(SCULPTOR_ROOT / "Generation" / "screening" / "model" / "e3nn_geometry_scorer.pt"),
    )
    geom_train.add_argument("--max-rows", type=int, default=None)
    geom_train.add_argument("--epochs", type=int, default=50)
    geom_train.add_argument("--hidden-dim", type=int, default=64)
    geom_train.add_argument("--learning-rate", type=float, default=1e-3)
    geom_train.add_argument("--val-fraction", type=float, default=0.2)
    geom_train.add_argument("--seed", type=int, default=0)

    geom_predict = subparsers.add_parser("predict-geometry-score")
    geom_predict.add_argument("--checkpoint", required=True)
    geom_predict.add_argument("--smiles", required=True)
    geom_predict.add_argument("--seed", type=int, default=0)

    calibrate = subparsers.add_parser("calibrate-novelty-threshold")
    calibrate.add_argument("--csv", required=True)
    calibrate.add_argument("--novelty-column", default="fingerprint_novelty")
    calibrate.add_argument("--actual-column", default=None)
    calibrate.add_argument("--predicted-column", default=None)
    calibrate.add_argument("--abs-error-column", default=None)
    calibrate.add_argument("--max-mae", type=float, required=True)
    calibrate.add_argument("--max-quantile-error", type=float, required=True)
    calibrate.add_argument("--quantile", type=float, default=0.90)
    calibrate.add_argument("--min-e3nn-fraction", type=float, default=0.20)
    calibrate.add_argument(
        "--error-mode",
        choices=("relative", "absolute"),
        default="relative",
    )
    calibrate.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "result" / "novelty_threshold_calibration.json"),
    )

    rescore = subparsers.add_parser("rescore-candidates")
    rescore.add_argument("--input", required=True)
    rescore.add_argument("--output", required=True)
    rescore.add_argument("--summary-output", default=None)
    rescore.add_argument("--density-column", default="density_g_cm3")
    rescore.add_argument("--heating-value-column", default="heating_value_mj_kg")
    rescore.add_argument("--melting-point-column", default="melting_point_k")
    rescore.add_argument("--boiling-point-column", default="external_boiling_point_k")
    rescore.add_argument("--flash-point-column", default="external_flash_point_k")
    rescore.add_argument(
        "--formation-enthalpy-column",
        default="external_formation_enthalpy_kj_mol",
    )
    rescore.add_argument("--molar-volume-column", default="external_molar_volume_cm3_mol")
    rescore.add_argument("--isp-column", default="external_isp")
    rescore.add_argument("--novelty-column", default="fingerprint_novelty")
    rescore.add_argument("--e3nn-geometry-column", default="e3nn_geometry_score")
    rescore.add_argument("--e3nn-symmetry-column", default="symmetry_consistency")
    rescore.add_argument("--sa-score-column", default="sa_score")
    rescore.add_argument("--density-min", type=float, default=0.85)
    rescore.add_argument("--heating-value-min", type=float, default=38.0)
    rescore.add_argument("--heating-value-possible-min", type=float, default=37.0)
    rescore.add_argument("--melting-point-max", type=float, default=273.15)
    rescore.add_argument("--property-score-weight", type=float, default=0.70)
    rescore.add_argument("--e3nn-geometry-score-weight", type=float, default=0.10)
    rescore.add_argument("--e3nn-symmetry-score-weight", type=float, default=0.10)
    rescore.add_argument("--sa-score-weight", type=float, default=0.0)
    rescore.add_argument("--novelty-score-weight", type=float, default=0.10)
    rescore.add_argument("--density-score-weight", type=float, default=1.0)
    rescore.add_argument("--heating-value-score-weight", type=float, default=1.0)
    rescore.add_argument("--melting-point-score-weight", type=float, default=1.0)
    rescore.add_argument("--boiling-point-score-weight", type=float, default=1.0)
    rescore.add_argument("--flash-point-score-weight", type=float, default=1.0)
    rescore.add_argument("--property-sa-score-weight", type=float, default=1.0)
    rescore.add_argument("--formation-enthalpy-score-weight", type=float, default=0.0)
    rescore.add_argument("--molar-volume-score-weight", type=float, default=0.0)
    rescore.add_argument("--isp-score-weight", type=float, default=0.0)

    export_inputs = subparsers.add_parser("export-candidate-inputs")
    export_inputs.add_argument("--csv", required=True)
    export_inputs.add_argument(
        "--output-dir",
        default=str(SCULPTOR_ROOT / "result" / "candidate_inputs"),
    )
    export_inputs.add_argument("--smiles-column", default="canonical_smiles")
    export_inputs.add_argument("--id-column", default="candidate_id")
    export_inputs.add_argument("--max-rows", type=int, default=None)
    export_inputs.add_argument("--seed", type=int, default=0)

    thesis_density = subparsers.add_parser("estimate-thesis-density")
    thesis_density.add_argument("--smiles", required=True)
    thesis_density.add_argument("--seed", type=int, default=0)
    thesis_density.add_argument("--radius-scale", type=float, default=1.4)
    thesis_density.add_argument("--surface-points-per-atom", type=int, default=960)
    thesis_density.add_argument("--novelty-score", type=float, default=None)
    thesis_density.add_argument("--high-novelty", action="store_true")

    train_mp = subparsers.add_parser("train-sfs-vr-melting")
    train_mp.add_argument(
        "--data",
        default=str(DEFAULT_SFS_VR_MP_DATA_PATH),
    )
    train_mp.add_argument(
        "--model-output",
        default=str(DEFAULT_SFS_VR_MP_MODEL_PATH),
    )

    predict_mp = subparsers.add_parser("predict-sfs-vr-melting")
    predict_mp.add_argument("--smiles", default=None)
    predict_mp.add_argument("--csv", default=None)
    predict_mp.add_argument("--smiles-column", default="canonical_smiles")
    predict_mp.add_argument("--novelty-score", type=float, default=None)
    predict_mp.add_argument("--novelty-column", default="fingerprint_novelty")
    predict_mp.add_argument(
        "--output",
        default=str(SCULPTOR_ROOT / "result" / "sfs_vr_melting_predictions.csv"),
    )
    predict_mp.add_argument(
        "--data",
        default=str(DEFAULT_SFS_VR_MP_DATA_PATH),
    )
    predict_mp.add_argument(
        "--model",
        default=str(DEFAULT_SFS_VR_MP_MODEL_PATH),
    )
    predict_mp.add_argument("--high-novelty", action="store_true")

    thesis_heat = subparsers.add_parser("heating-value-from-formation")
    thesis_heat.add_argument("--smiles", required=True)
    thesis_heat.add_argument("--formation-enthalpy-kj-mol", type=float, required=True)
    thesis_heat.add_argument(
        "--water-phase",
        choices=("gas", "liquid"),
        default="gas",
    )

    perturb_prepare = subparsers.add_parser("prepare-perturbation-campaign")
    perturb_prepare.add_argument("--finalized-hits", required=True)
    perturb_prepare.add_argument("--output-dir", required=True)
    perturb_prepare.add_argument("--samples", type=int, required=True)
    perturb_prepare.add_argument("--seed", type=int, default=0)
    perturb_prepare.add_argument("--density-min", type=float, default=0.85)
    perturb_prepare.add_argument("--heating-value-min", type=float, default=38.0)
    perturb_prepare.add_argument("--melting-point-max", type=float, default=273.15)
    perturb_prepare.add_argument(
        "--operation",
        action="append",
        choices=sorted(ALLOWED_PERTURBATION_OPERATIONS),
        default=None,
        help="Perturbation operation to enable. Repeat to use multiple; omit for all.",
    )
    perturb_prepare.add_argument("--max-seeds", type=int, default=None)

    perturb_generate = subparsers.add_parser("generate-perturbations")
    perturb_generate.add_argument("--finalized-hits", required=True)
    perturb_generate.add_argument("--output-dir", required=True)
    perturb_generate.add_argument("--samples", type=int, required=True)
    perturb_generate.add_argument("--known-smiles", default=None)
    perturb_generate.add_argument("--temporary-smiles", default=None)
    perturb_generate.add_argument("--seed", type=int, default=0)
    perturb_generate.add_argument("--density-min", type=float, default=0.85)
    perturb_generate.add_argument("--heating-value-min", type=float, default=38.0)
    perturb_generate.add_argument("--melting-point-max", type=float, default=273.15)
    perturb_generate.add_argument(
        "--operation",
        action="append",
        choices=sorted(ALLOWED_PERTURBATION_OPERATIONS),
        default=None,
        help="Perturbation operation to enable. Repeat to use multiple; omit for all.",
    )
    perturb_generate.add_argument("--max-seeds", type=int, default=None)


    perturb_pipeline = subparsers.add_parser("run-perturbation-pipeline")
    perturb_pipeline.add_argument("--finalized-hits", required=True)
    perturb_pipeline.add_argument("--output-dir", required=True)
    perturb_pipeline.add_argument("--samples", type=int, required=True)
    perturb_pipeline.add_argument(
        "--samples-per-seed-operation",
        action="store_true",
        help="Apply --samples separately to every selected seed and operation.",
    )
    perturb_pipeline.add_argument("--known-smiles", default=None)
    perturb_pipeline.add_argument("--temporary-smiles", default=None)
    perturb_pipeline.add_argument("--perturbation-temporary-smiles", default=None)
    perturb_pipeline.add_argument(
        "--screen-workers",
        type=int,
        default=1,
        help="worker processes for the MMFF screening pass (default: 1; 0 = automatic)",
    )
    perturb_pipeline.add_argument(
        "--property-workers",
        type=int,
        default=1,
        help="worker processes for property prediction (default: 1; 0 = automatic)",
    )
    perturb_pipeline.add_argument(
        "--no-property-cache",
        action="store_true",
        help="do not reuse property predictions cached in the output directory",
    )
    perturb_pipeline.add_argument(
        "--reproducibility-scope",
        choices=["all", "tier12", "tier123"],
        default="tier12",
        help="which candidates get the confirmatory reproducibility diagnostic",
    )
    perturb_pipeline.add_argument(
        "--replace-existing-products",
        action="store_true",
        help=(
            "Replace the perturbation products already stored in the output "
            "folder instead of merging this campaign into them (the default "
            "keeps every campaign in the same tables)."
        ),
    )
    perturb_pipeline.add_argument("--known-fuel-check", action="store_true")
    perturb_pipeline.add_argument("--seed", type=int, default=0)
    perturb_pipeline.add_argument("--density-min", type=float, default=0.85)
    perturb_pipeline.add_argument("--heating-value-min", type=float, default=38.0)
    perturb_pipeline.add_argument("--heating-value-possible-min", type=float, default=37.0)
    perturb_pipeline.add_argument("--melting-point-max", type=float, default=273.15)
    perturb_pipeline.add_argument("--boiling-point-min", type=float, default=450.0)
    perturb_pipeline.add_argument("--flash-point-min", type=float, default=310.0)
    perturb_pipeline.add_argument(
        "--volumetric-heating-value-min",
        type=float,
        default=32.3,
    )
    perturb_pipeline.add_argument("--novelty-threshold", type=float, default=0.75)
    perturb_pipeline.add_argument("--uncertainty-scale", type=float, default=1.0)
    perturb_pipeline.add_argument("--ignore-density", action="store_true")
    perturb_pipeline.add_argument("--use-heating-value", action="store_true")
    perturb_pipeline.add_argument("--use-melting-point-screen", action="store_true")
    perturb_pipeline.add_argument("--use-boiling-point-screen", action="store_true")
    perturb_pipeline.add_argument("--use-flash-point-screen", action="store_true")
    perturb_pipeline.add_argument(
        "--use-volumetric-heating-value-screen",
        action="store_true",
    )
    perturb_pipeline.add_argument("--missing-property-penalty", type=float, default=0.15)
    perturb_pipeline.add_argument("--heating-values-csv", default=None)
    perturb_pipeline.add_argument("--mp-model", default=str(DEFAULT_SFS_VR_MP_MODEL_PATH))
    perturb_pipeline.add_argument("--mp-data", default=str(DEFAULT_SFS_VR_MP_DATA_PATH))
    perturb_pipeline.add_argument("--e3nn-geometry-checkpoint", default=None)
    perturb_pipeline.add_argument("--e3nn-property-checkpoint", default=None)
    perturb_pipeline.add_argument("--e3nn-property-name", default="qm9_h298_hartree")
    perturb_pipeline.add_argument("--sta-property-model", default=None)
    perturb_pipeline.add_argument(
        "--sta-density-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["density_g_cm3"]),
    )
    perturb_pipeline.add_argument("--sta-heating-value-model", default=None)
    perturb_pipeline.add_argument(
        "--sta-melting-point-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["melting_point_k"]),
    )
    perturb_pipeline.add_argument(
        "--sta-boiling-point-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS.get("boiling_point_k", DEFAULT_STA_MODEL_PATH)),
    )
    perturb_pipeline.add_argument(
        "--sta-flash-point-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["flash_point_k"]),
    )
    perturb_pipeline.add_argument(
        "--phase-fusion-melting-point-model",
        default=str(DEFAULT_PHASE_FUSION_MODEL_PATHS["melting_point_k"]),
    )
    perturb_pipeline.add_argument(
        "--phase-fusion-boiling-point-model",
        default=str(DEFAULT_PHASE_FUSION_MODEL_PATHS["boiling_point_k"]),
    )
    perturb_pipeline.add_argument(
        "--phase-fusion-flash-point-model",
        default=str(DEFAULT_PHASE_FUSION_MODEL_PATHS["flash_point_k"]),
    )
    perturb_pipeline.add_argument(
        "--sta-formation-enthalpy-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["formation_enthalpy_kj_mol"]),
    )
    perturb_pipeline.add_argument(
        "--sta-molar-volume-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["molar_volume_cm3_mol"]),
    )
    perturb_pipeline.add_argument(
        "--sta-isp-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["isp"]),
    )
    perturb_pipeline.add_argument(
        "--density-model",
        choices=("thesis_vdw_density", "sta", "none"),
        default="thesis_vdw_density",
    )
    perturb_pipeline.add_argument(
        "--heating-value-model",
        choices=("yang_cbh_bdc", "sta", "none"),
        default="yang_cbh_bdc",
    )
    perturb_pipeline.add_argument(
        "--melting-point-model",
        choices=("sfs_vr_mp", "sta", "phase_fusion", "none"),
        default="phase_fusion",
    )
    perturb_pipeline.add_argument(
        "--boiling-point-model",
        choices=("none", "sta", "phase_fusion"),
        default="none",
    )
    perturb_pipeline.add_argument(
        "--flash-point-model",
        choices=("none", "sta", "phase_fusion"),
        default="none",
    )
    perturb_pipeline.add_argument(
        "--formation-enthalpy-model",
        choices=("yang_cbh_bdc", "sta", "none"),
        default="yang_cbh_bdc",
    )
    perturb_pipeline.add_argument(
        "--molar-volume-model",
        choices=("derived", "none"),
        default="derived",
    )
    perturb_pipeline.add_argument(
        "--isp-model",
        choices=("none", "sta"),
        default="none",
    )
    perturb_pipeline.add_argument("--disable-sta-weak-feedback", action="store_true")
    perturb_pipeline.add_argument("--sta-weak-feedback-weight", type=float, default=0.35)
    perturb_pipeline.add_argument("--cbh-bdc-model", default=str(DEFAULT_CBH_BDC_MODEL_PATH))
    perturb_pipeline.add_argument(
        "--cbh-bdc-feature-csv",
        default=str(DEFAULT_CBH_BDC_FORMATION_FEATURE_PATH),
    )
    perturb_pipeline.add_argument(
        "--cbh-bdc-yang-root",
        default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "CBH-BDC" / "CBH-BDC-deltaML-main"),
    )
    perturb_pipeline.add_argument("--no-cbh-bdc-heating", action="store_true")
    perturb_pipeline.add_argument(
        "--operation",
        action="append",
        choices=sorted(ALLOWED_PERTURBATION_OPERATIONS),
        default=None,
        help="Perturbation operation to enable. Repeat to use multiple; omit for all.",
    )
    perturb_pipeline.add_argument("--max-seeds", type=int, default=None)
    perturb_pipeline.add_argument("--min-h-to-c", type=float, default=0.7)
    perturb_pipeline.add_argument("--max-h-to-c", type=float, default=2.2)
    perturb_pipeline.add_argument("--min-ring-count", type=int, default=1)
    perturb_pipeline.add_argument("--max-ring-count", type=int, default=6)
    perturb_pipeline.add_argument("--max-rotatable-bonds", type=int, default=4)
    perturb_pipeline.add_argument("--min-fraction-sp3", type=float, default=0.95)
    perturb_pipeline.add_argument("--max-sa-score", type=float, default=4.5)
    perturb_pipeline.add_argument("--target-h-to-c", type=float, default=1.35)
    perturb_pipeline.add_argument("--target-ring-count", type=int, default=4)
    perturb_pipeline.add_argument("--preferred-ring-weight", type=float, default=0.10)
    perturb_pipeline.add_argument("--preferred-ring-target", type=int, default=2)
    perturb_pipeline.add_argument("--skip-mmff-geometry", action="store_true")
    perturb_pipeline.add_argument("--min-mmff-stability-score", type=float, default=0.05)
    perturb_pipeline.add_argument("--max-mmff-energy-per-heavy-atom", type=float, default=120.0)
    perturb_pipeline.add_argument("--property-score-weight", type=float, default=0.70)
    perturb_pipeline.add_argument(
        "--route-score-weight",
        type=float,
        dest="property_score_weight",
        help=argparse.SUPPRESS,
    )
    perturb_pipeline.add_argument("--e3nn-geometry-score-weight", type=float, default=0.15)
    perturb_pipeline.add_argument("--e3nn-symmetry-score-weight", type=float, default=0.15)
    perturb_pipeline.add_argument("--sa-score-weight", type=float, default=0.0)
    perturb_pipeline.add_argument("--novelty-score-weight", type=float, default=0.0)
    perturb_pipeline.add_argument("--density-score-weight", type=float, default=1.0)
    perturb_pipeline.add_argument("--heating-value-score-weight", type=float, default=1.0)
    perturb_pipeline.add_argument(
        "--volumetric-heating-value-score-weight",
        type=float,
        default=0.0,
    )
    perturb_pipeline.add_argument("--melting-point-score-weight", type=float, default=1.0)
    perturb_pipeline.add_argument("--boiling-point-score-weight", type=float, default=1.0)
    perturb_pipeline.add_argument("--flash-point-score-weight", type=float, default=1.0)
    perturb_pipeline.add_argument("--property-sa-score-weight", type=float, default=1.0)
    perturb_pipeline.add_argument(
        "--structure-validity-score-weight",
        type=float,
        default=1.0,
    )
    perturb_pipeline.add_argument("--formation-enthalpy-score-weight", type=float, default=0.0)
    perturb_pipeline.add_argument("--molar-volume-score-weight", type=float, default=0.0)
    perturb_pipeline.add_argument("--isp-score-weight", type=float, default=0.0)
    perturb_pipeline.add_argument(
        "--iterations",
        type=int,
        default=1,
        help=(
            "Iterative perturbation rounds towards Tier 1; 1 keeps the plain "
            "single campaign behaviour."
        ),
    )
    perturb_pipeline.add_argument(
        "--per-round-budget", type=int, default=2000
    )
    perturb_pipeline.add_argument(
        "--max-seeds-per-round", type=int, default=300
    )
    perturb_pipeline.add_argument(
        "--attempts-per-seed-operation", type=int, default=10
    )
    perturb_pipeline.add_argument(
        "--attempts-per-molecule",
        type=int,
        default=0,
        help=(
            "Iterative mode: total perturbations per lead and round, split over "
            "the operators by the model's sampling weights (0 = use "
            "--attempts-per-seed-operation instead)."
        ),
    )
    perturb_pipeline.add_argument(
        "--operator-attempt",
        action="append",
        default=None,
        help=(
            "Iterative mode: starting attempts per molecule for one operator, as "
            "NAME=COUNT. Repeat for the operators you want to steer; the rounds "
            "then adapt this split from what they measured."
        ),
    )
    perturb_pipeline.add_argument(
        "--operator-adaptation-rate",
        type=float,
        default=0.5,
        help="How strongly a round shifts the operator split (0 = keep it fixed).",
    )
    perturb_pipeline.add_argument(
        "--ablation-arm",
        default=None,
        choices=["A", "B", "C", "D", "E", "a", "b", "c", "d", "e"],
        help=(
            "Controlled ablation of the iterative line: A property ranking only, "
            "B plus ring reward/penalty, C plus ring expansion, D plus adaptive "
            "operator split, E ring expansion only. Every arm shares the same "
            "leads, rounds, attempts and budget."
        ),
    )
    perturb_pipeline.add_argument(
        "--rounds-without-improvement", type=int, default=0
    )
    perturb_pipeline.add_argument("--iteration-model", default=None)
    perturb_pipeline.add_argument(
        "--seed-pool",
        action="append",
        default=None,
        help="Channel used as an iteration seed pool; repeat the flag to add more.",
    )

    sta_prepare = subparsers.add_parser("prepare-sta-gdb-dataset")
    sta_prepare.add_argument(
        "--input",
        action="append",
        required=True,
        help="GDB property CSV path. Repeat to merge multiple CSV files.",
    )
    sta_prepare.add_argument("--output", default=str(DEFAULT_STA_DATA_PATH))
    sta_prepare.add_argument("--target", action="append", default=None)

    sta_train = subparsers.add_parser("train-sta-property-model")
    sta_train.add_argument("--data", default=str(DEFAULT_STA_DATA_PATH))
    sta_train.add_argument("--output", default=str(DEFAULT_STA_MODEL_PATH))
    sta_train.add_argument("--target", action="append", default=None)
    sta_train.add_argument("--epochs", type=int, default=100)
    sta_train.add_argument("--batch-size", type=int, default=10)
    sta_train.add_argument("--d-model", type=int, default=512)
    sta_train.add_argument("--num-heads", type=int, default=8)
    sta_train.add_argument("--num-layers", type=int, default=12)
    sta_train.add_argument("--learning-rate", type=float, default=1e-5)
    sta_train.add_argument("--enumerations-per-molecule", type=int, default=1)
    sta_train.add_argument("--validation-fraction", type=float, default=0.1)
    sta_train.add_argument("--seed", type=int, default=42)

    sta_specialists = subparsers.add_parser("train-sta-specialist-models")
    sta_specialists.add_argument("--data", default=str(DEFAULT_STA_DATA_PATH))
    sta_specialists.add_argument("--output-dir", default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "evaluation" / "model" / "STA"))
    sta_specialists.add_argument("--epochs", type=int, default=30)
    sta_specialists.add_argument("--batch-size", type=int, default=4096)
    sta_specialists.add_argument("--d-model", type=int, default=128)
    sta_specialists.add_argument("--num-heads", type=int, default=4)
    sta_specialists.add_argument("--num-layers", type=int, default=3)
    sta_specialists.add_argument("--learning-rate", type=float, default=1e-4)
    sta_specialists.add_argument("--validation-fraction", type=float, default=0.1)
    sta_specialists.add_argument("--seed", type=int, default=42)

    sta_optimized = subparsers.add_parser("train-sta-optimized-property-models")
    sta_optimized.add_argument("--data", default=str(DEFAULT_STA_DATA_PATH))
    sta_optimized.add_argument("--output-dir", default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "evaluation" / "model" / "STA"))
    sta_optimized.add_argument("--validation-fraction", type=float, default=0.1)
    sta_optimized.add_argument("--seed", type=int, default=42)

    sta_paper = subparsers.add_parser("train-sta-paper-property-models")
    sta_paper.add_argument("--data", default=str(DEFAULT_STA_GDB13_DATA_PATH))
    sta_paper.add_argument("--output-dir", default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "evaluation" / "model" / "STA"))
    sta_paper.add_argument("--batch-size", type=int, default=None)
    sta_paper.add_argument("--epochs", type=int, default=None)
    sta_paper.add_argument("--validation-fraction", type=float, default=0.1)
    sta_paper.add_argument("--seed", type=int, default=42)

    sta_predict = subparsers.add_parser("predict-sta-properties")
    sta_predict.add_argument("--input", required=True)
    sta_predict.add_argument("--output", required=True)
    sta_predict.add_argument("--model", default=str(DEFAULT_STA_MODEL_PATH))
    sta_predict.add_argument("--smiles-column", default="canonical_smiles")

    sta_feedback = subparsers.add_parser("export-sta-weak-feedback")
    sta_feedback.add_argument("--input", required=True)
    sta_feedback.add_argument("--output", required=True)
    sta_feedback.add_argument("--novelty-threshold", type=float, default=0.75)
    sta_feedback.add_argument("--base-label-weight", type=float, default=0.35)
    sta_feedback.add_argument("--density-min", type=float, default=0.85)
    sta_feedback.add_argument("--heating-value-min", type=float, default=38.0)
    sta_feedback.add_argument("--heating-value-possible-min", type=float, default=37.0)
    sta_feedback.add_argument("--melting-point-max", type=float, default=273.15)
    sta_feedback.add_argument(
        "--water-phase",
        choices=("gas", "liquid"),
        default="gas",
    )

    phase_fusion = subparsers.add_parser("train-phase-fusion-models")
    phase_fusion.add_argument(
        "--data",
        default=str(DEFAULT_STA_GDB13_DATA_PATH),
    )
    phase_fusion.add_argument(
        "--output-dir",
        default=None,
        help="training outputs; defaults to a total-reference 2D/3D result directory",
    )
    phase_fusion.add_argument(
        "--model-dir",
        default=None,
        help="artifact output directory; defaults to the selected 2D/3D schema directory",
    )
    phase_fusion.add_argument("--feature-cache", default=None)
    phase_fusion.add_argument(
        "--target",
        action="append",
        choices=DEFAULT_PHASE_TARGETS,
        default=None,
    )
    phase_fusion.add_argument("--random-state", type=int, default=42)
    phase_fusion.add_argument("--max-rows", type=int, default=None)
    phase_feature_group = phase_fusion.add_mutually_exclusive_group()
    phase_feature_group.add_argument(
        "--include-e3nn-features",
        action="store_true",
        help="train the optional 3D descriptor variant (the default is the reported 2D head)",
    )
    phase_feature_group.add_argument(
        "--no-e3nn-features",
        action="store_false",
        dest="include_e3nn_features",
        help="compatibility alias for the default 2D descriptor head",
    )
    phase_fusion.set_defaults(include_e3nn_features=False)
    phase_fusion.add_argument(
        "--model-family",
        choices=("mlp_ensemble", "legacy_tree"),
        default="mlp_ensemble",
    )
    phase_fusion.add_argument("--ensemble-size", type=int, default=2)
    phase_fusion.add_argument("--mlp-epochs", type=int, default=100)
    phase_fusion.add_argument("--mlp-batch-size", type=int, default=1024)
    phase_fusion.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    phase_fusion.add_argument("--mlp-patience", type=int, default=12)
    phase_fusion.add_argument("--workers", type=int, default=1)

    pipeline = subparsers.add_parser("run-pipeline")
    seed_t123 = subparsers.add_parser("build-t123-seeds")
    seed_t123.add_argument("--output-dir", required=True)
    seed_t123.add_argument("--output", default=None)
    seed_potential = subparsers.add_parser("build-potential-seeds")
    seed_potential.add_argument("--output-dir", required=True)
    seed_potential.add_argument("--output", default=None)
    seed_potential.add_argument(
        "--one-item-fraction",
        type=float,
        default=POTENTIAL_SEED_TOP_FRACTION,
        help=(
            "Share of each one-beaten potential pool (Potential 2 and "
            "Potential 3) that becomes a seed, ranked by the composite score "
            "(default: best 10%% of each)."
        ),
    )
    run_retier = subparsers.add_parser("retier-run-outputs")
    run_retier.add_argument("--output-dir", required=True)
    train_iteration = subparsers.add_parser("train-perturbation-iteration-model")
    train_iteration.add_argument("--output-dir", required=True)
    train_iteration.add_argument("--model-out", default=None)
    train_iteration.add_argument("--metrics-out", default=None)
    train_iteration.add_argument("--model-family", default=None)
    compare_iteration = subparsers.add_parser("compare-perturbation-iteration-models")
    compare_iteration.add_argument("--output-dir", required=True)
    compare_iteration.add_argument("--folds", type=int, default=5)
    compare_iteration.add_argument("--top", type=int, default=3)
    compare_iteration.add_argument("--model-dir", default=None)
    compare_iteration.add_argument(
        "--no-save-best",
        action="store_true",
        help="Report the comparison without writing the winner to the production path.",
    )
    run_organize = subparsers.add_parser("organize-run-outputs")
    run_organize.add_argument("--output-dir", required=True)
    run_organize.add_argument("--report", default=None)
    pipeline.add_argument("--checkpoint", required=True)
    pipeline.add_argument(
        "--semi-base-checkpoint",
        default=None,
        help="Dedicated base-to-semi-base GNN checkpoint; defaults to --checkpoint.",
    )
    pipeline.add_argument("--semi-base-threshold", type=float, default=0.73)
    pipeline.add_argument("--semi-base-temperature", type=float, default=1.0)
    pipeline.add_argument("--semi-base-top-k", type=int, default=None)
    pipeline.add_argument("--known-smiles", required=True)
    pipeline.add_argument("--temporary-smiles", default=None)
    pipeline.add_argument("--known-fuel-check", action="store_true")
    pipeline.add_argument("--total-carbons", type=int, required=True)
    pipeline.add_argument("--samples", type=int, default=100)
    pipeline.add_argument("--iterations", type=int, default=1)
    pipeline.add_argument("--threshold", type=float, default=0.94)
    pipeline.add_argument("--temperature", type=float, default=1.0)
    pipeline.add_argument("--top-k", type=int, default=2)
    pipeline.add_argument("--seed", type=int, default=None)
    pipeline.add_argument("--feedback-reward", action="store_true")
    pipeline.add_argument("--feedback-strength", type=float, default=1.25)
    pipeline.add_argument(
        "--disable-online-property-feedback",
        action="store_true",
        help="Keep iterative feedback structural-only instead of predicting properties each round.",
    )
    pipeline.add_argument(
        "--property-feedback-warmup-iterations",
        type=int,
        default=20,
    )
    pipeline.add_argument(
        "--property-feedback-warmup-weight",
        type=float,
        default=0.35,
    )
    pipeline.add_argument(
        "--property-feedback-weight",
        type=float,
        default=0.70,
    )
    pipeline.add_argument(
        "--property-reward-slope",
        type=float,
        default=2.5,
        help="Contrast multiplier around the useful-property score boundary.",
    )
    pipeline.add_argument(
        "--partition-feedback-gain",
        type=float,
        default=6.0,
        help="Learning gain for n1/n2/n3 partition sampling weights.",
    )
    pipeline.add_argument(
        "--edge-feedback-gain",
        type=float,
        default=2.0,
        help="Learning gain for topology-aware bond action biases.",
    )
    pipeline.add_argument(
        "--feedback-exploration-fraction",
        type=float,
        default=0.12,
        help="Minimum uniform exploration share retained during feedback.",
    )
    pipeline.add_argument("--disable-sta-weak-feedback", action="store_true")
    pipeline.add_argument("--sta-weak-feedback-weight", type=float, default=0.35)
    pipeline.add_argument("--base-mode", action="store_true")
    pipeline.add_argument("--novelty-threshold", type=float, default=0.75)
    pipeline.add_argument("--max-sa-score", type=float, default=4.5)
    pipeline.add_argument("--preferred-ring-weight", type=float, default=0.10)
    pipeline.add_argument("--preferred-ring-target", type=int, default=2)
    pipeline.add_argument(
        "--objective-mode",
        choices=("property_maximize", "target_property", "novelty_exploration"),
        default="property_maximize",
    )
    pipeline.add_argument("--target-density", type=float, default=None)
    pipeline.add_argument("--target-heating-value", type=float, default=None)
    pipeline.add_argument(
        "--target-volumetric-heating-value",
        type=float,
        default=None,
    )
    pipeline.add_argument("--target-melting-point", type=float, default=None)
    pipeline.add_argument("--target-boiling-point", type=float, default=None)
    pipeline.add_argument("--target-flash-point", type=float, default=None)
    pipeline.add_argument("--objective-relative-window", type=float, default=0.15)
    pipeline.add_argument("--property-score-weight", type=float, default=0.65)
    pipeline.add_argument("--e3nn-geometry-score-weight", type=float, default=0.10)
    pipeline.add_argument("--e3nn-symmetry-score-weight", type=float, default=0.10)
    pipeline.add_argument("--sa-score-weight", type=float, default=0.0)
    pipeline.add_argument("--novelty-score-weight", type=float, default=0.15)
    pipeline.add_argument("--density-score-weight", type=float, default=1.0)
    pipeline.add_argument("--heating-value-score-weight", type=float, default=1.0)
    pipeline.add_argument(
        "--volumetric-heating-value-score-weight",
        type=float,
        default=0.0,
    )
    pipeline.add_argument("--melting-point-score-weight", type=float, default=0.0)
    pipeline.add_argument("--boiling-point-score-weight", type=float, default=0.0)
    pipeline.add_argument("--flash-point-score-weight", type=float, default=0.0)
    pipeline.add_argument("--property-sa-score-weight", type=float, default=1.0)
    pipeline.add_argument("--structure-validity-score-weight", type=float, default=1.0)
    pipeline.add_argument("--formation-enthalpy-score-weight", type=float, default=0.0)
    pipeline.add_argument("--molar-volume-score-weight", type=float, default=0.0)
    pipeline.add_argument("--isp-score-weight", type=float, default=0.0)
    pipeline.add_argument("--heating-values-csv", default=None)
    pipeline.add_argument("--ignore-density", action="store_true")
    pipeline.add_argument("--ignore-heating-value", action="store_true")
    pipeline.add_argument("--use-melting-point-screen", action="store_true")
    pipeline.add_argument("--use-boiling-point-screen", action="store_true")
    pipeline.add_argument("--use-flash-point-screen", action="store_true")
    pipeline.add_argument(
        "--use-volumetric-heating-value-screen",
        action="store_true",
    )
    pipeline.add_argument("--missing-property-penalty", type=float, default=0.15)
    pipeline.add_argument("--e3nn-geometry-checkpoint", default=None)
    pipeline.add_argument("--e3nn-property-checkpoint", default=None)
    pipeline.add_argument("--e3nn-property-name", default="qm9_h298_hartree")
    pipeline.add_argument("--sta-property-model", default=None)
    pipeline.add_argument(
        "--sta-density-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["density_g_cm3"]),
    )
    pipeline.add_argument(
        "--sta-heating-value-model",
        default=None,
    )
    pipeline.add_argument(
        "--sta-melting-point-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["melting_point_k"]),
    )
    pipeline.add_argument(
        "--sta-boiling-point-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS.get("boiling_point_k", DEFAULT_STA_MODEL_PATH)),
    )
    pipeline.add_argument(
        "--sta-flash-point-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["flash_point_k"]),
    )
    pipeline.add_argument(
        "--phase-fusion-melting-point-model",
        default=str(DEFAULT_PHASE_FUSION_MODEL_PATHS["melting_point_k"]),
    )
    pipeline.add_argument(
        "--phase-fusion-boiling-point-model",
        default=str(DEFAULT_PHASE_FUSION_MODEL_PATHS["boiling_point_k"]),
    )
    pipeline.add_argument(
        "--phase-fusion-flash-point-model",
        default=str(DEFAULT_PHASE_FUSION_MODEL_PATHS["flash_point_k"]),
    )
    pipeline.add_argument(
        "--sta-formation-enthalpy-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["formation_enthalpy_kj_mol"]),
    )
    pipeline.add_argument(
        "--sta-molar-volume-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["molar_volume_cm3_mol"]),
    )
    pipeline.add_argument(
        "--sta-isp-model",
        default=str(DEFAULT_SPECIALIST_MODEL_PATHS["isp"]),
    )
    pipeline.add_argument(
        "--density-model",
        choices=("thesis_vdw_density", "sta", "none"),
        default="thesis_vdw_density",
    )
    pipeline.add_argument(
        "--heating-value-model",
        choices=("yang_cbh_bdc", "sta", "none"),
        default="yang_cbh_bdc",
    )
    pipeline.add_argument(
        "--melting-point-model",
        choices=("sfs_vr_mp", "sta", "phase_fusion", "none"),
        default="phase_fusion",
    )
    pipeline.add_argument(
        "--boiling-point-model",
        choices=("none", "sta", "phase_fusion"),
        default="none",
    )
    pipeline.add_argument(
        "--flash-point-model",
        choices=("none", "sta", "phase_fusion"),
        default="none",
    )
    pipeline.add_argument(
        "--formation-enthalpy-model",
        choices=("yang_cbh_bdc", "sta", "none"),
        default="yang_cbh_bdc",
    )
    pipeline.add_argument(
        "--molar-volume-model",
        choices=("derived", "none"),
        default="derived",
    )
    pipeline.add_argument(
        "--isp-model",
        choices=("none", "sta"),
        default="none",
    )
    pipeline.add_argument("--cbh-bdc-model", default=str(DEFAULT_CBH_BDC_MODEL_PATH))
    pipeline.add_argument(
        "--cbh-bdc-feature-csv",
        default=str(DEFAULT_CBH_BDC_FORMATION_FEATURE_PATH),
    )
    pipeline.add_argument(
        "--cbh-bdc-yang-root",
        default=str(SCULPTOR_ROOT / "Evaluation_Iteration" / "CBH-BDC" / "CBH-BDC-deltaML-main"),
    )
    pipeline.add_argument("--no-cbh-bdc-heating", action="store_true")
    pipeline.add_argument("--uncertainty-scale", type=float, default=1.0)
    pipeline.add_argument(
        "--screen-workers",
        type=int,
        default=1,
        help="worker processes for the MMFF screening pass (default: 1; 0 = automatic)",
    )
    pipeline.add_argument(
        "--property-workers",
        type=int,
        default=1,
        help="worker processes for property prediction (default: 1; 0 = automatic)",
    )
    pipeline.add_argument(
        "--no-property-cache",
        action="store_true",
        help="do not reuse property predictions cached in the output directory",
    )
    pipeline.add_argument(
        "--reproducibility-scope",
        choices=["all", "tier12", "tier123"],
        default="tier12",
        help="which candidates get the confirmatory reproducibility diagnostic",
    )
    pipeline.add_argument("--density-min", type=float, default=0.85)
    pipeline.add_argument("--heating-value-min", type=float, default=38.0)
    pipeline.add_argument("--heating-value-possible-min", type=float, default=37.0)
    pipeline.add_argument("--melting-point-max", type=float, default=273.15)
    pipeline.add_argument("--boiling-point-min", type=float, default=450.0)
    pipeline.add_argument("--flash-point-min", type=float, default=310.0)
    pipeline.add_argument(
        "--volumetric-heating-value-min",
        type=float,
        default=32.3,
    )
    pipeline.add_argument(
        "--mp-model",
        default=str(DEFAULT_SFS_VR_MP_MODEL_PATH),
    )
    pipeline.add_argument(
        "--mp-data",
        default=str(DEFAULT_SFS_VR_MP_DATA_PATH),
    )
    pipeline.add_argument("--skip-dft-input-export", action="store_true")
    pipeline.add_argument(
        "--output-dir",
        default=str(SCULPTOR_ROOT / "result" / "pipeline_run"),
    )

    sym_vis = subparsers.add_parser("draw-symmetric-grid")
    sym_vis.add_argument("--csv", required=True)
    sym_vis.add_argument("--output", required=True)
    sym_vis.add_argument("--smiles-column", default="canonical_smiles")
    sym_vis.add_argument("--id-column", default="usable_id")
    sym_vis.add_argument("--limit", type=int, default=80)
    sym_vis.add_argument("--mols-per-row", type=int, default=4)

    schedule_parser = subparsers.add_parser("generation-schedule")
    schedule_parser.add_argument("--round", type=int, required=True)
    schedule_parser.add_argument("--novelty-rate", type=float, default=None)
    schedule_parser.add_argument("--valid-rate", type=float, default=None)

    # Every sub-command accepts the same device switch so that a run can be
    # pinned to CPU, forced onto the GPU, or left on auto from the CLI or the
    # control panel.
    for subparser in subparsers.choices.values():
        if any("--device-mode" in action.option_strings
               for action in subparser._actions):
            continue
        subparser.add_argument(
            "--device-mode", choices=("auto", "cpu", "gpu"),
            default=argparse.SUPPRESS,
            help="auto (probe CUDA, fall back to CPU), cpu (never use CUDA) "
                 "or gpu (require CUDA; fail if it cannot be used).",
        )

    args = parser.parse_args(argv)
    device_mode = getattr(args, "device_mode", None)
    if device_mode:
        from device_policy import announce_device, describe_device, set_device_mode

        chosen = set_device_mode(device_mode)
        report = describe_device(chosen)
        # Resolved once here, at the start of the command; every later lookup
        # in this process (and in its children) reuses this decision.
        announce_device(report)
    if args.command == "generate-base":
        rng = random.Random(args.seed)
        graph = random_base_graph(rng, args.min_carbons, args.max_carbons)
        print(json.dumps(graph.as_record(), ensure_ascii=False))
        return 0

    if args.command == "count-partitions":
        print(
            json.dumps(
                {"partitions": iter_count_partitions(args.total_carbons)},
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "generate-base-by-counts":
        rng = random.Random(args.seed)
        graph = generate_base_for_counts(args.n1, args.n2, args.n3, rng=rng)
        print(json.dumps(graph.as_record(), ensure_ascii=False))
        return 0

    if args.command == "generate-semi-base-by-counts":
        rng = random.Random(args.seed)
        base = generate_base_for_counts(args.n1, args.n2, args.n3, rng=rng)
        graph = predict_semi_base_with_model(
            load_bond_model(args.checkpoint),
            base,
            threshold=args.threshold,
            temperature=args.temperature,
            sample=args.sample,
            seed=args.seed,
        ).to_graph()
        print(json.dumps(graph.as_record(), ensure_ascii=False))
        return 0

    if args.command == "generate-bases-for-total":
        rng = random.Random(args.seed)
        graphs = iter_generated_bases_for_total(
            args.total_carbons, rng=rng, per_partition=args.per_partition
        )
        print(json.dumps({"graphs": [graph.as_record() for graph in graphs]}, ensure_ascii=False))
        return 0

    if args.command == "mirror":
        edges = [_parse_edge(value) for value in args.edge]
        graph = HalfGraph(args.n1, args.n2, args.n3, tuple(edges))
        print(json.dumps({"edges_full": mirror_complete(graph)}, ensure_ascii=False))
        return 0

    if args.command == "preprocess-gdb":
        stats = process_gdb_archive(
            archive_path=args.archive,
            samples_output=args.output,
            known_smiles_output=args.known_output,
            skips_output=args.skips_output,
            members=args.member,
            max_molecules=args.max_molecules,
            max_molecules_per_member=args.max_molecules_per_member,
            max_lines_per_member=args.max_lines_per_member,
            variants_per_molecule=args.variants,
            variant_attempts=args.variant_attempts,
            min_carbons=args.min_carbons,
            use_semi_base_samples=args.semi_base_samples,
            seed=args.seed,
        )
        print(json.dumps(stats.__dict__, ensure_ascii=False))
        return 0

    if args.command == "build-known-smiles":
        stats = build_known_smiles_library(
            source_path=args.source,
            output_path=args.output,
            skips_output=args.skips_output,
            members=args.member,
            min_carbons=args.min_carbons,
            max_carbons=args.max_carbons,
        )
        print(
            json.dumps(
                {
                    "source": args.source,
                    "output": args.output,
                    **stats.__dict__,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "preprocess-ani1x":
        stats = process_ani1x_hydrocarbons(
            input_path=args.input,
            output_path=args.output,
            skips_output_path=args.skips_output,
            max_molecules=args.max_molecules,
            max_conformers=args.max_conformers,
            min_carbons=args.min_carbons,
            max_carbons=args.max_carbons,
            include_coordinates=not args.omit_coordinates,
        )
        print(json.dumps(asdict(stats), ensure_ascii=False))
        return 0

    if args.command in ("train-bond-model", "train-semi-base-model"):
        result = train_bond_model(
            csv_path=args.csv,
            checkpoint_path=args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            val_fraction=args.val_fraction,
            seed=args.seed,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            num_workers=args.num_workers,
            use_amp=not args.disable_amp,
            model_role=(
                "base_to_semi_base"
                if args.command == "train-semi-base-model"
                else "half_graph_completion"
            ),
        )
        print(result_to_json(result))
        return 0

    if args.command == "export-training-csv":
        written = export_training_csv(args.input, args.output)
        print(
            json.dumps(
                {
                    "input": args.input,
                    "output": args.output,
                    "rows": written,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "export-semi-base-training-csv":
        written = export_semi_base_training_csv(args.input, args.output)
        print(
            json.dumps(
                {
                    "input": args.input,
                    "output": args.output,
                    "rows": written,
                    "model_stage": "base_to_semi_base",
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "predict-bonds":
        edges = [_parse_edge(value) for value in args.edge]
        graph = HalfGraph(args.n1, args.n2, args.n3, tuple(edges))
        result = predict_bonds(
            checkpoint_path=args.checkpoint,
            base_graph=graph,
            threshold=args.threshold,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=args.sample,
            seed=args.seed,
        )
        print(prediction_to_json(result))
        return 0

    if args.command == "generate-and-predict":
        rng = random.Random(args.seed)
        graphs = iter_generated_bases_for_total(
            args.total_carbons,
            rng=rng,
            per_partition=args.per_partition,
        )
        if args.limit is not None:
            graphs = graphs[: args.limit]

        model = load_bond_model(args.checkpoint)
        semi_base_model = (
            load_bond_model(args.semi_base_checkpoint or args.checkpoint)
            if args.semi_base
            else None
        )
        results = []
        for graph in graphs:
            model_input = graph
            if args.semi_base:
                model_input = predict_semi_base_with_model(
                    semi_base_model,
                    graph,
                    threshold=args.threshold,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    sample=args.sample,
                    seed=rng.randrange(2**32) if args.sample else None,
                ).to_graph()
            result = predict_bonds_with_model(
                model,
                model_input,
                threshold=args.threshold,
                temperature=args.temperature,
                top_k=args.top_k,
                sample=args.sample,
                seed=rng.randrange(2**32) if args.sample else None,
            )
            results.append(
                {
                    "base": graph.as_record(),
                    "model_input": model_input.as_record(),
                    "prediction": json.loads(prediction_to_json(result)),
                }
            )
        print(
            json.dumps(
                {
                    "total_carbons": args.total_carbons,
                    "per_partition": args.per_partition,
                    "use_semi_base": args.semi_base,
                    "results": results,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "evaluate-generation":
        evaluation = evaluate_generation(
            checkpoint_path=args.checkpoint,
            known_smiles_path=args.known_smiles,
            total_carbons=args.total_carbons,
            samples=args.samples,
            threshold=args.threshold,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=args.sample,
            seed=args.seed,
            use_semi_base=args.semi_base,
            semi_base_checkpoint_path=args.semi_base_checkpoint,
            semi_base_threshold=args.semi_base_threshold,
        )
        print(evaluation_to_json(evaluation))
        return 0

    if args.command == "sweep-top-k":
        top_k_values = args.top_k if args.top_k is not None else [1, 2, 3, 4]
        if any(value < 0 for value in top_k_values):
            raise ValueError("top-k values must be non-negative")
        rows = []
        for top_k in top_k_values:
            evaluation = evaluate_generation(
                checkpoint_path=args.checkpoint,
                known_smiles_path=args.known_smiles,
                total_carbons=args.total_carbons,
                samples=args.samples,
                threshold=args.threshold,
                temperature=args.temperature,
                top_k=top_k,
                sample=args.sample,
                seed=args.seed,
                use_semi_base=args.semi_base,
            )
            rows.append(_top_k_summary_row(top_k, evaluation))

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "known_smiles": args.known_smiles,
                    "rows": rows,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "sweep-generation-grid":
        rows = []
        modes = (
            [False, True]
            if args.mode == "both"
            else [args.mode == "semi-base"]
        )
        for total_carbons in args.total_carbons:
            for seed in args.seed:
                for top_k in args.top_k:
                    for use_semi_base in modes:
                        evaluation = evaluate_generation(
                            checkpoint_path=args.checkpoint,
                            known_smiles_path=args.known_smiles,
                            total_carbons=total_carbons,
                            samples=args.samples,
                            threshold=args.threshold,
                            temperature=args.temperature,
                            top_k=top_k,
                            sample=False,
                            seed=seed,
                            use_semi_base=use_semi_base,
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "mode": "semi-base" if use_semi_base else "base",
                                **_top_k_summary_row(top_k, evaluation),
                            }
                        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "rows": len(rows),
                    "samples_per_run": args.samples,
                    "known_smiles": args.known_smiles,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "export-generation-properties":
        evaluation = evaluate_generation(
            checkpoint_path=args.checkpoint,
            known_smiles_path=args.known_smiles,
            total_carbons=args.total_carbons,
            samples=args.samples,
            threshold=args.threshold,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=False,
            seed=args.seed,
            use_semi_base=args.semi_base,
        )
        rows = []
        for index, result in enumerate(evaluation.results, start=1):
            if not result.is_valid or not result.canonical_smiles:
                continue
            descriptors = descriptors_from_smiles(result.canonical_smiles)
            rows.append(
                {
                    "index": index,
                    "is_connected": result.is_connected,
                    "is_novel": result.is_novel,
                    "n1": result.n1,
                    "n2": result.n2,
                    "n3": result.n3,
                    "selected_count": result.selected_count,
                    "candidate_count": result.candidate_count,
                    **descriptors.__dict__,
                }
            )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "rows": len(rows),
                    "valid_molecules": evaluation.valid_molecules,
                    "connected_molecules": evaluation.connected_molecules,
                    "novel_molecules": evaluation.novel_molecules,
                    "unique_valid_smiles": evaluation.unique_valid_smiles,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "screen-generation":
        evaluation = evaluate_generation(
            checkpoint_path=args.checkpoint,
            known_smiles_path=args.known_smiles,
            total_carbons=args.total_carbons,
            samples=args.samples,
            threshold=args.threshold,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=False,
            seed=args.seed,
            use_semi_base=args.semi_base,
        )
        config = ScreeningConfig(
            require_connected=not args.allow_disconnected,
            require_novel=not args.allow_known,
            min_h_to_c=args.min_h_to_c,
            max_h_to_c=args.max_h_to_c,
            min_ring_count=args.min_ring_count,
            max_ring_count=args.max_ring_count,
            max_rotatable_bonds=args.max_rotatable_bonds,
            min_fraction_sp3=args.min_fraction_sp3,
            max_sa_score=args.max_sa_score,
            target_h_to_c=args.target_h_to_c,
            target_ring_count=args.target_ring_count,
            preferred_ring_weight=args.preferred_ring_weight,
            preferred_ring_target=args.preferred_ring_target,
            require_mmff_geometry=not args.skip_mmff_geometry,
            min_mmff_stability_score=args.min_mmff_stability_score,
            max_mmff_energy_per_heavy_atom=args.max_mmff_energy_per_heavy_atom,
            high_novelty_threshold=args.novelty_threshold,
        )
        candidates, summary = screen_generation(
            evaluation,
            config,
            known_smiles_path=args.known_smiles,
        )
        rows = [candidate.__dict__ for candidate in candidates]
        high_novelty_rows = [
            candidate.__dict__
            for candidate in candidates
            if candidate.high_novelty is True
        ]

        output_path = Path(args.output)
        high_novelty_path = Path(args.high_novelty_output)
        summary_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        high_novelty_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        with high_novelty_path.open("w", encoding="utf-8", newline="") as handle:
            if high_novelty_rows:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(high_novelty_rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(high_novelty_rows)

        summary_payload = {
            **summary.__dict__,
            "total_carbons": args.total_carbons,
            "samples": args.samples,
            "top_k": args.top_k,
            "seed": args.seed,
            "use_semi_base": args.semi_base,
            "screening_config": config.__dict__,
            "output": str(output_path),
            "high_novelty_output": str(high_novelty_path),
        }
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary_payload, ensure_ascii=False))
        return 0

    if args.command == "train-property-model":
        result = train_property_model(
            csv_path=args.csv,
            smiles_column=args.smiles_column,
            target_column=args.target_column,
            checkpoint_path=args.checkpoint,
            max_rows=args.max_rows,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            val_fraction=args.val_fraction,
            seed=args.seed,
            baseline_mode=args.baseline_mode,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return 0

    if args.command == "prepare-qm9-e3nn-property":
        result = prepare_qm9_e3nn_property_csv(
            input_path=args.input,
            output_path=args.output,
            source_target_column=args.source_target_column,
            target_column=args.target_column,
            property_name=args.property_name,
            target_units=args.target_units,
            min_carbons=args.min_carbons,
            max_carbons=args.max_carbons,
            max_rows=args.max_rows,
        )
        print(result.to_json())
        return 0

    if args.command == "train-qm9-e3nn-property":
        dataset = prepare_qm9_e3nn_property_csv(
            input_path=args.input,
            output_path=args.prepared_output,
            source_target_column=args.source_target_column,
            target_column=args.target_column,
            property_name=args.property_name,
            target_units=args.target_units,
            min_carbons=args.min_carbons,
            max_carbons=args.max_carbons,
            max_rows=args.max_rows,
        )
        training = train_property_model(
            csv_path=dataset.output_path,
            smiles_column="canonical_smiles",
            target_column=dataset.target_column,
            checkpoint_path=args.checkpoint,
            max_rows=None,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            val_fraction=args.val_fraction,
            seed=args.seed,
            baseline_mode=args.baseline_mode,
        )
        print(
            json.dumps(
                {
                    "dataset": asdict(dataset),
                    "training": training.__dict__,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "prepare-qm9-cbh-bdc-features":
        try:
            from cbh_bdc_delta_ml import prepare_qm9_cbh_bdc_feature_csv

            result = prepare_qm9_cbh_bdc_feature_csv(
                input_path=args.input,
                output_path=args.output,
                yang_root=args.yang_root,
                rung=args.rung,
                smiles_column=args.smiles_column,
                target_column=args.target_column,
                min_carbons=args.min_carbons,
                max_carbons=args.max_carbons,
                max_rows=args.max_rows,
            )
            print(json.dumps(asdict(result), ensure_ascii=False))
            return 0
        except RuntimeError as exc:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "reason": str(exc),
                        "input_path": args.input,
                        "output_path": args.output,
                        "yang_root": args.yang_root,
                        "rung": args.rung,
                    },
                    ensure_ascii=False,
                )
            )
            return 2

    if args.command == "prepare-cbh-bdc-formation-targets":
        from cbh_bdc_model import prepare_cbh_bdc_formation_target_csv

        result = prepare_cbh_bdc_formation_target_csv(
            input_path=args.input,
            output_path=args.output,
            smiles_column=args.smiles_column,
            h298_column=args.h298_column,
            target_column=args.target_column,
        )
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0

    if args.command == "train-cbh-bdc-delta-ml":
        from cbh_bdc_model import train_cbh_bdc_delta_ml_model

        result = train_cbh_bdc_delta_ml_model(
            input_path=args.csv,
            model_output=args.model_output,
            prediction_output=args.prediction_output,
            metrics_output=args.metrics_output,
            yang_root=args.yang_root,
            target_column=args.target_column,
            baseline_column=args.baseline_column,
            smiles_column=args.smiles_column,
            max_rows=args.max_rows,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        print(result.to_json())
        return 0

    if args.command == "predict-cbh-bdc-formation":
        from cbh_bdc_model import (
            predict_cbh_bdc_formation_csv,
            predict_cbh_bdc_formation_enthalpy,
        )

        if bool(args.smiles) == bool(args.csv):
            raise ValueError("supply exactly one of --smiles or --csv")
        if args.smiles:
            result = predict_cbh_bdc_formation_enthalpy(
                args.smiles,
                model_path=args.model,
                feature_csv=args.feature_csv,
                yang_root=args.yang_root,
                target_column=args.target_column,
                target_kind=args.target_kind,
                water_phase=args.water_phase,
                auto_train=not args.no_auto_train,
            )
            print(json.dumps(result.as_dict(), ensure_ascii=False))
            return 0
        rows = predict_cbh_bdc_formation_csv(
            input_path=args.csv,
            output_path=args.output,
            smiles_column=args.smiles_column,
            model_path=args.model,
            feature_csv=args.feature_csv,
            yang_root=args.yang_root,
            target_column=args.target_column,
            target_kind=args.target_kind,
            water_phase=args.water_phase,
            auto_train=not args.no_auto_train,
        )
        print(
            json.dumps(
                {
                    "rows": rows,
                    "output": args.output,
                    "model": args.model,
                    "target_kind": args.target_kind,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "predict-property":
        geometry = geometry_from_smiles(args.smiles, seed=args.seed)
        value = predict_property(args.checkpoint, geometry)
        print(
            json.dumps(
                {
                    "smiles": args.smiles,
                    "canonical_smiles": geometry.canonical_smiles,
                    "prediction": value,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "score-geometry":
        result = geometry_score_from_smiles(args.smiles, seed=args.seed)
        print(json.dumps(result.as_dict(), ensure_ascii=False))
        return 0

    if args.command == "score-geometry-csv":
        rows = write_geometry_scores_csv(
            input_path=args.csv,
            output_path=args.output,
            smiles_column=args.smiles_column,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "rows": rows,
                    "output": args.output,
                    "smiles_column": args.smiles_column,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "train-geometry-scorer":
        result = train_geometry_scorer(
            csv_path=args.csv,
            smiles_column=args.smiles_column,
            target_column=args.target_column,
            checkpoint_path=args.checkpoint,
            max_rows=args.max_rows,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0

    if args.command == "predict-geometry-score":
        result = predict_geometry_score(
            checkpoint_path=args.checkpoint,
            smiles=args.smiles,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0

    if args.command == "calibrate-novelty-threshold":
        result = calibrate_novelty_threshold_from_csv(
            csv_path=args.csv,
            novelty_column=args.novelty_column,
            actual_column=args.actual_column,
            predicted_column=args.predicted_column,
            abs_error_column=args.abs_error_column,
            max_mae=args.max_mae,
            max_quantile_error=args.max_quantile_error,
            quantile=args.quantile,
            min_e3nn_fraction=args.min_e3nn_fraction,
            error_mode=args.error_mode,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = threshold_calibration_to_json(result)
        output_path.write_text(payload, encoding="utf-8")
        print(payload)
        return 0

    if args.command == "rescore-candidates":
        summary = rescore_candidate_csv(
            input_path=args.input,
            output_path=args.output,
            summary_output_path=args.summary_output,
            density_column=args.density_column,
            heating_value_column=args.heating_value_column,
            melting_point_column=args.melting_point_column,
            boiling_point_column=args.boiling_point_column,
            flash_point_column=args.flash_point_column,
            formation_enthalpy_column=args.formation_enthalpy_column,
            molar_volume_column=args.molar_volume_column,
            isp_column=args.isp_column,
            novelty_column=args.novelty_column,
            e3nn_geometry_column=args.e3nn_geometry_column,
            e3nn_symmetry_column=args.e3nn_symmetry_column,
            sa_score_column=args.sa_score_column,
            thresholds=FuelPropertyThresholds(
                density_min_g_cm3=args.density_min,
                heating_value_min_mj_kg=args.heating_value_min,
                heating_value_possible_min_mj_kg=args.heating_value_possible_min,
                melting_point_max_k=args.melting_point_max,
            ),
            score_weights=CompositeScoreWeights(
                property_quality=args.property_score_weight,
                e3nn_geometry=args.e3nn_geometry_score_weight,
                e3nn_symmetry=args.e3nn_symmetry_score_weight,
                synthetic_accessibility=0.0,
                novelty=args.novelty_score_weight,
            ),
            property_score_weights=PropertyScoreWeights(
                density=args.density_score_weight,
                heating_value=args.heating_value_score_weight,
                melting_point=args.melting_point_score_weight,
                boiling_point=args.boiling_point_score_weight,
                flash_point=args.flash_point_score_weight,
                synthetic_accessibility=args.property_sa_score_weight,
                formation_enthalpy=args.formation_enthalpy_score_weight,
                molar_volume=args.molar_volume_score_weight,
                isp=args.isp_score_weight,
            ),
        )
        print(json.dumps(asdict(summary), ensure_ascii=False))
        return 0

    if args.command == "export-candidate-inputs":
        result = export_candidate_inputs_from_csv(
            csv_path=args.csv,
            output_dir=args.output_dir,
            smiles_column=args.smiles_column,
            id_column=args.id_column,
            max_rows=args.max_rows,
            seed=args.seed,
        )
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0

    if args.command == "estimate-thesis-density":
        result = estimate_vdw_density_from_smiles(
            args.smiles,
            seed=args.seed,
            radius_scale=args.radius_scale,
            surface_points_per_atom=args.surface_points_per_atom,
            novelty_score=args.novelty_score,
            high_novelty=args.high_novelty,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False))
        return 0

    if args.command == "train-sfs-vr-melting":
        from melting_point_sfs_vr import train_sfs_vr_melting_model

        result = train_sfs_vr_melting_model(
            data_path=args.data,
            model_output_path=args.model_output,
        )
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0

    if args.command == "predict-sfs-vr-melting":
        from melting_point_sfs_vr import (
            predict_sfs_vr_melting_csv,
            predict_sfs_vr_melting_point,
        )

        if bool(args.smiles) == bool(args.csv):
            raise ValueError("supply exactly one of --smiles or --csv")
        if args.smiles:
            result = predict_sfs_vr_melting_point(
                args.smiles,
                model_path=args.model,
                data_path=args.data,
                high_novelty=args.high_novelty,
                novelty_score=args.novelty_score,
            )
            print(json.dumps(result.as_dict(), ensure_ascii=False))
            return 0
        rows = predict_sfs_vr_melting_csv(
            input_path=args.csv,
            output_path=args.output,
            smiles_column=args.smiles_column,
            model_path=args.model,
            data_path=args.data,
            high_novelty=args.high_novelty,
            novelty_column=args.novelty_column,
        )
        print(
            json.dumps(
                {
                    "rows": rows,
                    "output": args.output,
                    "model": args.model,
                    "high_novelty": args.high_novelty,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "heating-value-from-formation":
        result = mass_heating_value_from_formation_enthalpy(
            args.smiles,
            formation_enthalpy_kj_mol=args.formation_enthalpy_kj_mol,
            water_phase=args.water_phase,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False))
        return 0

    if args.command == "prepare-perturbation-campaign":
        result = prepare_perturbation_campaign(
            finalized_hits_path=args.finalized_hits,
            output_dir=args.output_dir,
            samples=args.samples,
            seed=args.seed,
            operations=tuple(args.operation)
            if args.operation
            else tuple(sorted(ALLOWED_PERTURBATION_OPERATIONS)),
            max_seeds=args.max_seeds,
            thresholds=FuelPropertyThresholds(
                density_min_g_cm3=args.density_min,
                heating_value_min_mj_kg=args.heating_value_min,
                melting_point_max_k=args.melting_point_max,
            ),
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False))
        return 0

    if args.command == "generate-perturbations":
        result = run_perturbation_generation(
            finalized_hits_path=args.finalized_hits,
            output_dir=args.output_dir,
            samples=args.samples,
            seed=args.seed,
            operations=tuple(args.operation)
            if args.operation
            else tuple(sorted(ALLOWED_PERTURBATION_OPERATIONS)),
            max_seeds=args.max_seeds,
            exclude_smiles_paths=tuple(
                path
                for path in (args.known_smiles, args.temporary_smiles)
                if path is not None
            ),
            thresholds=FuelPropertyThresholds(
                density_min_g_cm3=args.density_min,
                heating_value_min_mj_kg=args.heating_value_min,
                melting_point_max_k=args.melting_point_max,
            ),
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False))
        return 0

    if args.command == "run-perturbation-pipeline":
        if int(args.iterations or 1) > 1:
            # Iterative mode: the same campaign arguments, but the rounds keep
            # re-sampling the leads towards Tier 1 with the static iteration model.
            if args.replace_existing_products:
                raise SystemExit(
                    "--replace-existing-products cannot be combined with "
                    "--iterations > 1: every round accumulates its products."
                )
            iterative_kwargs = _perturbation_pipeline_kwargs(args)
            iterative_kwargs.pop("merge_existing_products", None)
            operator_attempts: dict[str, int] = {}
            for entry in args.operator_attempt or []:
                name, _, count = str(entry).partition("=")
                if name.strip():
                    try:
                        operator_attempts[name.strip()] = int(float(count or 0))
                    except ValueError:
                        pass
            summary = run_iterative_perturbation(
                output_dir=args.output_dir,
                iterations=args.iterations,
                pools=tuple(args.seed_pool) if args.seed_pool else ITERATION_SEED_POOLS,
                iteration_model_path=args.iteration_model,
                per_round_budget=args.per_round_budget,
                max_seeds_per_round=args.max_seeds_per_round,
                samples_per_seed_operation=args.attempts_per_seed_operation,
                attempts_per_molecule=(
                    args.attempts_per_molecule or None
                ),
                rounds_without_improvement=args.rounds_without_improvement,
                operator_attempts=operator_attempts or None,
                operator_adaptation_rate=args.operator_adaptation_rate,
                ablation_arm=args.ablation_arm,
                **iterative_kwargs,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        result = run_perturbation_pipeline(
            finalized_hits_path=args.finalized_hits,
            output_dir=args.output_dir,
            samples=args.samples,
            samples_per_seed_operation=args.samples_per_seed_operation,
            **_perturbation_pipeline_kwargs(args),
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False))
        return 0

    if args.command == "prepare-sta-gdb-dataset":
        summary = prepare_sta_gdb_dataset(
            input_paths=args.input,
            output_path=args.output,
            target_columns=tuple(args.target) if args.target else DEFAULT_STA_TARGETS,
        )
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-sta-property-model":
        metrics = train_sta_property_model(
            data_path=args.data,
            model_output_path=args.output,
            target_columns=tuple(args.target) if args.target else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            enumerations_per_molecule=args.enumerations_per_molecule,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-sta-specialist-models":
        summary = train_sta_specialist_property_models(
            data_path=args.data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
        return 0

    if args.command in {
        "train-sta-optimized-property-models",
        "train-sta-paper-property-models",
    }:
        summary = train_sta_optimized_property_models(
            data_path=args.data,
            output_dir=args.output_dir,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            batch_size=getattr(args, "batch_size", None),
            epochs=getattr(args, "epochs", None),
        )
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
        return 0

    if args.command == "predict-sta-properties":
        rows = predict_sta_property_csv(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            smiles_column=args.smiles_column,
        )
        print(
            json.dumps(
                {
                    "input": args.input,
                    "output": args.output,
                    "model": args.model,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "export-sta-weak-feedback":
        summary = export_sta_weak_feedback_csv(
            input_path=args.input,
            output_path=args.output,
            thresholds=FuelPropertyThresholds(
                density_min_g_cm3=args.density_min,
                heating_value_min_mj_kg=args.heating_value_min,
                heating_value_possible_min_mj_kg=args.heating_value_possible_min,
                melting_point_max_k=args.melting_point_max,
            ),
            novelty_threshold=args.novelty_threshold,
            base_label_weight=args.base_label_weight,
            water_phase=args.water_phase,
        )
        print(
            json.dumps(
                sta_feedback_summary_to_dict(summary),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "train-phase-fusion-models":
        summary = train_phase_fusion_models(
            data_path=args.data,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            feature_cache_path=args.feature_cache,
            targets=tuple(args.target) if args.target else DEFAULT_PHASE_TARGETS,
            random_state=args.random_state,
            max_rows=args.max_rows,
            include_e3nn_features=args.include_e3nn_features,
            workers=args.workers,
            model_family=args.model_family,
            ensemble_size=args.ensemble_size,
            mlp_epochs=args.mlp_epochs,
            mlp_batch_size=args.mlp_batch_size,
            mlp_learning_rate=args.mlp_learning_rate,
            mlp_patience=args.mlp_patience,
        )
        print(
            json.dumps(
                {
                    **summary.__dict__,
                    "metrics": {
                        target: metric.__dict__
                        for target, metric in summary.metrics.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "run-pipeline":
        summary = run_generation_pipeline(
            checkpoint_path=args.checkpoint,
            semi_base_checkpoint_path=args.semi_base_checkpoint,
            semi_base_threshold=args.semi_base_threshold,
            semi_base_temperature=args.semi_base_temperature,
            semi_base_top_k=args.semi_base_top_k,
            known_smiles_path=args.known_smiles,
            output_dir=args.output_dir,
            total_carbons=args.total_carbons,
            samples=args.samples,
            iterations=args.iterations,
            threshold=args.threshold,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            feedback_enabled=args.feedback_reward,
            feedback_strength=args.feedback_strength,
            online_property_feedback=not args.disable_online_property_feedback,
            property_feedback_warmup_iterations=args.property_feedback_warmup_iterations,
            property_feedback_warmup_weight=args.property_feedback_warmup_weight,
            property_feedback_weight=args.property_feedback_weight,
            property_reward_slope=args.property_reward_slope,
            partition_feedback_gain=args.partition_feedback_gain,
            edge_feedback_gain=args.edge_feedback_gain,
            feedback_exploration_fraction=args.feedback_exploration_fraction,
            use_semi_base=not args.base_mode,
            novelty_threshold=args.novelty_threshold,
            max_sa_score=args.max_sa_score,
            preferred_ring_weight=args.preferred_ring_weight,
            preferred_ring_target=args.preferred_ring_target,
            export_dft_inputs=not args.skip_dft_input_export,
            heating_values_csv=args.heating_values_csv,
            uncertainty_scale=args.uncertainty_scale,
            ignore_density=args.ignore_density,
            ignore_heating_value=args.ignore_heating_value,
            use_melting_point_screen=args.use_melting_point_screen,
            use_boiling_point_screen=args.use_boiling_point_screen,
            use_flash_point_screen=args.use_flash_point_screen,
            use_volumetric_heating_value_screen=(
                args.use_volumetric_heating_value_screen
            ),
            e3nn_geometry_checkpoint=args.e3nn_geometry_checkpoint,
            e3nn_property_checkpoint=args.e3nn_property_checkpoint,
            e3nn_property_name=args.e3nn_property_name,
            sta_property_model_path=args.sta_property_model,
            sta_density_model_path=args.sta_density_model,
            sta_heating_value_model_path=args.sta_heating_value_model,
            sta_melting_point_model_path=args.sta_melting_point_model,
            sta_boiling_point_model_path=args.sta_boiling_point_model,
            sta_flash_point_model_path=args.sta_flash_point_model,
            sta_formation_enthalpy_model_path=args.sta_formation_enthalpy_model,
            sta_molar_volume_model_path=args.sta_molar_volume_model,
            sta_isp_model_path=args.sta_isp_model,
            density_model=args.density_model,
            heating_value_model=args.heating_value_model,
            melting_point_model=args.melting_point_model,
            boiling_point_model=args.boiling_point_model,
            flash_point_model=args.flash_point_model,
            formation_enthalpy_model=args.formation_enthalpy_model,
            molar_volume_model=args.molar_volume_model,
            isp_model=args.isp_model,
            cbh_bdc_model_path=args.cbh_bdc_model,
            cbh_bdc_feature_csv=args.cbh_bdc_feature_csv,
            cbh_bdc_yang_root=args.cbh_bdc_yang_root,
            phase_fusion_melting_point_model_path=args.phase_fusion_melting_point_model,
            phase_fusion_boiling_point_model_path=args.phase_fusion_boiling_point_model,
            phase_fusion_flash_point_model_path=args.phase_fusion_flash_point_model,
            use_cbh_bdc_heating=(
                not args.no_cbh_bdc_heating
                and (
                    args.heating_value_model == "yang_cbh_bdc"
                    or args.formation_enthalpy_model == "yang_cbh_bdc"
                )
            ),
            objective_mode=args.objective_mode,
            screen_workers=args.screen_workers,
            property_workers=args.property_workers,
            use_property_cache=not args.no_property_cache,
            reproducibility_scope=args.reproducibility_scope,
            score_weights=CompositeScoreWeights(
                property_quality=args.property_score_weight,
                e3nn_geometry=args.e3nn_geometry_score_weight,
                e3nn_symmetry=args.e3nn_symmetry_score_weight,
                synthetic_accessibility=0.0,
                novelty=args.novelty_score_weight,
            ),
            property_score_weights=PropertyScoreWeights(
                density=args.density_score_weight,
                heating_value=args.heating_value_score_weight,
                volumetric_heating_value=(
                    args.volumetric_heating_value_score_weight
                ),
                melting_point=args.melting_point_score_weight,
                boiling_point=args.boiling_point_score_weight,
                flash_point=args.flash_point_score_weight,
                synthetic_accessibility=args.property_sa_score_weight,
                structure_validity=args.structure_validity_score_weight,
                formation_enthalpy=args.formation_enthalpy_score_weight,
                molar_volume=args.molar_volume_score_weight,
                isp=args.isp_score_weight,
            ),
            property_targets=PropertyTargetValues(
                density=args.target_density,
                heating_value=args.target_heating_value,
                volumetric_heating_value=args.target_volumetric_heating_value,
                melting_point=args.target_melting_point,
                boiling_point=args.target_boiling_point,
                flash_point=args.target_flash_point,
            ),
            objective_relative_window=args.objective_relative_window,
            missing_property_penalty=args.missing_property_penalty,
            sta_weak_feedback_enabled=not args.disable_sta_weak_feedback,
            sta_weak_feedback_weight=args.sta_weak_feedback_weight,
            thresholds=FuelPropertyThresholds(
                density_min_g_cm3=args.density_min,
                heating_value_min_mj_kg=args.heating_value_min,
                heating_value_possible_min_mj_kg=args.heating_value_possible_min,
                melting_point_max_k=args.melting_point_max,
                boiling_point_min_k=args.boiling_point_min,
                flash_point_min_k=args.flash_point_min,
                volumetric_heating_value_min_mj_l=(
                    args.volumetric_heating_value_min
                ),
            ),
            mp_model_path=args.mp_model,
            mp_data_path=args.mp_data,
            temporary_smiles_path=args.temporary_smiles,
            known_fuel_check=args.known_fuel_check,
        )
        print(json.dumps(asdict(summary), ensure_ascii=False))
        return 0

    if args.command == "build-t123-seeds":
        seeds = build_t123_perturbation_seeds(args.output_dir, args.output)
        print(json.dumps(seeds, ensure_ascii=False))
        return 0

    if args.command == "build-potential-seeds":
        seeds = build_potential_perturbation_seeds(
            args.output_dir,
            args.output,
            one_item_fraction=args.one_item_fraction,
        )
        print(json.dumps(seeds, ensure_ascii=False))
        return 0

    if args.command == "organize-run-outputs":
        organized = organize_run_outputs(args.output_dir, args.report)
        print(json.dumps(organized, ensure_ascii=False))
        return 0

    if args.command == "retier-run-outputs":
        retiered = retier_run_outputs(args.output_dir)
        print(json.dumps(retiered, ensure_ascii=False))
        return 0

    if args.command == "train-perturbation-iteration-model":
        trained = train_iteration_model_from_run(
            args.output_dir,
            model_path=args.model_out,
            metrics_path=args.metrics_out,
            model_family=args.model_family,
        )
        print(json.dumps(trained, ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare-perturbation-iteration-models":
        comparison = compare_iteration_models_from_run(
            args.output_dir,
            folds=args.folds,
            top=args.top,
            model_dir=args.model_dir,
            save_best=not args.no_save_best,
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return 0

    if args.command == "draw-symmetric-grid":
        count = draw_symmetric_grid_from_csv(
            csv_path=args.csv,
            output_path=args.output,
            smiles_column=args.smiles_column,
            id_column=args.id_column,
            limit=args.limit,
            mols_per_row=args.mols_per_row,
        )
        print(
            json.dumps(
                {
                    "output": args.output,
                    "molecules_drawn": count,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "generation-schedule":
        schedule = generation_schedule(args.round)
        if args.novelty_rate is not None or args.valid_rate is not None:
            if args.novelty_rate is None or args.valid_rate is None:
                raise ValueError("novelty-rate and valid-rate must be supplied together")
            schedule = adjust_schedule_with_feedback(
                schedule,
                novelty_rate=args.novelty_rate,
                valid_rate=args.valid_rate,
            )
        print(schedule_to_json(schedule))
        return 0

    return 2


def _parse_edge(value: str) -> tuple[int, int]:
    left, right = value.split(",", 1)
    return int(left), int(right)


def _top_k_summary_row(top_k: int, evaluation) -> dict[str, int | float | bool]:
    requested = evaluation.requested_samples
    valid = evaluation.valid_molecules
    connected = evaluation.connected_molecules
    novel = evaluation.novel_molecules
    return {
        "top_k": top_k,
        "total_carbons": evaluation.total_carbons,
        "requested_samples": requested,
        "generated_bases": evaluation.generated_bases,
        "predicted_with_candidates": evaluation.predicted_with_candidates,
        "predicted_with_edges": evaluation.predicted_with_edges,
        "valid_molecules": valid,
        "connected_molecules": connected,
        "novel_molecules": novel,
        "duplicate_known_molecules": evaluation.duplicate_known_molecules,
        "unique_valid_smiles": evaluation.unique_valid_smiles,
        "failures": evaluation.failures,
        "valid_rate": round(valid / requested, 6) if requested else 0.0,
        "connected_rate": round(connected / requested, 6) if requested else 0.0,
        "novel_rate": round(novel / requested, 6) if requested else 0.0,
        "unique_valid_rate": round(evaluation.unique_valid_smiles / requested, 6)
        if requested
        else 0.0,
        "use_semi_base": evaluation.use_semi_base,
    }


if __name__ == "__main__":
    raise SystemExit(main())
