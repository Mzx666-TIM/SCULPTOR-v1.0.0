"""Central path registry for the SCULPTOR framework layout.

Every model / data / output / tool path used by the library is derived from
these constants, so a layout change only needs to touch this file.
"""
from __future__ import annotations

from pathlib import Path

SCULPTOR_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCULPTOR_ROOT

GENERATION = SCULPTOR_ROOT / "Generation"
PERTURBATION = SCULPTOR_ROOT / "Perturbation"
EVALUATION_ITERATION = SCULPTOR_ROOT / "Evaluation_Iteration"
DATASET_ROOT = SCULPTOR_ROOT / "dataset"
RESULT_ROOT = SCULPTOR_ROOT / "result"

# Generation pipelines
BC1_MODEL_DIR = GENERATION / "bond_completion_1" / "model"
BC1_DATA_DIR = GENERATION / "bond_completion_1" / "data"
BC2_MODEL_DIR = GENERATION / "bond_completion_2" / "model"
BC2_DATA_DIR = GENERATION / "bond_completion_2" / "data"
SCREENING_MODEL_DIR = GENERATION / "screening" / "model"
SCREENING_DATA_DIR = GENERATION / "screening" / "data"
TOTAL_REFERENCE_SMILES_PATH = SCREENING_DATA_DIR / "known_smiles_total_reference.txt"

# Evaluation / property prediction
EVAL_DIR = EVALUATION_ITERATION / "evaluation"
EVAL_MODEL_DIR = EVAL_DIR / "model"
EVAL_DATA_DIR = EVAL_DIR / "data"
STA_MODEL_DIR = EVAL_MODEL_DIR / "STA"
STA_820K_MODEL_DIR = STA_MODEL_DIR / "sta_820k_v2"
# Production melt/boil/flash fusion heads (MLP ensembles). The legacy tree
# heads were retired on 2026-09-22 and are no longer part of the runtime layout.
SCHEMA2_MLP_FUSION_DIR = EVAL_MODEL_DIR / "schema 2" / "phase_fusion_gdb820k_mlp_ens2d"
SCHEMA3_MLP_FUSION_DIR = EVAL_MODEL_DIR / "schema 3" / "phase_fusion_gdb820k_mlp_ens3d"
E3NN_MODEL_DIR = EVAL_MODEL_DIR / "e3nn"
# Iteration model of the perturbation line: predicts the property change of one
# perturbation step and steers later rounds towards Tier 1.
PERTURBATION_ITERATION_MODEL_DIR = EVAL_MODEL_DIR / "perturbation_iteration"
DEFAULT_PERTURBATION_ITERATION_MODEL_PATH = (
    PERTURBATION_ITERATION_MODEL_DIR / "perturbation_delta_iteration.joblib"
)
PERTURBATION_ITERATION_METRICS_PATH = (
    PERTURBATION_ITERATION_MODEL_DIR / "perturbation_delta_iteration.json"
)

# Alternative method pipelines (Evaluation_Iteration level)
CBH_BDC_DIR = EVALUATION_ITERATION / "CBH-BDC"
SFS_VR_DIR = EVALUATION_ITERATION / "SFS-VR"
KNOWN_FUEL_DATA_DIR = EVALUATION_ITERATION / "known_fuel_matching" / "data"
