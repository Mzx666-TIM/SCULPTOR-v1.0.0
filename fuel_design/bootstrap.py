"""Add every SCULPTOR pipeline code folder to sys.path (flat imports)."""
from __future__ import annotations

import sys
from pathlib import Path

SCULPTOR_ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = (
    SCULPTOR_ROOT / "core/code",
    SCULPTOR_ROOT / "Evaluation_Iteration/evaluation/code",
    SCULPTOR_ROOT / "Evaluation_Iteration/feedback_iteration_loop/code",
    SCULPTOR_ROOT / "Evaluation_Iteration/known_fuel_matching/code",
    SCULPTOR_ROOT / "Generation/bond_completion_2/code",
    SCULPTOR_ROOT / "Generation/routing/code",
    SCULPTOR_ROOT / "Generation/screening/code",
    SCULPTOR_ROOT / "Generation/sym_molecule_generation/code",
    SCULPTOR_ROOT / "Perturbation/perturbation_generation",
)


def add_code_paths() -> None:
    for d in CODE_DIRS:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
