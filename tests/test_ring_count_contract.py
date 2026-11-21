from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fuel_design.bootstrap import add_code_paths

add_code_paths()


def test_formula_ring_count_for_saturated_hydrocarbons():
    from chemistry import degree_of_unsaturation_from_hydrocarbon_formula

    assert degree_of_unsaturation_from_hydrocarbon_formula("CH4") == 0
    assert degree_of_unsaturation_from_hydrocarbon_formula("C6H12") == 1
    assert degree_of_unsaturation_from_hydrocarbon_formula("C10H16") == 3
    assert degree_of_unsaturation_from_hydrocarbon_formula("C10H20") == 1
    with pytest.raises(ValueError, match="neutral closed-shell"):
        degree_of_unsaturation_from_hydrocarbon_formula("C6H11")
    with pytest.raises(ValueError, match="hydrocarbon"):
        degree_of_unsaturation_from_hydrocarbon_formula("CH4O")


def test_descriptors_separate_cycle_rank_from_sssr_count():
    from chemistry import descriptors_from_smiles

    simple = descriptors_from_smiles("CC1CCCCCCCC1")
    assert simple.formula == "C10H20"
    assert simple.ring_count == 1
    assert simple.rdkit_ring_count == 1
    assert simple.strain_ring_score == 0

    cyclohexane = descriptors_from_smiles("C1CCCCC1")
    assert cyclohexane.ring_count == 1
    assert cyclohexane.six_member_rings == 1
    assert cyclohexane.strain_ring_score == 0
    assert cyclohexane.preferred_ring_score == 2

    cage = descriptors_from_smiles("CCCC12C3C(C31)C21CC1")
    assert cage.formula == "C10H14"
    assert cage.ring_count == 4
    assert cage.rdkit_ring_count == 5


def test_generation_screen_uses_full_molecule_ring_count():
    from evaluation import EvaluatedGeneration, GenerationEvaluation
    from screening import ScreeningConfig, screen_generation

    smiles = "CC1CCCCCCCC1"
    half_edges = ((0, 1), (1, 4), (2, 6), (4, 5), (5, 6))
    full_edges = (
        (0, 1),
        (1, 4),
        (1, 7),
        (2, 3),
        (2, 6),
        (3, 9),
        (4, 5),
        (5, 6),
        (7, 8),
        (8, 9),
    )
    result = EvaluatedGeneration(
        generation_iteration=1,
        n1=2,
        n2=2,
        n3=6,
        edges_initial_base=(),
        edges_base=half_edges,
        semi_base_edges_added=(),
        predicted_edges=(),
        edges_half_predicted=half_edges,
        edges_full_predicted=full_edges,
        candidate_count=0,
        selected_count=0,
        is_valid=True,
        is_connected=True,
        is_novel=True,
        canonical_smiles=smiles,
        error=None,
    )
    evaluation = GenerationEvaluation(
        total_carbons=10,
        requested_samples=1,
        generated_bases=1,
        predicted_with_candidates=1,
        predicted_with_edges=1,
        valid_molecules=1,
        connected_molecules=1,
        novel_molecules=1,
        duplicate_known_molecules=0,
        unique_valid_smiles=1,
        failures=0,
        use_semi_base=False,
        results=(result,),
    )
    config = ScreeningConfig(
        require_novel=False,
        require_mmff_geometry=False,
        max_sa_score=10.0,
        max_rotatable_bonds=10,
    )
    screened, summary = screen_generation(
        evaluation,
        config=config,
        known_smiles_path=None,
    )
    candidate = screened[0]
    assert summary.ring_count_definition == (
        "formula_cycle_rank_full_molecule_v1"
    )
    assert candidate.ring_count == 1
    assert candidate.ring_count_definition == "formula_cycle_rank_full_molecule_v1"
    assert candidate.half_projected_ring_count == 0
    assert candidate.six_member_rings == 0
    assert candidate.strain_ring_score == 0
    assert candidate.preferred_ring_score == 0
    assert candidate.passes_screen is True


def test_iteration_dataset_repairs_stored_ring_count():
    from perturbation_iteration import _with_current_ring_count

    cache: dict[str, int] = {}
    repaired = _with_current_ring_count(
        {"canonical_smiles": "CC1CCCCCCCC1", "ring_count": "0"},
        cache,
    )
    assert repaired["ring_count"] == 1
    assert repaired["ring_count_definition"] == "formula_cycle_rank_full_molecule_v1"
    assert cache == {"CC1CCCCCCCC1": 1}


def test_parent_index_repairs_legacy_ring_count(tmp_path):
    import csv

    from perturbation_iteration import _parent_index

    with (tmp_path / "perturbation_property_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["canonical_smiles", "ring_count"],
        )
        writer.writeheader()
        writer.writerow(
            {"canonical_smiles": "CC1CCCCCCCC1", "ring_count": "0"}
        )
    indexed = _parent_index(tmp_path)["CC1CCCCCCCC1"]
    assert indexed["ring_count"] == 1
    assert indexed["ring_count_definition"] == (
        "formula_cycle_rank_full_molecule_v1"
    )


def test_iteration_ring_preference_matches_screening_descriptors():
    from chemistry import descriptors_from_smiles
    from perturbation_iteration import ring_preference

    smiles = "CC1C2CCC1CC2"
    descriptors = descriptors_from_smiles(smiles)
    _reward, _penalty, strained, preferred = ring_preference(smiles)
    assert strained == descriptors.strain_ring_score
    assert preferred == descriptors.preferred_ring_score


def test_iteration_model_requires_current_ring_contract(tmp_path):
    import joblib

    from perturbation_iteration import (
        RING_COUNT_DEFINITION,
        iteration_model_contract_is_current,
        load_iteration_model,
    )

    model_path = tmp_path / "iteration.joblib"
    joblib.dump({"model": {}}, model_path)
    assert iteration_model_contract_is_current(model_path) is False
    with pytest.raises(ValueError, match="full-molecule ring-count"):
        load_iteration_model(model_path)

    joblib.dump(
        {"model": {}, "ring_count_definition": RING_COUNT_DEFINITION},
        model_path,
    )
    assert iteration_model_contract_is_current(model_path) is True
