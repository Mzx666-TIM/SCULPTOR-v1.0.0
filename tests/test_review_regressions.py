"""Regression coverage for issues found during the manuscript/code audit."""
import csv
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fuel_design.bootstrap import add_code_paths
add_code_paths()

def test_nonfinite_property_cannot_pass_threshold():
    from property_screening import classify_property_threshold_with_uncertainty, ThresholdDirection
    for value in [float('inf'), float('-inf'), float('nan')]:
        with pytest.raises(ValueError, match='finite'):
            classify_property_threshold_with_uncertainty('density', value, 0.01, 0.85, ThresholdDirection.MINIMUM)

def test_missing_phase_descriptor_is_not_silently_imputed(monkeypatch):
    import phase_fusion as p
    class Model:
        def predict(self, x):
            return [123.0]
    monkeypatch.setattr(p, '_load_phase_fusion_artifact', lambda _: {
        'target':'melting_point_k','feature_schema_version':4,
        'feature_columns':['absent_feature'], 'model':Model(), 'metrics':{'test_mae':1.0}})
    monkeypatch.setattr(p, 'phase_fusion_features_from_smiles', lambda _: {})
    with pytest.raises(ValueError, match='missing.*feature'):
        p.predict_phase_fusion_property('C1CCCCC1','melting_point_k',model_path='fake')

def test_unavailable_3d_features_are_not_valid_predictions(monkeypatch):
    import phase_fusion as p
    class Model:
        def predict(self, x): return [123.0]
    monkeypatch.setattr(p, '_load_phase_fusion_artifact', lambda _: {
        'target':'melting_point_k','feature_schema_version':4,
        'feature_columns':['e3nn_3d_feature_00'], 'model':Model(), 'metrics':{'test_mae':1.0}})
    monkeypatch.setattr(p, 'phase_fusion_features_from_smiles', lambda _: {})
    monkeypatch.setattr(p, 'e3nn_phase_features_from_smiles', lambda _: {'e3nn_3d_feature_00':0.,'e3nn_3d_available':0.})
    with pytest.raises(ValueError, match='3D'):
        p.predict_phase_fusion_property('C1CCCCC1','melting_point_k',model_path='fake')


def test_perturbation_seed_rows_record_the_round_that_perturbed_them(tmp_path):
    """The seed manifest must say which round perturbed a lead.

    ``iteration_round`` inherited from the product table only says when the
    molecule was produced; seeds that already existed before the iteration keep
    an empty value there, so a dedicated column is required.
    """

    from finalization import mark_perturbation_seed_rows

    rows = [
        {"canonical_smiles": "C1CCCCC1", "carbon_count": 6, "iteration_round": "2"},
        {"canonical_smiles": "C1CCCC1", "carbon_count": 5},
    ]
    marked = mark_perturbation_seed_rows([dict(row) for row in rows])
    assert all(int(row["seed_perturbation_round"]) == 0 for row in marked)

    import perturbation as pert

    table = pert._write_iteration_seed_table(tmp_path, rows, 3)
    with table.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 2
    assert all(int(row["seed_perturbation_round"]) == 3 for row in written)
    # the production-provenance column is untouched
    assert {str(row.get("iteration_round") or "") for row in written} == {"2", ""}


def test_iteration_seed_queue_skips_tier1_and_used_leads(tmp_path):
    """The iterative seed queue is Tier 2/3 + Potential 1-3 minus used leads."""

    import perturbation as pert

    def write(name, rows):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write(pert.CHANNEL_FILENAMES[pert.TIER1], [
        {"canonical_smiles": "CCCC", "jp10_beat_count": 3, "composite_ranking_score": 0.9},
    ])
    write(pert.CHANNEL_FILENAMES[pert.TIER2], [
        {"canonical_smiles": "CCCCC", "jp10_beat_count": 2, "composite_ranking_score": 0.8},
        {"canonical_smiles": "CCCCCC", "jp10_beat_count": 2, "composite_ranking_score": 0.7},
    ])
    queue = pert._iteration_seed_rows(
        tmp_path, pert.ITERATION_SEED_POOLS, already_perturbed={"CCCCC"}
    )
    assert [row["canonical_smiles"] for row in queue] == ["CCCCCC"]
    assert queue[0]["seed_source_pool"] == pert.TIER2


def test_geometry_export_marks_failure_without_zero_scores(tmp_path, monkeypatch):
    from property_model import e3nn_geometry_scorer as scorer

    source = tmp_path / "molecules.csv"
    source.write_text("canonical_smiles\nC1CCCCC1\n", encoding="utf-8")
    output = tmp_path / "geometry.csv"

    def fail(*_args, **_kwargs):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(scorer, "geometry_score_from_smiles", fail)
    assert scorer.write_geometry_scores_csv(source, output) == 1
    with output.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["geometry_status"] == "failed"
    assert row["e3nn_geometry_score"] == ""
    assert "embedding failed" in row["geometry_failure_reason"]

def test_carbon_limit_inherited_from_generation_run(tmp_path):
    from perturbation import _campaign_screening_config
    from screening import ScreeningConfig, passes_screen
    from chemistry import descriptors_from_smiles
    for c in (9,15): (tmp_path/f'C{c}').mkdir()
    config = _campaign_screening_config(tmp_path, ScreeningConfig(require_mmff_geometry=False,require_novel=False,max_sa_score=10,max_rotatable_bonds=100))
    ok,reasons=passes_screen(descriptors_from_smiles('CCCCCCCCCCC1CCCCC1'),True,True,config)
    assert not ok and 'carbon_count_above_max' in reasons
    assert (config.min_carbon_count,config.max_carbon_count)==(9,15)

def test_bond_split_keeps_all_variants_of_molecule_together(tmp_path):
    from bond_model.dataset import BondCompletionDataset
    from bond_model.train import _split_dataset
    path=tmp_path/'train.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['canonical_smiles','candidate_edges','candidate_labels'])
        writer.writeheader()
        for s in ['C1CC1','C1CCC1','C1CCCC1']:
            for _ in range(4): writer.writerow({'canonical_smiles':s,'candidate_edges':'[[0,1],[0,2]]','candidate_labels':'[1,0]'})
    dataset=BondCompletionDataset(path)
    train,val=_split_dataset(dataset,0.34,42)
    assert train and val
    assert not ({dataset.molecule_groups[i] for i in train} & {dataset.molecule_groups[i] for i in val})

def test_bond_validation_refuses_identityless_legacy_export(tmp_path):
    from bond_model.dataset import BondCompletionDataset
    from bond_model.train import _split_dataset
    path=tmp_path/'old.csv'
    path.write_text('candidate_edges,candidate_labels\n"[[0,1],[0,2]]","[1,0]"\n"[[0,1],[0,2]]","[0,1]"\n')
    with pytest.raises(ValueError,match='canonical_smiles'):
        _split_dataset(BondCompletionDataset(path),0.1,42)

def test_changed_external_values_invalidate_property_cache():
    from property_cache import config_fingerprint
    assert config_fingerprint({'heating_values':{'C':40.}},seed=0) != config_fingerprint({'heating_values':{'C':44.}},seed=0)

def test_different_model_files_with_same_name_invalidate_cache(tmp_path):
    import os
    from property_cache import config_fingerprint
    a=tmp_path/'a'/'model.bin'; b=tmp_path/'b'/'model.bin'
    a.parent.mkdir(); b.parent.mkdir(); a.write_bytes(b'aa'); b.write_bytes(b'bb')
    os.utime(a,ns=(1000000000,1000000000)); os.utime(b,ns=(1000000000,1000000000))
    assert config_fingerprint({'model':a},seed=0) != config_fingerprint({'model':b},seed=0)


def test_reported_phase_head_is_the_runtime_default():
    import phase_fusion

    assert phase_fusion.DEFAULT_PHASE_FUSION_SCHEMA == "schema2_mlp"
    assert "schema 2" in str(phase_fusion.PHASE_FUSION_SCHEMA_DIR)


def test_phase_mlp_trainer_emits_deployable_ensemble():
    from mlp_phase_head import fit_mlp_ensemble

    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    y = (2.0 * x[:, 0] - x[:, 1] + 0.1).astype(np.float32)
    model, details = fit_mlp_ensemble(
        x[:32], y[:32], x[32:], y[32:],
        seeds=(11,), hidden=(8, 4), dropout=0.0,
        epochs=3, batch_size=8, patience=2,
        target="melting_point_k", feature_set="mlp_ens2d",
    )
    prediction = model.predict(x[32:])
    assert prediction.shape == (8,)
    assert np.isfinite(prediction).all()
    assert model.q90 == pytest.approx(details["conformal90_halfwidth"])
    assert details["members"][0]["selected_epoch"] >= 1


def test_phase_feature_cache_cannot_silently_change_2d_training_to_3d():
    import pandas as pd
    from phase_fusion import _select_phase_feature_schema

    frame = pd.DataFrame({
        "descriptor": [1.0],
        "e3nn_3d_feature_00": [2.0],
        "e3nn_3d_available": [1.0],
    })
    selected = _select_phase_feature_schema(frame, include_e3nn_features=False)
    assert list(selected.columns) == ["descriptor"]


def test_bond_checkpoint_separates_deployable_best_from_resume_state(tmp_path):
    import torch
    from bond_model.dataset import NODE_FEATURE_DIM
    from bond_model.model import BondCompletionModel
    from bond_model.train import _save_checkpoint

    model = BondCompletionModel(NODE_FEATURE_DIM, hidden_dim=4, num_layers=1)
    current_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    selected_state = {name: value + 1.0 for name, value in current_state.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint = tmp_path / "bond.pt"
    _save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        completed_epochs=5,
        hidden_dim=4,
        num_layers=1,
        dataset_size=10,
        history=[],
        positive_label_fraction=0.2,
        positive_class_weight=4.0,
        recommended_threshold=0.5,
        val_best_f1=0.6,
        model_role="half_graph_completion",
        selected_state_dict=selected_state,
        selected_epoch=3,
        selected_metrics={
            "val_loss": 0.2,
            "val_accuracy": 0.8,
            "val_precision": 0.7,
            "val_recall": 0.9,
            "val_f1": 0.7875,
            "recommended_threshold": 0.73,
            "val_best_f1": 0.81,
        },
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    first_name = next(iter(current_state))
    assert torch.equal(saved["model_state_dict"][first_name], selected_state[first_name])
    assert torch.equal(saved["resume_model_state_dict"][first_name], current_state[first_name])
    assert saved["selected_epoch"] == 3
    assert saved["recommended_threshold"] == pytest.approx(0.73)
    assert saved["checkpoint_selection"] == "max_validation_calibrated_f1"


def test_perturbation_cli_propagates_worker_limits_and_cache_policy():
    from fuel_design.cli import _perturbation_pipeline_kwargs

    class Args:
        screen_workers = 1
        property_workers = 1
        no_property_cache = False
        reproducibility_scope = "tier12"
        operation = None

        def __getattr__(self, name):
            if name.endswith("_model") or name.endswith("_path") or name in {
                "known_smiles",
                "temporary_smiles",
                "perturbation_temporary_smiles",
                "heating_values_csv",
                "mp_model",
                "mp_data",
                "cbh_bdc_model",
                "cbh_bdc_feature_csv",
                "cbh_bdc_yang_root",
                "phase_fusion_melting_point_model",
                "phase_fusion_boiling_point_model",
                "phase_fusion_flash_point_model",
            }:
                return None
            return 0

    kwargs = _perturbation_pipeline_kwargs(Args())
    assert kwargs["screen_workers"] == 1
    assert kwargs["property_workers"] == 1
    assert kwargs["use_property_cache"] is True
    assert kwargs["reproducibility_scope"] == "tier12"


def test_perturbation_metrics_aggregate_all_campaigns(tmp_path):
    """Folder-level metrics must not report only the last iteration round."""

    import json
    from finalization import _perturbation_metrics

    (tmp_path / "perturbation_generation_summary.json").write_text(
        json.dumps(
            {
                "folder_attempted_samples": 20,
                "folder_generated_rows": 15,
                "folder_duplicate_rows": 4,
                "folder_failed_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "perturbation_pipeline_summary.json").write_text(
        json.dumps(
            {
                "passed_screen_rows": 1,
                "merged_product_rows": 2,
                "property_failed_rows": 0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "perturbation_screened.csv").write_text(
        "candidate_id,passes_screen\n"
        "a,true\n"
        "b,true\n"
        "c,true\n"
        "d,false\n",
        encoding="utf-8",
    )
    (tmp_path / "property_failed_candidates.csv").write_text(
        "candidate_id\n"
        "c\n",
        encoding="utf-8",
    )

    metrics = _perturbation_metrics(tmp_path)
    assert metrics["requests"] == 20
    assert metrics["valid"] == 15
    assert metrics["passed_screen"] == 3
    assert metrics["property_failed"] == 1
    assert metrics["folder_products"] == 2
    assert metrics["unique_novel"] == 2
