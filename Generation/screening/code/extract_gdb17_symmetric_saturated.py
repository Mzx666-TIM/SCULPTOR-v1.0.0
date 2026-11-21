from __future__ import annotations

import sys
import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import fuel_design  # noqa: E402,F401
from chemistry import degree_of_unsaturation_from_hydrocarbon_formula  # noqa: E402
from symmetry import (
    UnsupportedSymmetryError,
    find_generation_reflection_decomposition,
    find_reflection_decomposition,
)


CSV_FIELDS = (
    "gdb17_id",
    "source_line",
    "canonical_smiles",
    "formula",
    "carbon_count",
    "hydrogen_count",
    "ring_count",
    "n1",
    "n2",
    "n3",
    "num_half_nodes",
    "half_nodes",
    "edges_half",
    "mirror_permutation",
    "passes_n1_linear_forest",
    "passes_generation_constraints",
)
ALLOWED_FAST_CHARS = frozenset("C()0123456789%")
SATURATED_FIELDS = (
    "gdb17_saturated_id",
    "source_line",
    "canonical_smiles",
    "formula",
    "carbon_count",
    "hydrogen_count",
    "ring_count",
    "edge_count",
    "carbon_edges",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract project-compatible symmetric saturated hydrocarbons."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--smiles", required=True, type=Path)
    parser.add_argument("--compatible-smiles", type=Path)
    parser.add_argument("--all-saturated-csv", type=Path)
    parser.add_argument("--all-saturated-smiles", type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--max-lines", type=int)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compatible_smiles = args.compatible_smiles or args.smiles.with_name(
        f"{args.smiles.stem}_generation_compatible{args.smiles.suffix}"
    )
    saturated_csv = args.all_saturated_csv or args.csv.with_name(
        "gdb17_saturated_hydrocarbons.csv"
    )
    saturated_smiles = args.all_saturated_smiles or args.smiles.with_name(
        "gdb17_saturated_hydrocarbons.smi"
    )
    for path in (
        args.csv,
        args.smiles,
        compatible_smiles,
        saturated_csv,
        saturated_smiles,
        args.summary,
        args.state,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    csv_partial = args.csv.with_suffix(args.csv.suffix + ".partial")
    smiles_partial = args.smiles.with_suffix(args.smiles.suffix + ".partial")
    compatible_partial = compatible_smiles.with_suffix(
        compatible_smiles.suffix + ".partial"
    )
    saturated_csv_partial = saturated_csv.with_suffix(saturated_csv.suffix + ".partial")
    saturated_smiles_partial = saturated_smiles.with_suffix(
        saturated_smiles.suffix + ".partial"
    )
    counters = {
        "scanned": 0,
        "fast_candidates": 0,
        "saturated_hydrocarbons": 0,
        "symmetric_saturated_hydrocarbons": 0,
        "generation_compatible": 0,
        "duplicates": 0,
        "parse_failures": 0,
        "unsupported_symmetry": 0,
    }
    seen: set[str] = set()
    resume_line = 0
    if args.resume and args.state.exists() and csv_partial.exists():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        resume_line = int(state.get("scanned", 0))
        for key in counters:
            counters[key] = int(state.get(key, 0))
        if smiles_partial.exists():
            with smiles_partial.open("r", encoding="utf-8") as handle:
                seen.update(line.strip() for line in handle if line.strip())
    else:
        for path in (
            csv_partial,
            smiles_partial,
            compatible_partial,
            saturated_csv_partial,
            saturated_smiles_partial,
        ):
            path.unlink(missing_ok=True)

    started = time.time()
    csv_mode = "a" if resume_line else "w"
    smiles_mode = "a" if resume_line else "w"
    with (
        gzip.open(args.input, "rt", encoding="ascii", newline="") as source,
        csv_partial.open(csv_mode, encoding="utf-8", newline="") as csv_handle,
        smiles_partial.open(smiles_mode, encoding="utf-8", newline="") as smi_handle,
        compatible_partial.open(
            smiles_mode, encoding="utf-8", newline=""
        ) as compatible_handle,
        saturated_csv_partial.open(csv_mode, encoding="utf-8", newline="") as saturated_csv_handle,
        saturated_smiles_partial.open(smiles_mode, encoding="utf-8", newline="") as saturated_smi_handle,
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        saturated_writer = csv.DictWriter(
            saturated_csv_handle, fieldnames=SATURATED_FIELDS
        )
        if not resume_line:
            writer.writeheader()
            saturated_writer.writeheader()

        for source_line, raw_line in enumerate(source, start=1):
            if source_line <= resume_line:
                continue
            if args.max_lines is not None and source_line > args.max_lines:
                break
            counters["scanned"] = source_line
            smiles = raw_line.strip().split()[0] if raw_line.strip() else ""
            if not _fast_saturated_carbon_candidate(smiles):
                _maybe_report(args, counters, started)
                continue
            counters["fast_candidates"] += 1

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None or not _is_saturated_hydrocarbon(mol):
                    _maybe_report(args, counters, started)
                    continue
                counters["saturated_hydrocarbons"] += 1
                canonical = Chem.MolToSmiles(mol, canonical=True)
                formula = rdMolDescriptors.CalcMolFormula(mol)
                carbon_edges = tuple(
                    sorted(
                        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                         max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                        for b in mol.GetBonds()
                    )
                )
                saturated_writer.writerow(
                    {
                        "gdb17_saturated_id": (
                            f"gdb17_sat_{counters['saturated_hydrocarbons']:09d}"
                        ),
                        "source_line": source_line,
                        "canonical_smiles": canonical,
                        "formula": formula,
                        "carbon_count": mol.GetNumAtoms(),
                        "hydrogen_count": sum(
                            a.GetTotalNumHs() for a in mol.GetAtoms()
                        ),
                        "ring_count": degree_of_unsaturation_from_hydrocarbon_formula(
                            formula
                        ),
                        "edge_count": len(carbon_edges),
                        "carbon_edges": json.dumps(
                            carbon_edges, separators=(",", ":")
                        ),
                    }
                )
                saturated_smi_handle.write(canonical + "\n")
                if canonical in seen:
                    counters["duplicates"] += 1
                    _maybe_report(args, counters, started)
                    continue
                edges = carbon_edges
                decomposition = find_reflection_decomposition(mol.GetNumAtoms(), edges)
                half = decomposition.to_half_graph()
            except UnsupportedSymmetryError:
                counters["unsupported_symmetry"] += 1
                _maybe_report(args, counters, started)
                continue
            except Exception:
                counters["parse_failures"] += 1
                _maybe_report(args, counters, started)
                continue

            seen.add(canonical)
            counters["symmetric_saturated_hydrocarbons"] += 1
            passes_n1 = _passes(half.validate_n1_linear_forest)
            passes_generation = _has_generation_compatible_axis(
                mol.GetNumAtoms(), edges
            )
            if passes_generation:
                counters["generation_compatible"] += 1
            writer.writerow(
                {
                    "gdb17_id": (
                        f"gdb17_g_{counters['symmetric_saturated_hydrocarbons']:09d}"
                    ),
                    "source_line": source_line,
                    "canonical_smiles": canonical,
                    "formula": formula,
                    "carbon_count": mol.GetNumAtoms(),
                    "hydrogen_count": sum(a.GetTotalNumHs() for a in mol.GetAtoms()),
                    "ring_count": degree_of_unsaturation_from_hydrocarbon_formula(
                        formula
                    ),
                    "n1": half.n1,
                    "n2": half.n2,
                    "n3": half.n3,
                    "num_half_nodes": half.half_node_count,
                    "half_nodes": json.dumps(half.representative_nodes, separators=(",", ":")),
                    "edges_half": json.dumps(half.edges, separators=(",", ":")),
                    "mirror_permutation": json.dumps(
                        decomposition.permutation, separators=(",", ":")
                    ),
                    "passes_n1_linear_forest": passes_n1,
                    "passes_generation_constraints": passes_generation,
                }
            )
            smi_handle.write(canonical + "\n")
            if passes_generation:
                compatible_handle.write(canonical + "\n")
            _maybe_report(args, counters, started)

    elapsed = time.time() - started
    summary = {
        **counters,
        "input": str(args.input),
        "csv": str(args.csv),
        "smiles": str(args.smiles),
        "generation_compatible_smiles": str(compatible_smiles),
        "all_saturated_csv": str(saturated_csv),
        "all_saturated_smiles": str(saturated_smiles),
        "elapsed_seconds": round(elapsed, 3),
        "completed": args.max_lines is None,
        "definition": (
            "connected carbon-only molecule with single bonds and a non-trivial "
            "project-compatible involutive graph automorphism"
        ),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.max_lines is None:
        os.replace(csv_partial, args.csv)
        os.replace(smiles_partial, args.smiles)
        os.replace(compatible_partial, compatible_smiles)
        os.replace(saturated_csv_partial, saturated_csv)
        os.replace(saturated_smiles_partial, saturated_smiles)
        args.state.unlink(missing_ok=True)
    else:
        _write_state(args.state, counters, elapsed)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def _fast_saturated_carbon_candidate(smiles: str) -> bool:
    return bool(smiles) and "C" in smiles and all(c in ALLOWED_FAST_CHARS for c in smiles)


def _is_saturated_hydrocarbon(mol) -> bool:
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    for atom in mol.GetAtoms():
        if (
            atom.GetAtomicNum() != 6
            or atom.GetIsAromatic()
            or atom.GetFormalCharge() != 0
            or atom.GetNumRadicalElectrons() != 0
        ):
            return False
    return all(
        bond.GetBondType() == Chem.BondType.SINGLE and not bond.GetIsAromatic()
        for bond in mol.GetBonds()
    )


def _passes(validator) -> bool:
    try:
        validator()
        return True
    except ValueError:
        return False


def _has_generation_compatible_axis(num_nodes: int, edges) -> bool:
    try:
        find_generation_reflection_decomposition(num_nodes, edges)
        return True
    except UnsupportedSymmetryError:
        return False


def _maybe_report(args, counters: dict[str, int], started: float) -> None:
    scanned = counters["scanned"]
    if not scanned or scanned % args.progress_every:
        return
    elapsed = time.time() - started
    _write_state(args.state, counters, elapsed)
    print(
        " ".join(
            (
                f"scanned={scanned}",
                f"saturated={counters['saturated_hydrocarbons']}",
                f"symmetric={counters['symmetric_saturated_hydrocarbons']}",
                f"rate={scanned / max(elapsed, 1e-9):.0f}/s",
            )
        ),
        flush=True,
    )


def _write_state(path: Path, counters: dict[str, int], elapsed: float) -> None:
    path.write_text(
        json.dumps(
            {**counters, "elapsed_seconds_current_session": round(elapsed, 3)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
