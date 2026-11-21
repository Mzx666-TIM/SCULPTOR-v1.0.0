from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from chemistry import canonicalize_smiles


@dataclass(frozen=True)
class GeometryRecord:
    canonical_smiles: str
    atomic_numbers: tuple[int, ...]
    positions: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class PropertyRecord:
    geometry: GeometryRecord
    target: float


def records_from_csv(
    csv_path: str | Path,
    smiles_column: str = "smiles",
    target_column: str = "target",
    max_rows: int | None = None,
    seed: int = 0,
    skip_invalid: bool = True,
) -> tuple[PropertyRecord, ...]:
    """Load property records from a CSV with SMILES and one target column."""

    records: list[PropertyRecord] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if smiles_column not in (reader.fieldnames or ()):
            raise ValueError(f"missing SMILES column: {smiles_column}")
        if target_column not in (reader.fieldnames or ()):
            raise ValueError(f"missing target column: {target_column}")
        for row in reader:
            if max_rows is not None and len(records) >= max_rows:
                break
            smiles = row[smiles_column].strip()
            if not smiles:
                continue
            try:
                records.append(
                    PropertyRecord(
                        geometry=geometry_from_smiles(smiles, seed=seed + len(records)),
                        target=float(row[target_column]),
                    )
                )
            except Exception:
                if skip_invalid:
                    continue
                raise
    return tuple(records)


@lru_cache(maxsize=8192)
def geometry_from_smiles(smiles: str, seed: int = 0) -> GeometryRecord:
    """Generate a deterministic RDKit 3D conformer for one molecule.

    The conformer is a deterministic function of ``(smiles, seed)``: the ETKDG
    embedding seed is derived from ``seed`` and the molecule is re-embedded from
    scratch on every call. Caching keyed on ``(smiles, seed)`` therefore returns
    the bit-identical same coordinates while avoiding repeated 3D conformer
    generation for molecules that are re-scored from multiple code paths (the
    generation pipeline calls geometry_from_smiles for the same molecule more
    than once per round).
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    status = -1
    for attempt in range(4):
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed) + attempt * 1009
        params.useRandomCoords = attempt > 0
        mol.RemoveAllConformers()
        status = AllChem.EmbedMolecule(mol, params)
        if status == 0:
            break
    if status != 0:
        raise ValueError(f"failed to embed SMILES: {smiles}")
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    conformer = mol.GetConformer()
    atomic_numbers: list[int] = []
    positions: list[tuple[float, float, float]] = []
    for atom in mol.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        atomic_numbers.append(int(atom.GetAtomicNum()))
        positions.append((float(pos.x), float(pos.y), float(pos.z)))

    return GeometryRecord(
        canonical_smiles=canonicalize_smiles(smiles),
        atomic_numbers=tuple(atomic_numbers),
        positions=tuple(positions),
    )


def geometry_to_tensors(record: GeometryRecord) -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor(record.atomic_numbers, dtype=torch.long)
    positions = torch.tensor(record.positions, dtype=torch.float32)
    return atomic_numbers, positions
