from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from chemistry import canonicalize_smiles, smiles_to_generation_half_graph
from generator import base_graph_from_full_half


@dataclass(frozen=True)
class MoleculeRecord:
    canonical_smiles: str
    n1: int
    n2: int
    n3: int
    edges_base: tuple[tuple[int, int], ...]
    edges_full: tuple[tuple[int, int], ...]
    density: float | None
    heat: float | None


def process_smiles_records(
    rows: Iterable[dict[str, str]],
    smiles_column: str = "smiles",
    density_column: str = "density",
    heat_column: str = "heat",
    deletion_seed: int = 0,
) -> list[MoleculeRecord]:
    """Create preprocessing records from CSV-like dictionaries.

    RDKit is required because SMILES parsing is chemistry-specific.
    """

    import random

    rng = random.Random(deletion_seed)
    records: list[MoleculeRecord] = []
    for row in rows:
        smiles = row[smiles_column]
        half = smiles_to_generation_half_graph(smiles)
        base = base_graph_from_full_half(
            half,
            rng=rng,
            deletion_probability=rng.uniform(0.3, 0.7),
        )
        records.append(
            MoleculeRecord(
                canonical_smiles=canonicalize_smiles(smiles),
                n1=half.n1,
                n2=half.n2,
                n3=half.n3,
                edges_base=base.edges,
                edges_full=half.edges,
                density=_optional_float(row.get(density_column)),
                heat=_optional_float(row.get(heat_column)),
            )
        )
    return records


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_molecules_csv(path: str | Path, records: Iterable[MoleculeRecord]) -> None:
    fieldnames = [
        "canonical_smiles",
        "n1",
        "n2",
        "n3",
        "edges_base",
        "edges_full",
        "density",
        "heat",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "canonical_smiles": record.canonical_smiles,
                    "n1": record.n1,
                    "n2": record.n2,
                    "n3": record.n3,
                    "edges_base": repr(record.edges_base),
                    "edges_full": repr(record.edges_full),
                    "density": "" if record.density is None else record.density,
                    "heat": "" if record.heat is None else record.heat,
                }
            )


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
