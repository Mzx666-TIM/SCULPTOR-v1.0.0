# -*- coding: utf-8 -*-
"""Classic high-energy fuel library: identifies known fuels in generation and
perturbation pipelines.

Matching: exact canonical SMILES. Library file:
`SCULPTOR/Evaluation_Iteration/known_fuel_matching/data/known_fuel_library.txt`（name, SMILES, canonical_smiles）。
"""

from __future__ import annotations

import csv
import sys


from functools import lru_cache
from pathlib import Path

from chemistry import canonicalize_smiles

from fuel_design.paths import KNOWN_FUEL_DATA_DIR

DEFAULT_KNOWN_FUEL_LIBRARY = KNOWN_FUEL_DATA_DIR / "known_fuel_library.txt"

@lru_cache(maxsize=1)
def load_known_fuel_library(
    path: str | Path = DEFAULT_KNOWN_FUEL_LIBRARY,
) -> dict[str, list[str]]:
    """Return {canonical_smiles: [names]} for every known fuel."""
    table: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name, _raw, canonical = parts[0], parts[1], parts[2].strip()
            table.setdefault(canonical, []).append(name)
    return table

def match_known_fuel(
    smiles: str,
    path: str | Path = DEFAULT_KNOWN_FUEL_LIBRARY,
) -> dict[str, object] | None:
    """Match one SMILES against the known fuel library.

    Returns None if not a known fuel; otherwise a dict with
    `known_fuel`, `known_fuel_names`, `known_fuel_smiles`.
    """
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        return None
    names = load_known_fuel_library(path).get(canonical)
    if not names:
        return None
    return {
        "known_fuel": True,
        "known_fuel_names": names,
        "known_fuel_smiles": canonical,
    }

def annotate_known_fuel(
    rows: list[dict[str, object]],
    smiles_key: str = "canonical_smiles",
    path: str | Path = DEFAULT_KNOWN_FUEL_LIBRARY,
) -> list[dict[str, object]]:
    """Add `known_fuel` / `known_fuel_names` columns to a list of row dicts."""
    table = load_known_fuel_library(path)
    out: list[dict[str, object]] = []
    for row in rows:
        smi = str(row.get(smiles_key) or "").strip()
        canonical = canonicalize_smiles(smi) if smi else ""
        names = table.get(canonical) if canonical else None
        row["known_fuel"] = bool(names)
        row["known_fuel_names"] = ";".join(names) if names else ""
        out.append(row)
    return out


def build_known_fuel_hit_table(
    rows: list[dict[str, object]],
    output_path: str | Path,
    smiles_key: str = "canonical_smiles",
    path: str | Path = DEFAULT_KNOWN_FUEL_LIBRARY,
) -> dict[str, object]:
    """Scan a batch of molecules against the classic fuel library and write a hit table.

    Every molecule in ``rows`` (regardless of screening outcome) is compared with
    the classic fuel library by canonical SMILES. The output CSV lists each fuel
    that was hit: fuel name, canonical SMILES, hit count, and matched molecules.
    """
    table = load_known_fuel_library(path)
    hits: dict[str, dict[str, object]] = {}
    for row in rows:
        smi = str(row.get(smiles_key) or "").strip()
        canonical = canonicalize_smiles(smi) if smi else ""
        names = table.get(canonical) if canonical else None
        if not names:
            continue
        entry = hits.setdefault(
            canonical,
            {
                "fuel_name": ";".join(names),
                "fuel_smiles": canonical,
                "hit_count": 0,
                "hit_molecules": [],
            },
        )
        entry["hit_count"] = int(entry["hit_count"]) + 1
        if smi not in entry["hit_molecules"]:
            entry["hit_molecules"].append(smi)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fuel_name", "fuel_smiles", "hit_count", "hit_molecules"])
        for canonical in sorted(hits):
            entry = hits[canonical]
            writer.writerow(
                [
                    entry["fuel_name"],
                    entry["fuel_smiles"],
                    entry["hit_count"],
                    ";".join(entry["hit_molecules"]),
                ]
            )
    return {
        "output_path": str(out),
        "total_fuels_hit": len(hits),
        "total_hits": sum(int(e["hit_count"]) for e in hits.values()),
    }
