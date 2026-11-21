from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from property_model.data import GeometryRecord, geometry_from_smiles


@dataclass(frozen=True)
class CandidateExportResult:
    input_rows: int
    exported: int
    skipped: int
    output_dir: str
    skipped_log: str


def export_candidate_inputs_from_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    smiles_column: str = "canonical_smiles",
    id_column: str = "candidate_id",
    max_rows: int | None = None,
    seed: int = 0,
) -> CandidateExportResult:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if max_rows is not None:
        rows = rows[:max_rows]

    exported = 0
    skipped = 0
    skipped_rows: list[dict[str, str]] = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        smiles = row.get(smiles_column, "").strip()
        if not smiles:
            skipped += 1
            skipped_rows.append(
                {
                    **row,
                    "skip_reason": f"missing SMILES column value: {smiles_column}",
                }
            )
            continue
        candidate_id = row.get(id_column, "").strip() or f"candidate_{index:06d}"
        try:
            geometry = geometry_from_smiles(smiles, seed=seed + index)
            _write_candidate_folder(output_path, candidate_id, smiles, geometry, row)
            exported += 1
        except Exception as exc:
            skipped += 1
            skipped_rows.append(
                {
                    **row,
                    "candidate_id": candidate_id,
                    "canonical_smiles": smiles,
                    "skip_reason": str(exc),
                }
            )

    skipped_log = output_path / "skipped_candidates.csv"
    _write_rows(skipped_log, skipped_rows)
    result = CandidateExportResult(
        input_rows=len(rows),
        exported=exported,
        skipped=skipped,
        output_dir=str(output_path),
        skipped_log=str(skipped_log),
    )
    (output_path / "export_summary.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _write_candidate_folder(
    output_dir: Path,
    candidate_id: str,
    smiles: str,
    geometry: GeometryRecord,
    row: dict[str, str],
) -> None:
    folder = output_dir / _safe_filename(candidate_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.smi").write_text(f"{smiles}\t{candidate_id}\n", encoding="utf-8")
    (folder / "input.xyz").write_text(_geometry_to_xyz(geometry), encoding="utf-8")
    metadata = {
        **row,
        "candidate_id": candidate_id,
        "canonical_smiles": geometry.canonical_smiles,
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _geometry_to_xyz(geometry: GeometryRecord) -> str:
    lines = [
        str(len(geometry.atomic_numbers)),
        geometry.canonical_smiles,
    ]
    for atomic_number, position in zip(geometry.atomic_numbers, geometry.positions):
        x, y, z = position
        lines.append(
            f"{_atomic_symbol(atomic_number)} {x:.8f} {y:.8f} {z:.8f}"
        )
    return "\n".join(lines) + "\n"


def _atomic_symbol(atomic_number: int) -> str:
    symbols = {
        1: "H",
        6: "C",
    }
    return symbols.get(atomic_number, f"X{atomic_number}")


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return safe or "candidate"


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
