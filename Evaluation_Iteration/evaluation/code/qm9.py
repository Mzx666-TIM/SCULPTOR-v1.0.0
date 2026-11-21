from __future__ import annotations

import sys
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from chemistry import canonicalize_smiles
from fuel_design.paths import EVAL_DATA_DIR

DEFAULT_QM9_HYDROCARBON_PATH = EVAL_DATA_DIR / "g_QM9.csv"
DEFAULT_QM9_E3NN_PROPERTY_PATH = EVAL_DATA_DIR / "qm9_e3nn_h298_property.csv"

@dataclass(frozen=True)
class QM9E3NNPropertyDatasetResult:
    input_path: str
    output_path: str
    rows: int
    skipped_rows: int
    source_target_column: str
    target_column: str
    property_name: str
    target_units: str
    min_carbons: int | None
    max_carbons: int | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

def prepare_qm9_e3nn_property_csv(
    input_path: str | Path = DEFAULT_QM9_HYDROCARBON_PATH,
    output_path: str | Path = DEFAULT_QM9_E3NN_PROPERTY_PATH,
    source_target_column: str = "h298",
    target_column: str = "target",
    property_name: str = "qm9_h298_hartree",
    target_units: str = "hartree",
    min_carbons: int | None = None,
    max_carbons: int | None = None,
    max_rows: int | None = None,
) -> QM9E3NNPropertyDatasetResult:
    """Prepare a compact QM9 hydrocarbon table for the e3nn property regressor."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    skipped_rows = 0
    fieldnames = [
        "canonical_smiles",
        "smiles",
        "formula",
        "carbon_count",
        "hydrogen_count",
        target_column,
        property_name,
        "qm9_source_target_column",
        "qm9_target_units",
        "qm9_label_note",
    ]

    with (
        input_path.open("r", encoding="utf-8", newline="") as input_handle,
        output_path.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        reader = csv.DictReader(input_handle)
        if source_target_column not in (reader.fieldnames or ()):
            raise ValueError(f"missing QM9 target column: {source_target_column}")
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if max_rows is not None and rows_written >= max_rows:
                break
            try:
                carbon_count = _optional_int(row.get("carbon_count"))
                if min_carbons is not None and carbon_count < min_carbons:
                    continue
                if max_carbons is not None and carbon_count > max_carbons:
                    continue
                target_value = float(str(row.get(source_target_column, "")).strip())
                smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
                canonical = canonicalize_smiles(smiles)
                writer.writerow(
                    {
                        "canonical_smiles": canonical,
                        "smiles": str(row.get("smiles") or canonical).strip(),
                        "formula": str(row.get("formula") or "").strip(),
                        "carbon_count": carbon_count,
                        "hydrogen_count": _optional_int(row.get("hydrogen_count")),
                        target_column: target_value,
                        property_name: target_value,
                        "qm9_source_target_column": source_target_column,
                        "qm9_target_units": target_units,
                        "qm9_label_note": (
                            "QM9 thermochemical target used as an e3nn proxy label; "
                            "not a final experimental standard formation enthalpy."
                        ),
                    }
                )
                rows_written += 1
            except Exception:
                skipped_rows += 1
                continue

    return QM9E3NNPropertyDatasetResult(
        input_path=str(input_path),
        output_path=str(output_path),
        rows=rows_written,
        skipped_rows=skipped_rows,
        source_target_column=source_target_column,
        target_column=target_column,
        property_name=property_name,
        target_units=target_units,
        min_carbons=min_carbons,
        max_carbons=max_carbons,
    )

def _optional_int(value: object) -> int:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError("missing integer value")
    return int(float(text))
