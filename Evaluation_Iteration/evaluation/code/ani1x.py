from __future__ import annotations

import sys
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from fuel_design.paths import EVAL_DATA_DIR

ELEMENT_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
Z_TO_ELEMENT = {value: key for key, value in ELEMENT_TO_Z.items()}
COVALENT_RADII_ANGSTROM = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}

@dataclass(frozen=True)
class ANI1xProcessingStats:
    scanned_molecules: int
    scanned_conformers: int
    accepted_molecules: int
    accepted_conformers: int
    skipped_molecules: int
    skipped_conformers: int
    output_path: str
    skips_output_path: str | None

@dataclass(frozen=True)
class ANI1xMoleculeRecord:
    source_path: str
    hdf5_group: str
    canonical_smiles: str
    formula: str
    carbon_count: int
    hydrogen_count: int
    atom_symbols: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]
    coordinates: Any
    properties: dict[str, Any]

def process_ani1x_hydrocarbons(
    input_path: str | Path,
    output_path: str | Path = EVAL_DATA_DIR / "g_ANI-1x.csv",
    skips_output_path: str | Path | None = None,
    max_molecules: int | None = None,
    max_conformers: int | None = None,
    min_carbons: int = 1,
    max_carbons: int | None = None,
    include_coordinates: bool = True,
) -> ANI1xProcessingStats:
    """Collect saturated hydrocarbon conformers from an ANI-1x HDF5 file.

    The processor accepts the common ANI HDF5 layout where each molecule group
    contains ``species`` and ``coordinates`` datasets, plus optional per-conformer
    property datasets such as energies.
    """

    h5py = _require_h5py()
    input_path = Path(input_path)
    output_path = Path(output_path)
    skips_path = Path(skips_output_path) if skips_output_path is not None else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if skips_path is not None:
        skips_path.parent.mkdir(parents=True, exist_ok=True)

    scanned_molecules = 0
    scanned_conformers = 0
    accepted_molecules = 0
    accepted_conformers = 0
    skipped_molecules = 0
    skipped_conformers = 0
    fieldnames = _output_fieldnames(include_coordinates)

    with (
        h5py.File(input_path, "r") as handle,
        output_path.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        skip_handle = (
            skips_path.open("w", encoding="utf-8", newline="")
            if skips_path is not None
            else None
        )
        try:
            skip_writer = (
                csv.DictWriter(skip_handle, fieldnames=_skip_fieldnames())
                if skip_handle is not None
                else None
            )
            if skip_writer:
                skip_writer.writeheader()

            for molecule in _iter_ani1x_molecules(handle, str(input_path)):
                if max_molecules is not None and accepted_molecules >= max_molecules:
                    break
                scanned_molecules += 1
                try:
                    filtered = _filter_saturated_hydrocarbon(
                        molecule,
                        min_carbons=min_carbons,
                        max_carbons=max_carbons,
                    )
                except Exception as exc:
                    skipped_molecules += 1
                    _write_skip(skip_writer, molecule, -1, type(exc).__name__, str(exc))
                    continue

                molecule_accepted = False
                conformer_count = len(filtered.coordinates)
                for conformer_index in range(conformer_count):
                    if (
                        max_conformers is not None
                        and accepted_conformers >= max_conformers
                    ):
                        break
                    scanned_conformers += 1
                    try:
                        writer.writerow(
                            _conformer_row(
                                filtered,
                                conformer_index,
                                include_coordinates=include_coordinates,
                            )
                        )
                    except Exception as exc:
                        skipped_conformers += 1
                        _write_skip(
                            skip_writer,
                            filtered,
                            conformer_index,
                            type(exc).__name__,
                            str(exc),
                        )
                        continue
                    accepted_conformers += 1
                    molecule_accepted = True

                if molecule_accepted:
                    accepted_molecules += 1
                elif max_conformers is None:
                    skipped_molecules += 1
        finally:
            if skip_handle is not None:
                skip_handle.close()

    return ANI1xProcessingStats(
        scanned_molecules=scanned_molecules,
        scanned_conformers=scanned_conformers,
        accepted_molecules=accepted_molecules,
        accepted_conformers=accepted_conformers,
        skipped_molecules=skipped_molecules,
        skipped_conformers=skipped_conformers,
        output_path=str(output_path),
        skips_output_path=str(skips_path) if skips_path is not None else None,
    )

def _iter_ani1x_molecules(root: Any, source_path: str) -> Iterable[ANI1xMoleculeRecord]:
    h5py = _require_h5py()

    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Group):
            return
        if "species" not in obj or "coordinates" not in obj:
            return
        molecules.append(_molecule_from_group(obj, source_path, name))

    molecules: list[ANI1xMoleculeRecord] = []
    root.visititems(visitor)
    if "species" in root and "coordinates" in root:
        molecules.append(_molecule_from_group(root, source_path, "/"))
    yield from molecules

def _molecule_from_group(group: Any, source_path: str, group_name: str) -> ANI1xMoleculeRecord:
    species = _decode_species(group["species"][()])
    coordinates = _as_python_array(group["coordinates"][()])
    if not coordinates or not isinstance(coordinates[0], list):
        raise ValueError("coordinates dataset is empty or malformed")
    if coordinates and coordinates[0] and not isinstance(coordinates[0][0], list):
        coordinates = [coordinates]
    properties = _properties_from_group(group, len(coordinates))
    smiles = _read_optional_scalar(group, ("smiles", "canonical_smiles"))
    if all(number in (1, 6) for number in species):
        bonds = _infer_bonds_from_coordinates(species, coordinates[0])
        canonical = _canonical_smiles_from_bonds(species, bonds, smiles)
    else:
        bonds = ()
        canonical = _canonicalize_optional_smiles(smiles) if smiles else ""
    carbon_count = sum(1 for atomic_number in species if atomic_number == 6)
    hydrogen_count = sum(1 for atomic_number in species if atomic_number == 1)
    return ANI1xMoleculeRecord(
        source_path=source_path,
        hdf5_group=group_name,
        canonical_smiles=canonical,
        formula=_formula_from_species(species),
        carbon_count=carbon_count,
        hydrogen_count=hydrogen_count,
        atom_symbols=tuple(Z_TO_ELEMENT[number] for number in species),
        bonds=bonds,
        coordinates=coordinates,
        properties=properties,
    )

def _filter_saturated_hydrocarbon(
    molecule: ANI1xMoleculeRecord,
    min_carbons: int,
    max_carbons: int | None,
) -> ANI1xMoleculeRecord:
    if molecule.carbon_count < min_carbons:
        raise ValueError(f"carbon_count {molecule.carbon_count} < {min_carbons}")
    if max_carbons is not None and molecule.carbon_count > max_carbons:
        raise ValueError(f"carbon_count {molecule.carbon_count} > {max_carbons}")
    if any(symbol not in ("C", "H") for symbol in molecule.atom_symbols):
        raise ValueError("contains non C/H atoms")
    _validate_saturated_valences(molecule.atom_symbols, molecule.bonds)
    if _has_unsaturated_bonds(molecule.canonical_smiles):
        raise ValueError("SMILES contains unsaturated or aromatic bonds")
    return molecule

def _conformer_row(
    molecule: ANI1xMoleculeRecord,
    conformer_index: int,
    include_coordinates: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "source_dataset": "ANI-1x",
        "source_file": molecule.source_path,
        "hdf5_group": molecule.hdf5_group,
        "molecule_id": _safe_id(molecule.hdf5_group),
        "conformer_id": f"{_safe_id(molecule.hdf5_group)}_conf_{conformer_index:06d}",
        "conformer_index": conformer_index,
        "canonical_smiles": molecule.canonical_smiles,
        "formula": molecule.formula,
        "carbon_count": molecule.carbon_count,
        "hydrogen_count": molecule.hydrogen_count,
        "atom_symbols_json": json.dumps(list(molecule.atom_symbols), separators=(",", ":")),
        "bonds_json": json.dumps([list(edge) for edge in molecule.bonds], separators=(",", ":")),
    }
    row["properties_json"] = json.dumps(
        {
            name: _value_at(values, conformer_index)
            for name, values in sorted(molecule.properties.items())
        },
        separators=(",", ":"),
    )
    if include_coordinates:
        row["coordinates_json"] = json.dumps(
            molecule.coordinates[conformer_index],
            separators=(",", ":"),
        )
    return row

def _decode_species(raw: Any) -> tuple[int, ...]:
    values = _as_python_array(raw)
    if values and isinstance(values[0], list):
        values = values[0]
    decoded: list[int] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            symbol = value.strip()
            if symbol not in ELEMENT_TO_Z:
                raise ValueError(f"unsupported element: {symbol}")
            decoded.append(ELEMENT_TO_Z[symbol])
        else:
            number = int(value)
            if number not in Z_TO_ELEMENT:
                raise ValueError(f"unsupported atomic number: {number}")
            decoded.append(number)
    return tuple(decoded)

def _properties_from_group(group: Any, conformer_count: int) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, dataset in group.items():
        if name in {"species", "coordinates", "smiles", "canonical_smiles"}:
            continue
        if not hasattr(dataset, "shape") or len(dataset.shape) == 0:
            continue
        if int(dataset.shape[0]) != conformer_count:
            continue
        safe_name = name.replace("/", "_").replace(".", "_")
        properties[safe_name] = _as_python_array(dataset[()])
    return properties

def _read_optional_scalar(group: Any, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name not in group:
            continue
        value = group[name][()]
        values = _as_python_array(value)
        if isinstance(values, list):
            if not values:
                return None
            value = values[0]
        else:
            value = values
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)
    return None

def _infer_bonds_from_coordinates(
    species: tuple[int, ...],
    coordinates: list[list[float]],
    tolerance: float = 0.45,
) -> tuple[tuple[int, int], ...]:
    bonds: list[tuple[int, int]] = []
    for left in range(len(species)):
        for right in range(left + 1, len(species)):
            cutoff = (
                COVALENT_RADII_ANGSTROM[species[left]]
                + COVALENT_RADII_ANGSTROM[species[right]]
                + tolerance
            )
            if _distance(coordinates[left], coordinates[right]) <= cutoff:
                bonds.append((left, right))
    return tuple(bonds)

def _validate_saturated_valences(
    atom_symbols: tuple[str, ...],
    bonds: tuple[tuple[int, int], ...],
) -> None:
    degrees = [0 for _ in atom_symbols]
    for left, right in bonds:
        degrees[left] += 1
        degrees[right] += 1
    for symbol, degree in zip(atom_symbols, degrees):
        expected = 4 if symbol == "C" else 1
        if degree != expected:
            raise ValueError(f"{symbol} atom has degree {degree}, expected {expected}")

def _canonical_smiles_from_bonds(
    species: tuple[int, ...],
    bonds: tuple[tuple[int, int], ...],
    smiles: str | None,
) -> str:
    Chem = _require_rdkit()
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES in ANI group: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True)
    rw_mol = Chem.RWMol()
    heavy_index_by_original: dict[int, int] = {}
    for index, atomic_number in enumerate(species):
        if atomic_number == 1:
            continue
        heavy_index_by_original[index] = rw_mol.AddAtom(Chem.Atom(int(atomic_number)))
    for left, right in bonds:
        if species[left] == 1 or species[right] == 1:
            continue
        rw_mol.AddBond(
            heavy_index_by_original[left],
            heavy_index_by_original[right],
            Chem.BondType.SINGLE,
        )
    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True)

def _canonicalize_optional_smiles(smiles: str | None) -> str:
    if not smiles:
        return ""
    Chem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True)

def _formula_from_species(species: tuple[int, ...]) -> str:
    counts: dict[str, int] = {}
    for atomic_number in species:
        symbol = Z_TO_ELEMENT.get(atomic_number, f"X{atomic_number}")
        counts[symbol] = counts.get(symbol, 0) + 1
    symbols = ["C", "H", "N", "O", "F"]
    return "".join(
        f"{symbol}{counts[symbol]}"
        for symbol in symbols
        if symbol in counts
    )

def _has_unsaturated_bonds(smiles: str) -> bool:
    Chem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.GetIsAromatic():
            return True
    return False

def _as_python_array(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value

def _value_at(values: Any, index: int) -> object:
    value = values[index]
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return value

def _output_fieldnames(include_coordinates: bool) -> list[str]:
    fields = [
        "source_dataset",
        "source_file",
        "hdf5_group",
        "molecule_id",
        "conformer_id",
        "conformer_index",
        "canonical_smiles",
        "formula",
        "carbon_count",
        "hydrogen_count",
        "atom_symbols_json",
        "bonds_json",
        "properties_json",
    ]
    if include_coordinates:
        fields.append("coordinates_json")
    return fields

def _skip_fieldnames() -> list[str]:
    return [
        "source_file",
        "hdf5_group",
        "conformer_index",
        "error_type",
        "error",
    ]

def _write_skip(
    writer: csv.DictWriter | None,
    molecule: ANI1xMoleculeRecord,
    conformer_index: int,
    error_type: str,
    error: str,
) -> None:
    if writer is None:
        return
    writer.writerow(
        {
            "source_file": molecule.source_path,
            "hdf5_group": molecule.hdf5_group,
            "conformer_index": conformer_index,
            "error_type": error_type,
            "error": error,
        }
    )

def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return safe.strip("_") or "ani1x_molecule"

def _distance(left: list[float], right: list[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) ** 0.5

def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency message path
        raise RuntimeError("ANI-1x processing requires h5py.") from exc
    return h5py

def _require_rdkit():
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - dependency message path
        raise RuntimeError("ANI-1x hydrocarbon filtering requires RDKit.") from exc
    return Chem
