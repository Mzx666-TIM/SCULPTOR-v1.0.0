from __future__ import annotations

import csv
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from canonical import (
    base_signature,
    canonical_base_key,
    semi_base_added_edge_code,
    semi_base_extension_key,
)
from chemistry import (
    carbon_graph_from_smiles,
    canonicalize_smiles,
    smiles_to_generation_half_graph as smiles_to_half_graph,
)
from generator import (
    base_graph_from_full_half,
    base_graph_to_semi_base_graph,
    completion_candidate_edges,
)
from graph import Edge, HalfGraph


@dataclass(frozen=True)
class GDBProcessingStats:
    scanned: int
    accepted_molecules: int
    sample_rows: int
    skipped: int


@dataclass(frozen=True)
class KnownSmilesStats:
    scanned: int
    written: int
    duplicates: int
    skipped: int


def build_known_smiles_library(
    source_path: str | Path,
    output_path: str | Path,
    skips_output: str | Path | None = None,
    members: Iterable[str] | None = None,
    min_carbons: int | None = None,
    max_carbons: int | None = None,
) -> KnownSmilesStats:
    """Write a deduplicated canonical-SMILES library from GDB `.smi` files."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skips_path = Path(skips_output) if skips_output is not None else None
    if skips_path is not None:
        skips_path.parent.mkdir(parents=True, exist_ok=True)

    scanned = 0
    skipped = 0
    seen: set[str] = set()
    duplicate_count = 0

    skip_handle = (
        skips_path.open("w", encoding="utf-8", newline="") if skips_path else None
    )
    try:
        skip_writer = (
            csv.DictWriter(skip_handle, fieldnames=_skip_fieldnames())
            if skip_handle
            else None
        )
        if skip_writer:
            skip_writer.writeheader()

        for source_file, line_number, smiles in _iter_smi_lines(source_path, members):
            scanned += 1
            try:
                canonical = canonicalize_smiles(smiles)
                n_carbons = _carbon_count(canonical)
                if min_carbons is not None and n_carbons < min_carbons:
                    raise TooFewCarbonsError(
                        f"total carbons {n_carbons} < {min_carbons}"
                    )
                if max_carbons is not None and n_carbons > max_carbons:
                    raise TooManyCarbonsError(
                        f"total carbons {n_carbons} > {max_carbons}"
                    )
            except Exception as exc:
                skipped += 1
                if skip_writer:
                    skip_writer.writerow(
                        {
                            "source_file": source_file,
                            "source_line": line_number,
                            "smiles": smiles,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                continue

            if canonical in seen:
                duplicate_count += 1
            seen.add(canonical)
    finally:
        if skip_handle:
            skip_handle.close()

    output_path.write_text("\n".join(sorted(seen)) + "\n", encoding="utf-8")
    return KnownSmilesStats(
        scanned=scanned,
        written=len(seen),
        duplicates=duplicate_count,
        skipped=skipped,
    )


def process_gdb_archive(
    archive_path: str | Path,
    samples_output: str | Path,
    known_smiles_output: str | Path,
    skips_output: str | Path,
    members: Iterable[str] | None = None,
    max_molecules: int | None = None,
    max_molecules_per_member: int | None = None,
    max_lines_per_member: int | None = None,
    variants_per_molecule: int = 3,
    seed: int = 0,
    deletion_probability_min: float = 0.3,
    deletion_probability_max: float = 0.7,
    variant_attempts: int = 50,
    min_carbons: int = 6,
    use_semi_base_samples: bool = False,
) -> GDBProcessingStats:
    """Process a GDB `.tgz` SMILES archive into bond-completion samples."""

    archive_path = Path(archive_path)
    samples_output = Path(samples_output)
    known_smiles_output = Path(known_smiles_output)
    skips_output = Path(skips_output)
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    known_smiles_output.parent.mkdir(parents=True, exist_ok=True)
    skips_output.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    scanned = 0
    accepted = 0
    sample_rows = 0
    skipped = 0
    known_smiles: list[str] = []
    seen_global_variant_keys: set[str] = set()

    with (
        tarfile.open(archive_path, "r:gz") as tar,
        samples_output.open("w", encoding="utf-8", newline="") as samples_handle,
        skips_output.open("w", encoding="utf-8", newline="") as skips_handle,
    ):
        sample_writer = csv.DictWriter(samples_handle, fieldnames=_sample_fieldnames())
        skip_writer = csv.DictWriter(skips_handle, fieldnames=_skip_fieldnames())
        sample_writer.writeheader()
        skip_writer.writeheader()

        selected_members = list(members) if members is not None else _smi_members(tar)
        for member in selected_members:
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            accepted_in_member = 0

            for line_number, raw_line in enumerate(extracted, start=1):
                if max_lines_per_member is not None and line_number > max_lines_per_member:
                    break
                smiles = raw_line.decode("utf-8").strip()
                if not smiles:
                    continue

                scanned += 1
                try:
                    canonical = canonicalize_smiles(smiles)
                    half_graph = smiles_to_half_graph(canonical)
                    if half_graph.total_carbons < min_carbons:
                        raise TooFewCarbonsError(
                            f"total carbons {half_graph.total_carbons} < {min_carbons}"
                        )
                    half_graph.validate_half_degrees()
                except Exception as exc:
                    skipped += 1
                    skip_writer.writerow(
                        {
                            "source_file": member,
                            "source_line": line_number,
                            "smiles": smiles,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    continue

                mol_id = f"gdb13_g_{accepted + 1:08d}"
                known_smiles.append(canonical)
                accepted += 1
                accepted_in_member += 1

                seen_variant_keys: set[str] = set()
                variant = 0
                attempts = 0
                while variant < variants_per_molecule and attempts < variant_attempts:
                    attempts += 1
                    deletion_probability = rng.uniform(
                        deletion_probability_min, deletion_probability_max
                    )
                    base_graph = base_graph_from_full_half(
                        half_graph,
                        rng=rng,
                        deletion_probability=deletion_probability,
                    )
                    signature = base_signature(base_graph)
                    canonical_key = canonical_base_key(base_graph)

                    try:
                        model_input_graph = (
                            base_graph_to_semi_base_graph(
                                base_graph,
                                rng=rng,
                                allowed_added_edges=set(half_graph.edges)
                                - set(base_graph.edges),
                            )
                            if use_semi_base_samples
                            else base_graph
                        )
                    except RuntimeError:
                        continue

                    added_edges = tuple(
                        edge
                        for edge in model_input_graph.edges
                        if edge not in set(base_graph.edges)
                    )
                    added_edge_code = (
                        semi_base_added_edge_code(base_graph, model_input_graph)
                        if use_semi_base_samples
                        else ""
                    )
                    extension_key = (
                        semi_base_extension_key(base_graph, model_input_graph)
                        if use_semi_base_samples
                        else ""
                    )
                    variant_key = extension_key if use_semi_base_samples else canonical_key
                    if use_semi_base_samples and variant_key in seen_global_variant_keys:
                        continue
                    if variant_key in seen_variant_keys:
                        continue
                    seen_variant_keys.add(variant_key)
                    if use_semi_base_samples:
                        seen_global_variant_keys.add(variant_key)
                    variant += 1
                    equivalence_class = len(seen_variant_keys)

                    candidate_edges = completion_candidate_edges(model_input_graph)
                    missing_edges = set(half_graph.edges) - set(model_input_graph.edges)
                    sample_rows += 1
                    sample_writer.writerow(
                        {
                            "sample_id": f"{mol_id}_v{variant:02d}",
                            "mol_id": mol_id,
                            "source_file": member,
                            "source_line": line_number,
                            "canonical_smiles": canonical,
                            "n1": half_graph.n1,
                            "n2": half_graph.n2,
                            "n3": half_graph.n3,
                            "num_half_nodes": half_graph.half_node_count,
                            "half_nodes": json.dumps(
                                list(half_graph.representative_nodes),
                                separators=(",", ":"),
                            ),
                            "main_structure_nodes": json.dumps(
                                list(half_graph.main_structure_nodes),
                                separators=(",", ":"),
                            ),
                            "edges_initial_base": _json_edges(base_graph.edges),
                            "edges_base": _json_edges(model_input_graph.edges),
                            "base_signature": signature,
                            "base_canonical_key": canonical_key,
                            "base_equivalence_class": equivalence_class,
                            "semi_base_edges_added": _json_edges(added_edges),
                            "semi_base_added_edge_code": added_edge_code,
                            "semi_base_extension_key": extension_key,
                            "edges_target": _json_edges(half_graph.edges),
                            "edges_missing": _json_edges(sorted(missing_edges)),
                            "candidate_edges": _json_edges(candidate_edges),
                            "candidate_labels": json.dumps(
                                [1 if edge in missing_edges else 0 for edge in candidate_edges],
                                separators=(",", ":"),
                            ),
                            "deletion_probability": f"{deletion_probability:.6f}",
                            "seed": seed,
                            "variant": variant,
                        }
                    )

                if max_molecules is not None and accepted >= max_molecules:
                    known_smiles_output.write_text(
                        "\n".join(known_smiles) + "\n", encoding="utf-8"
                    )
                    return GDBProcessingStats(scanned, accepted, sample_rows, skipped)
                if (
                    max_molecules_per_member is not None
                    and accepted_in_member >= max_molecules_per_member
                ):
                    break

    known_smiles_output.write_text("\n".join(known_smiles) + "\n", encoding="utf-8")
    return GDBProcessingStats(scanned, accepted, sample_rows, skipped)


def _iter_smi_lines(
    source_path: Path, members: Iterable[str] | None
) -> Iterable[tuple[str, int, str]]:
    selected_members = list(members) if members is not None else None
    if source_path.is_file() and source_path.suffix.lower() == ".csv":
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return
            smiles_column = next(
                (
                    column
                    for column in ("smiles", "canonical_smiles", "SMILES")
                    if column in reader.fieldnames
                ),
                None,
            )
            if smiles_column is None:
                raise ValueError(
                    f"CSV has no smiles or canonical_smiles column: {source_path}"
                )
            for line_number, row in enumerate(reader, start=2):
                smiles = str(row.get(smiles_column) or "").strip()
                if smiles:
                    yield source_path.name, line_number, smiles
        return

    if source_path.is_dir():
        files = (
            [source_path / member for member in selected_members]
            if selected_members is not None
            else sorted(source_path.glob("*.smi"), key=lambda path: _member_sort_key(path.name))
        )
        for smi_path in files:
            with smi_path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    smiles = raw_line.strip()
                    if smiles:
                        yield smi_path.name, line_number, smiles
        return

    with tarfile.open(source_path, "r:gz") as tar:
        selected = selected_members if selected_members is not None else _smi_members(tar)
        for member in selected:
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            for line_number, raw_line in enumerate(extracted, start=1):
                smiles = raw_line.decode("utf-8").strip()
                if smiles:
                    yield member, line_number, smiles


def _smi_members(tar: tarfile.TarFile) -> list[str]:
    return sorted(
        (member.name for member in tar.getmembers() if member.name.endswith(".smi")),
        key=_member_sort_key,
    )


def _member_sort_key(name: str) -> tuple[int, str]:
    prefix = name.split(".", 1)[0]
    return (int(prefix), name) if prefix.isdigit() else (10**9, name)


def _json_edges(edges: Iterable[Edge]) -> str:
    return json.dumps([list(edge) for edge in edges], separators=(",", ":"))


def _sample_fieldnames() -> list[str]:
    return [
        "sample_id",
        "mol_id",
        "source_file",
        "source_line",
        "canonical_smiles",
        "n1",
        "n2",
        "n3",
        "num_half_nodes",
        "half_nodes",
        "main_structure_nodes",
        "edges_initial_base",
        "edges_base",
        "base_signature",
        "base_canonical_key",
        "base_equivalence_class",
        "semi_base_edges_added",
        "semi_base_added_edge_code",
        "semi_base_extension_key",
        "edges_target",
        "edges_missing",
        "candidate_edges",
        "candidate_labels",
        "deletion_probability",
        "seed",
        "variant",
    ]


def _skip_fieldnames() -> list[str]:
    return ["source_file", "source_line", "smiles", "error_type", "error"]


class TooFewCarbonsError(ValueError):
    pass


class TooManyCarbonsError(ValueError):
    pass


def _carbon_count(smiles: str) -> int:
    return carbon_graph_from_smiles(smiles)[0]
