from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from graph import Edge, HalfGraph, NodeKind
from generator import semi_base_candidate_edges


NODE_FEATURE_DIM = 10
TRAINING_FIELDNAMES = [
    "canonical_smiles",
    "n1",
    "n2",
    "n3",
    "edges_base",
    "candidate_edges",
    "candidate_labels",
]


class BondCompletionDataset:
    """Lazily indexed CSV dataset for candidate-edge bond completion."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self._handle = None
        self.fieldnames: list[str] = []
        self.offsets: list[int] = []
        self.positive_labels = 0
        self.negative_labels = 0
        self.molecule_groups: list[str] = []
        self.label_counts: list[tuple[int, int]] = []
        with self.csv_path.open("rb") as handle:
            header = handle.readline().decode("utf-8-sig").rstrip("\r\n")
            self.fieldnames = next(csv.reader([header]))
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                if not raw_line.strip():
                    continue
                row = self._parse_raw_line(raw_line)
                if _has_training_labels(row):
                    self.offsets.append(offset)
                    labels = json.loads(row["candidate_labels"])
                    positives = sum(1 for value in labels if float(value) >= 0.5)
                    self.positive_labels += positives
                    self.negative_labels += len(labels) - positives
                    self.molecule_groups.append(str(row.get("canonical_smiles") or "").strip())
                    self.label_counts.append((positives, len(labels) - positives))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Data:
        if self._handle is None:
            self._handle = self.csv_path.open("rb")
        self._handle.seek(self.offsets[index])
        return bond_sample_to_data(self._parse_raw_line(self._handle.readline()))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()

    def _parse_raw_line(self, raw_line: bytes) -> dict[str, str]:
        values = next(csv.reader([raw_line.decode("utf-8").rstrip("\r\n")]))
        return dict(zip(self.fieldnames, values))


def export_training_csv(source_csv: str | Path, output_csv: str | Path) -> int:
    """Write the compact columns required by `BondCompletionDataset`."""

    source_path = Path(source_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with (
        source_path.open("r", encoding="utf-8", newline="") as source_handle,
        output_path.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        reader = csv.DictReader(source_handle)
        writer = csv.DictWriter(output_handle, fieldnames=TRAINING_FIELDNAMES)
        writer.writeheader()
        for row in reader:
            if not _has_training_labels(row):
                continue
            writer.writerow({field: row.get(field, "") for field in TRAINING_FIELDNAMES})
            written += 1
    return written


def export_semi_base_training_csv(
    source_csv: str | Path,
    output_csv: str | Path,
) -> int:
    """Build compact training rows for the base-to-semi-base GNN."""

    source_path = Path(source_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (
        source_path.open("r", encoding="utf-8", newline="") as source_handle,
        output_path.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        reader = csv.DictReader(source_handle)
        writer = csv.DictWriter(output_handle, fieldnames=TRAINING_FIELDNAMES)
        writer.writeheader()
        for row in reader:
            try:
                graph = HalfGraph(
                    int(row["n1"]),
                    int(row["n2"]),
                    int(row["n3"]),
                    _parse_edges(row["edges_initial_base"]),
                )
                graph.validate_base_graph()
                candidates = semi_base_candidate_edges(graph)
                if not candidates:
                    continue
                selected_edges = set(
                    _parse_edges(row.get("semi_base_edges_added", "[]"))
                )
                if not selected_edges:
                    target_edges = set(_parse_edges(row.get("edges_target", "[]")))
                    selected_edges = set(candidates) & target_edges
                labels = [1 if edge in selected_edges else 0 for edge in candidates]
                if not any(labels):
                    continue
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            writer.writerow(
                {
                    "canonical_smiles": row.get("canonical_smiles", ""),
                    "n1": graph.n1,
                    "n2": graph.n2,
                    "n3": graph.n3,
                    "edges_base": json.dumps(graph.edges, separators=(",", ":")),
                    "candidate_edges": json.dumps(candidates, separators=(",", ":")),
                    "candidate_labels": json.dumps(labels, separators=(",", ":")),
                }
            )
            written += 1
    return written


def bond_sample_to_data(row: dict[str, Any]) -> Data:
    n1 = int(row["n1"])
    n2 = int(row["n2"])
    n3 = int(row["n3"])
    edges_base = _parse_edges(row["edges_base"])
    candidate_edges = _parse_edges(row["candidate_edges"])
    labels = [float(value) for value in json.loads(row["candidate_labels"])]

    graph = HalfGraph(n1, n2, n3, tuple(edges_base))
    return graph_to_data(graph, candidate_edges, labels)


def graph_to_data(
    graph: HalfGraph,
    candidate_edges: tuple[Edge, ...],
    labels: list[float] | None = None,
) -> Data:
    node_ids = list(graph.representative_nodes)
    node_to_local = {node_id: idx for idx, node_id in enumerate(node_ids)}

    x = torch.tensor(
        [_node_features(graph, node_id) for node_id in node_ids], dtype=torch.float32
    )
    edge_index = _edge_index(graph.edges, node_to_local)
    candidate_edge_index = _candidate_edge_index(candidate_edges, node_to_local)
    y = (
        torch.tensor(labels, dtype=torch.float32)
        if labels is not None
        else torch.empty((candidate_edge_index.shape[1],), dtype=torch.float32)
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        candidate_edge_index=candidate_edge_index,
        y=y,
        num_nodes=len(node_ids),
    )
    data.n1 = graph.n1
    data.n2 = graph.n2
    data.n3 = graph.n3
    data.node_ids = torch.tensor(node_ids, dtype=torch.long)
    return data


def _node_features(graph: HalfGraph, node: int) -> list[float]:
    kind = graph.kind(node)
    degrees = graph.degrees()
    total = max(graph.total_carbons, 1)
    half_degree = degrees[node]
    if kind is NodeKind.FIXED:
        used_degree = graph.n1_full_degree_from_half(node)
        limit = 4
    elif kind is NodeKind.MIRROR_BONDED:
        used_degree = half_degree
        limit = 3
    else:
        used_degree = half_degree
        limit = 4

    return [
        1.0 if kind is NodeKind.FIXED else 0.0,
        1.0 if kind is NodeKind.MIRROR_BONDED else 0.0,
        1.0 if kind is NodeKind.MOVING else 0.0,
        1.0 if node in graph.main_structure_nodes else 0.0,
        half_degree / 4.0,
        max(limit - used_degree, 0) / 4.0,
        node / total,
        graph.n1 / total,
        graph.n2 / total,
        graph.n3 / total,
    ]


def _edge_index(edges: tuple[Edge, ...], node_to_local: dict[int, int]) -> torch.Tensor:
    directed_edges: list[tuple[int, int]] = []
    for u, v in edges:
        lu = node_to_local[u]
        lv = node_to_local[v]
        directed_edges.append((lu, lv))
        directed_edges.append((lv, lu))
    if not directed_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(directed_edges, dtype=torch.long).t().contiguous()


def _candidate_edge_index(
    candidate_edges: tuple[Edge, ...], node_to_local: dict[int, int]
) -> torch.Tensor:
    if not candidate_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(
        [[node_to_local[u], node_to_local[v]] for u, v in candidate_edges],
        dtype=torch.long,
    ).t().contiguous()


def _parse_edges(value: str) -> tuple[Edge, ...]:
    parsed = json.loads(value)
    return tuple((int(u), int(v)) for u, v in parsed)


def _has_training_labels(row: dict[str, Any]) -> bool:
    try:
        candidate_edges = json.loads(row["candidate_edges"])
        candidate_labels = json.loads(row["candidate_labels"])
    except (KeyError, json.JSONDecodeError):
        return False
    return bool(candidate_edges) and bool(candidate_labels)
