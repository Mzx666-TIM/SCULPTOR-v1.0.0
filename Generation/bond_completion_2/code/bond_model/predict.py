from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import torch

from generator import completion_candidate_edges
from graph import Edge, HalfGraph, normalize_edges
from symmetry import mirror_complete

from bond_model.dataset import graph_to_data
from bond_model.model import BondCompletionModel


@dataclass(frozen=True)
class CandidatePrediction:
    edge: Edge
    probability: float
    selected: bool


@dataclass(frozen=True)
class BondPredictionResult:
    n1: int
    n2: int
    n3: int
    edges_base: tuple[Edge, ...]
    predicted_edges: tuple[Edge, ...]
    edges_half_predicted: tuple[Edge, ...]
    edges_full_predicted: tuple[Edge, ...] | None
    candidates: tuple[CandidatePrediction, ...]


def predict_bonds(
    checkpoint_path: str | Path,
    base_graph: HalfGraph,
    threshold: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    sample: bool = False,
    seed: int | None = None,
    edge_feedback_biases: Mapping[str, float] | None = None,
) -> BondPredictionResult:
    model = load_bond_model(checkpoint_path)
    return predict_bonds_with_model(
        model,
        base_graph,
        threshold=threshold,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        seed=seed,
        edge_feedback_biases=edge_feedback_biases,
    )


def predict_bonds_with_model(
    model: BondCompletionModel,
    base_graph: HalfGraph,
    threshold: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    sample: bool = False,
    seed: int | None = None,
    edge_feedback_biases: Mapping[str, float] | None = None,
) -> BondPredictionResult:
    candidate_edges = completion_candidate_edges(base_graph)
    if not candidate_edges:
        return _empty_result(base_graph)

    model.eval()
    data = graph_to_data(base_graph, candidate_edges)

    with torch.no_grad():
        logits = model(data)
        if edge_feedback_biases:
            feedback_bias = torch.tensor(
                [
                    float(
                        edge_feedback_biases.get(
                            edge_feedback_key(base_graph, edge),
                            0.0,
                        )
                    )
                    for edge in candidate_edges
                ],
                dtype=logits.dtype,
                device=logits.device,
            )
            logits = logits + feedback_bias
        probabilities = torch.sigmoid(logits / max(temperature, 1e-6)).cpu().tolist()

    selected = _select_edges(
        candidate_edges,
        probabilities,
        threshold=threshold,
        top_k=top_k,
        sample=sample,
        seed=seed,
    )
    edges_half = normalize_edges(set(base_graph.edges) | set(selected))
    try:
        edges_full = mirror_complete(HalfGraph(base_graph.n1, base_graph.n2, base_graph.n3, edges_half))
    except ValueError:
        edges_full = None

    candidates = tuple(
        CandidatePrediction(
            edge=edge,
            probability=round(float(probability), 6),
            selected=edge in set(selected),
        )
        for edge, probability in zip(candidate_edges, probabilities)
    )
    return BondPredictionResult(
        n1=base_graph.n1,
        n2=base_graph.n2,
        n3=base_graph.n3,
        edges_base=base_graph.edges,
        predicted_edges=selected,
        edges_half_predicted=edges_half,
        edges_full_predicted=edges_full,
        candidates=candidates,
    )


@lru_cache(maxsize=4)
def load_bond_model(checkpoint_path: str | Path) -> BondCompletionModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BondCompletionModel(
        node_feature_dim=int(checkpoint["node_feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def prediction_to_json(result: BondPredictionResult) -> str:
    payload = asdict(result)
    return json.dumps(payload, ensure_ascii=False)


def _select_edges(
    candidate_edges: tuple[Edge, ...],
    probabilities: list[float],
    threshold: float,
    top_k: int | None,
    sample: bool,
    seed: int | None,
) -> tuple[Edge, ...]:
    if sample:
        rng = random.Random(seed)
        allowed_edges = set(candidate_edges)
        if top_k is not None:
            ranked = sorted(
                zip(candidate_edges, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )
            allowed_edges = {edge for edge, _ in ranked[:top_k]}
        selected = [
            edge
            for edge, probability in zip(candidate_edges, probabilities)
            if (
                edge in allowed_edges
                and probability >= threshold
                and rng.random() < probability
            )
        ]
    else:
        selected = [
            edge
            for edge, probability in zip(candidate_edges, probabilities)
            if probability >= threshold
        ]

    if top_k is not None and not sample:
        ranked = sorted(
            zip(candidate_edges, probabilities), key=lambda item: item[1], reverse=True
        )
        top_edges = {edge for edge, _ in ranked[:top_k]}
        selected = [edge for edge in candidate_edges if edge in top_edges]

    return normalize_edges(selected)


def edge_feedback_key(base_graph: HalfGraph, edge: Edge) -> str:
    """Describe a candidate bond without depending on absolute atom indexes."""
    u, v = edge
    degrees = base_graph.degrees()
    endpoint_features = sorted(
        (
            f"{base_graph.kind(u).value}:d{degrees[u]}",
            f"{base_graph.kind(v).value}:d{degrees[v]}",
        )
    )
    distance = _shortest_path_length(base_graph, u, v)
    distance_label = "x" if distance is None else str(min(distance, 8))
    return f"{endpoint_features[0]}|{endpoint_features[1]}|path:{distance_label}"


def _shortest_path_length(
    graph: HalfGraph,
    start: int,
    target: int,
) -> int | None:
    adjacency = {node: set() for node in graph.representative_nodes}
    for u, v in graph.edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    frontier = [(start, 0)]
    seen = {start}
    for node, distance in frontier:
        if node == target:
            return distance
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append((neighbor, distance + 1))
    return None


def _empty_result(base_graph: HalfGraph) -> BondPredictionResult:
    try:
        edges_full = mirror_complete(base_graph)
    except ValueError:
        edges_full = None
    return BondPredictionResult(
        n1=base_graph.n1,
        n2=base_graph.n2,
        n3=base_graph.n3,
        edges_base=base_graph.edges,
        predicted_edges=(),
        edges_half_predicted=base_graph.edges,
        edges_full_predicted=edges_full,
        candidates=(),
    )
