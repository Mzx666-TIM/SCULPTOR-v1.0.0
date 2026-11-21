from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from generator import semi_base_candidate_edges
from graph import Edge, HalfGraph, normalize_edges

from bond_model.dataset import graph_to_data
from bond_model.model import BondCompletionModel
from bond_model.predict import edge_feedback_key, load_bond_model


@dataclass(frozen=True)
class SemiBaseCandidatePrediction:
    step: int
    edge: Edge
    probability: float
    selected: bool


@dataclass(frozen=True)
class SemiBasePredictionResult:
    n1: int
    n2: int
    n3: int
    edges_initial_base: tuple[Edge, ...]
    added_edges: tuple[Edge, ...]
    edges_semi_base: tuple[Edge, ...]
    candidates: tuple[SemiBaseCandidatePrediction, ...]

    def to_graph(self) -> HalfGraph:
        return HalfGraph(self.n1, self.n2, self.n3, self.edges_semi_base)


def predict_semi_base(
    checkpoint_path: str | Path,
    base_graph: HalfGraph,
    threshold: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    sample: bool = False,
    seed: int | None = None,
    edge_feedback_biases: Mapping[str, float] | None = None,
) -> SemiBasePredictionResult:
    return predict_semi_base_with_model(
        load_bond_model(checkpoint_path),
        base_graph,
        threshold=threshold,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        seed=seed,
        edge_feedback_biases=edge_feedback_biases,
    )


def predict_semi_base_with_model(
    model: BondCompletionModel,
    base_graph: HalfGraph,
    threshold: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    sample: bool = False,
    seed: int | None = None,
    edge_feedback_biases: Mapping[str, float] | None = None,
) -> SemiBasePredictionResult:
    """Use a dedicated edge-scoring GNN to connect all base components."""

    base_graph.validate_base_graph()
    rng = random.Random(seed)
    current = base_graph
    added: list[Edge] = []
    predictions: list[SemiBaseCandidatePrediction] = []
    model.eval()

    for step in range(1, len(base_graph.representative_nodes) + 1):
        if current.is_connected():
            current.validate_semi_base_graph()
            return SemiBasePredictionResult(
                n1=current.n1,
                n2=current.n2,
                n3=current.n3,
                edges_initial_base=base_graph.edges,
                added_edges=normalize_edges(added),
                edges_semi_base=current.edges,
                candidates=tuple(predictions),
            )

        candidates = semi_base_candidate_edges(current)
        if not candidates:
            break
        data = graph_to_data(current, candidates)
        with torch.no_grad():
            logits = model(data)
            if edge_feedback_biases:
                logits = logits + torch.tensor(
                    [
                        float(
                            edge_feedback_biases.get(
                                edge_feedback_key(current, edge),
                                0.0,
                            )
                        )
                        for edge in candidates
                    ],
                    dtype=logits.dtype,
                    device=logits.device,
                )
            probabilities = torch.sigmoid(
                logits / max(float(temperature), 1e-6)
            ).cpu().tolist()

        selected = _select_connector(
            candidates,
            probabilities,
            threshold=threshold,
            top_k=top_k,
            sample=sample,
            rng=rng,
        )
        predictions.extend(
            SemiBaseCandidatePrediction(
                step=step,
                edge=edge,
                probability=round(float(probability), 6),
                selected=edge == selected,
            )
            for edge, probability in zip(candidates, probabilities)
        )
        added.append(selected)
        current = HalfGraph(
            current.n1,
            current.n2,
            current.n3,
            normalize_edges(set(current.edges) | {selected}),
        )

    raise RuntimeError("semi-base GNN could not connect all base graph components")


def _select_connector(
    candidates: tuple[Edge, ...],
    probabilities: list[float],
    *,
    threshold: float,
    top_k: int | None,
    sample: bool,
    rng: random.Random,
) -> Edge:
    ranked = sorted(
        zip(candidates, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    if top_k is not None:
        ranked = ranked[: max(int(top_k), 1)]
    eligible = [item for item in ranked if item[1] >= threshold]
    pool = eligible or ranked
    if not pool:
        raise RuntimeError("semi-base GNN received no connector candidates")
    if not sample:
        return pool[0][0]
    weights = [max(float(probability), 1e-9) for _, probability in pool]
    return rng.choices([edge for edge, _ in pool], weights=weights, k=1)[0]
