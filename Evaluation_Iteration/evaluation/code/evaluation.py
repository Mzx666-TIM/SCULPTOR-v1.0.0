from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import pickle
from typing import Mapping

from bond_model.predict import load_bond_model, predict_bonds_with_model
from bond_model.semi_base import predict_semi_base_with_model
from chemistry import canonical_smiles_from_carbon_edges
from generator import generate_base_for_counts, iter_count_partitions
from graph import Edge, HalfGraph


@dataclass(frozen=True)
class EvaluatedGeneration:
    generation_iteration: int
    n1: int
    n2: int
    n3: int
    edges_initial_base: tuple[Edge, ...]
    edges_base: tuple[Edge, ...]
    semi_base_edges_added: tuple[Edge, ...]
    predicted_edges: tuple[Edge, ...]
    edges_half_predicted: tuple[Edge, ...]
    edges_full_predicted: tuple[Edge, ...] | None
    candidate_count: int
    selected_count: int
    is_valid: bool
    is_connected: bool
    is_novel: bool | None
    canonical_smiles: str | None
    error: str | None


@dataclass(frozen=True)
class GenerationEvaluation:
    total_carbons: int
    requested_samples: int
    generated_bases: int
    predicted_with_candidates: int
    predicted_with_edges: int
    valid_molecules: int
    connected_molecules: int
    novel_molecules: int
    duplicate_known_molecules: int
    unique_valid_smiles: int
    failures: int
    use_semi_base: bool
    results: tuple[EvaluatedGeneration, ...]


def evaluate_generation(
    checkpoint_path: str | Path,
    known_smiles_path: str | Path | None,
    total_carbons: int,
    samples: int,
    threshold: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    sample: bool = False,
    seed: int | None = None,
    use_semi_base: bool = False,
    semi_base_checkpoint_path: str | Path | None = None,
    semi_base_threshold: float = 0.67,
    semi_base_temperature: float = 1.0,
    semi_base_top_k: int | None = None,
    partition_weights: Mapping[tuple[int, int, int], float] | None = None,
    edge_feedback_biases: Mapping[str, float] | None = None,
) -> GenerationEvaluation:
    """Generate base graphs, run the bond model, and validate full molecules."""

    if samples <= 0:
        raise ValueError("samples must be positive")

    rng = random.Random(seed)
    model = load_bond_model(checkpoint_path)
    semi_base_model = (
        load_bond_model(semi_base_checkpoint_path or checkpoint_path)
        if use_semi_base
        else None
    )
    known_smiles = _load_known_smiles(known_smiles_path)
    partitions = [
        counts
        for counts in iter_count_partitions(total_carbons)
        if counts[0] + counts[1] // 2 > 0
    ]
    if not partitions:
        raise ValueError("total_carbons has no partition with a main structure node")

    results: list[EvaluatedGeneration] = []
    failures = 0
    attempts = 0
    max_attempts = samples * 20
    while len(results) < samples and attempts < max_attempts:
        attempts += 1
        if partition_weights is None:
            n1, n2, n3 = partitions[len(results) % len(partitions)]
        else:
            weights = [
                max(0.0, float(partition_weights.get(partition, 0.0)))
                for partition in partitions
            ]
            if sum(weights) <= 0.0:
                weights = [1.0] * len(partitions)
            n1, n2, n3 = rng.choices(partitions, weights=weights, k=1)[0]
        try:
            base_graph = generate_base_for_counts(n1, n2, n3, rng=rng)
            if use_semi_base:
                semi_base_prediction = predict_semi_base_with_model(
                    semi_base_model,
                    base_graph,
                    threshold=semi_base_threshold,
                    temperature=semi_base_temperature,
                    top_k=semi_base_top_k,
                    sample=sample,
                    seed=rng.randrange(2**32) if sample else None,
                    edge_feedback_biases=edge_feedback_biases,
                )
                model_input_graph = semi_base_prediction.to_graph()
            else:
                model_input_graph = base_graph
            prediction = predict_bonds_with_model(
                model,
                model_input_graph,
                threshold=threshold,
                temperature=temperature,
                top_k=top_k,
                sample=sample,
                seed=rng.randrange(2**32) if sample else None,
                edge_feedback_biases=edge_feedback_biases,
            )
            semi_base_edges_added = tuple(
                edge for edge in model_input_graph.edges if edge not in set(base_graph.edges)
            )
            results.append(
                _evaluate_prediction(
                    prediction,
                    known_smiles,
                    edges_initial_base=base_graph.edges,
                    semi_base_edges_added=semi_base_edges_added,
                )
            )
        except Exception as exc:
            failures += 1
            if failures >= max_attempts:
                raise RuntimeError("generation evaluation failed repeatedly") from exc

    if len(results) < samples:
        failures += samples - len(results)

    valid = [result for result in results if result.is_valid]
    connected = [result for result in valid if result.is_connected]
    novel = [result for result in connected if result.is_novel]
    duplicate_known = [
        result for result in connected if result.is_novel is False
    ]
    unique_smiles = {
        result.canonical_smiles for result in valid if result.canonical_smiles
    }

    return GenerationEvaluation(
        total_carbons=total_carbons,
        requested_samples=samples,
        generated_bases=len(results),
        predicted_with_candidates=sum(1 for result in results if result.candidate_count),
        predicted_with_edges=sum(1 for result in results if result.selected_count),
        valid_molecules=len(valid),
        connected_molecules=len(connected),
        novel_molecules=len(novel),
        duplicate_known_molecules=len(duplicate_known),
        unique_valid_smiles=len(unique_smiles),
        failures=failures,
        use_semi_base=use_semi_base,
        results=tuple(results),
    )


def evaluation_to_json(evaluation: GenerationEvaluation) -> str:
    return json.dumps(asdict(evaluation), ensure_ascii=False)


def _evaluate_prediction(
    prediction,
    known_smiles: set[str],
    edges_initial_base: tuple[Edge, ...] | None = None,
    semi_base_edges_added: tuple[Edge, ...] = (),
) -> EvaluatedGeneration:
    error: str | None = None
    canonical_smiles: str | None = None
    is_valid = False
    is_connected = False
    is_novel: bool | None = None

    if prediction.edges_full_predicted is None:
        error = "mirror completion failed"
    else:
        try:
            is_connected = _is_connected(
                prediction.n1 + prediction.n2 + prediction.n3,
                prediction.edges_full_predicted,
            )
            canonical_smiles = canonical_smiles_from_carbon_edges(
                prediction.edges_full_predicted,
                prediction.n1 + prediction.n2 + prediction.n3,
            )
            is_valid = True
            is_novel = canonical_smiles not in known_smiles
        except Exception as exc:
            error = str(exc)

    return EvaluatedGeneration(
        generation_iteration=1,
        n1=prediction.n1,
        n2=prediction.n2,
        n3=prediction.n3,
        edges_initial_base=(
            prediction.edges_base if edges_initial_base is None else edges_initial_base
        ),
        edges_base=prediction.edges_base,
        semi_base_edges_added=semi_base_edges_added,
        predicted_edges=prediction.predicted_edges,
        edges_half_predicted=prediction.edges_half_predicted,
        edges_full_predicted=prediction.edges_full_predicted,
        candidate_count=len(prediction.candidates),
        selected_count=len(prediction.predicted_edges),
        is_valid=is_valid,
        is_connected=is_connected,
        is_novel=is_novel,
        canonical_smiles=canonical_smiles,
        error=error,
    )


@lru_cache(maxsize=8)
def _load_known_smiles(path: str | Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    known_path = Path(path)
    if not known_path.exists():
        raise FileNotFoundError(f"known SMILES file not found: {known_path}")
    cache_path = known_path.with_suffix(known_path.suffix + ".exact_set.pkl")
    source_stat = known_path.stat()
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if (
                cached.get("source_size") == source_stat.st_size
                and cached.get("source_mtime_ns") == source_stat.st_mtime_ns
            ):
                return frozenset(cached["smiles"])
        except (OSError, EOFError, AttributeError, KeyError, pickle.PickleError):
            pass
    smiles = frozenset(
        line.strip()
        for line in known_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(
                {
                    "source_size": source_stat.st_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "smiles": smiles,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary.replace(cache_path)
    except OSError:
        pass
    return smiles


def _is_connected(num_nodes: int, edges: tuple[Edge, ...]) -> bool:
    if num_nodes <= 1:
        return True

    adjacency = [set() for _ in range(num_nodes)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == num_nodes
