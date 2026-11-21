from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from graph import Edge, HalfGraph, normalize_edges


class UnsupportedSymmetryError(ValueError):
    """Raised when a graph cannot be represented by the project symmetry rules."""


@dataclass(frozen=True)
class ReflectionDecomposition:
    n1: int
    n2: int
    n3: int
    edges_half: tuple[Edge, ...]
    permutation: tuple[int, ...]

    def to_half_graph(self) -> HalfGraph:
        return HalfGraph(self.n1, self.n2, self.n3, self.edges_half)


def mirror_complete(half_graph: HalfGraph) -> tuple[Edge, ...]:
    """Complete a half graph into a full carbon skeleton by deterministic rules."""

    half_graph.validate_half_degrees()
    edges_full: set[Edge] = set(half_graph.edges)

    for node in half_graph.n2_original_range:
        edges_full.add(_norm(node, half_graph.mirror_node(node)))

    for u, v in half_graph.edges:
        u_fixed = u in half_graph.n1_range
        v_fixed = v in half_graph.n1_range
        if u_fixed and v_fixed:
            continue
        if not u_fixed and not v_fixed:
            edges_full.add(_norm(half_graph.mirror_node(u), half_graph.mirror_node(v)))
        elif not u_fixed and v_fixed:
            edges_full.add(_norm(half_graph.mirror_node(u), v))
        else:
            edges_full.add(_norm(u, half_graph.mirror_node(v)))

    completed = normalize_edges(edges_full)
    validate_full_carbon_degrees(half_graph.total_carbons, completed)
    return completed


def validate_full_carbon_degrees(
    num_carbons: int, edges: tuple[Edge, ...], max_carbon_degree: int = 4
) -> None:
    degree = [0] * num_carbons
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    for node, value in enumerate(degree):
        if value > max_carbon_degree:
            raise ValueError(
                f"full carbon node {node} degree {value} exceeds {max_carbon_degree}"
            )


def find_reflection_decomposition(
    num_nodes: int,
    edges: list[Edge] | tuple[Edge, ...],
    *,
    candidate_validator: Callable[[HalfGraph], None] | None = None,
) -> ReflectionDecomposition:
    """Find a non-trivial involutive graph automorphism and convert to a half graph.

    This is intentionally dependency-free and suitable for small carbon skeletons.
    It is a graph-level helper; SMILES parsing lives in `chemistry.py`.
    """

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    full_edges = normalize_edges(edges)
    adjacency = _adjacency(num_nodes, full_edges)
    degree = [len(adjacency[i]) for i in range(num_nodes)]

    candidates = _involutive_automorphisms(num_nodes, adjacency, degree)
    if not candidates:
        raise UnsupportedSymmetryError("no non-trivial reflection automorphism found")

    ordered = sorted(
        candidates, key=lambda p: _decomposition_score(p, adjacency), reverse=True
    )
    errors: list[Exception] = []
    for candidate in ordered:
        try:
            decomposition = _decompose_from_permutation(candidate, full_edges, adjacency)
            half_graph = decomposition.to_half_graph()
            if candidate_validator is not None:
                candidate_validator(half_graph)
            return decomposition
        except (UnsupportedSymmetryError, ValueError) as exc:
            errors.append(exc)

    if errors:
        raise UnsupportedSymmetryError(str(errors[0])) from errors[0]
    raise UnsupportedSymmetryError("no supported reflection decomposition found")


def find_generation_reflection_decomposition(
    num_nodes: int, edges: list[Edge] | tuple[Edge, ...]
) -> ReflectionDecomposition:
    """Select the best axis among generation-compatible candidates only."""

    return find_reflection_decomposition(
        num_nodes,
        edges,
        candidate_validator=lambda half: half.validate_complete_half_graph(),
    )


def _decompose_from_permutation(
    permutation: tuple[int, ...],
    full_edges: tuple[Edge, ...],
    adjacency: list[set[int]],
) -> ReflectionDecomposition:
    fixed = [i for i, j in enumerate(permutation) if i == j]
    pairs = [(i, permutation[i]) for i in range(len(permutation)) if i < permutation[i]]
    n2_pairs = [(a, b) for a, b in pairs if b in adjacency[a]]
    n3_pairs = [(a, b) for a, b in pairs if b not in adjacency[a]]

    n1 = len(fixed)
    n2 = len(n2_pairs) * 2
    n3 = len(n3_pairs) * 2
    canonical: dict[int, int] = {}
    for idx, old in enumerate(sorted(fixed)):
        canonical[old] = idx

    for idx, (a, b) in enumerate(n2_pairs):
        canonical[a] = n1 + idx
        canonical[b] = n1 + n2 // 2 + idx

    n3_start = n1 + n2
    n3_mirror_start = n1 + n2 + n3 // 2
    for idx, (a, b) in enumerate(n3_pairs):
        canonical[a] = n3_start + idx
        canonical[b] = n3_mirror_start + idx

    half_shape = HalfGraph(n1, n2, n3, ())
    half_nodes = set(half_shape.representative_nodes)
    n2_cross_edges = {
        _norm(node, half_shape.mirror_node(node))
        for node in half_shape.n2_original_range
    }

    edges_half: set[Edge] = set()
    for old_u, old_v in full_edges:
        u, v = canonical[old_u], canonical[old_v]
        edge = _project_full_edge_to_half(u, v, half_shape, half_nodes, n2_cross_edges)
        if edge is not None:
            edges_half.add(edge)

    return ReflectionDecomposition(
        n1=n1,
        n2=n2,
        n3=n3,
        edges_half=normalize_edges(edges_half),
        permutation=permutation,
    )


def _project_full_edge_to_half(
    u: int,
    v: int,
    half_shape: HalfGraph,
    half_nodes: set[int],
    n2_cross_edges: set[Edge],
) -> Edge | None:
    u_half = u in half_nodes
    v_half = v in half_nodes

    if u_half and v_half:
        return _norm(u, v)

    if u in half_shape.n1_range and not v_half:
        return _norm(u, _original_node(v, half_shape))
    if v in half_shape.n1_range and not u_half:
        return _norm(_original_node(u, half_shape), v)

    if not u_half and not v_half:
        return _norm(_original_node(u, half_shape), _original_node(v, half_shape))

    if _norm(u, v) in n2_cross_edges:
        return None
    raise UnsupportedSymmetryError(
        f"cross-face dynamic edge {(u, v)} is outside the supported rules"
    )


def _original_node(node: int, half_shape: HalfGraph) -> int:
    if node in half_shape.n2_mirror_range:
        return node - half_shape.n2 // 2
    if node in half_shape.n3_mirror_range:
        return node - half_shape.n3 // 2
    if node in half_shape.representative_nodes:
        return node
    raise UnsupportedSymmetryError(f"node {node} cannot be projected to half graph")


def _involutive_automorphisms(
    num_nodes: int, adjacency: list[set[int]], degree: list[int]
) -> list[tuple[int, ...]]:
    mapping = [-1] * num_nodes
    results: list[tuple[int, ...]] = []

    def assign_pair(a: int, b: int, undo: list[int]) -> bool:
        if mapping[a] != -1 or mapping[b] != -1:
            return False
        if degree[a] != degree[b]:
            return False
        mapping[a] = b
        undo.append(a)
        if a != b:
            mapping[b] = a
            undo.append(b)
        return _partial_is_consistent(mapping, adjacency)

    def backtrack() -> None:
        try:
            i = next(idx for idx, value in enumerate(mapping) if value == -1)
        except StopIteration:
            permutation = tuple(mapping)
            if any(i != j for i, j in enumerate(permutation)):
                results.append(permutation)
            return

        for j in range(i, num_nodes):
            if mapping[j] != -1:
                continue
            undo: list[int] = []
            if assign_pair(i, j, undo):
                backtrack()
            for idx in undo:
                mapping[idx] = -1

    backtrack()
    return results


def _partial_is_consistent(mapping: list[int], adjacency: list[set[int]]) -> bool:
    assigned = [i for i, value in enumerate(mapping) if value != -1]
    for u in assigned:
        mu = mapping[u]
        for v in assigned:
            mv = mapping[v]
            if (v in adjacency[u]) != (mv in adjacency[mu]):
                return False
    return True


def _decomposition_score(
    permutation: tuple[int, ...], adjacency: list[set[int]]
) -> tuple[int, int]:
    fixed = sum(1 for i, j in enumerate(permutation) if i == j)
    cross_bonds = sum(1 for i, j in enumerate(permutation) if i < j and j in adjacency[i])
    return fixed, cross_bonds


def _adjacency(num_nodes: int, edges: tuple[Edge, ...]) -> list[set[int]]:
    adjacency = [set() for _ in range(num_nodes)]
    for u, v in edges:
        if u >= num_nodes or v >= num_nodes:
            raise ValueError(f"edge {(u, v)} exceeds node range 0..{num_nodes - 1}")
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def _norm(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)
