from __future__ import annotations

from dataclasses import dataclass

from chemistry import (
    FIVE_MEMBER_RING_PREFERENCE_WEIGHT,
    FOUR_MEMBER_RING_STRAIN_WEIGHT,
    SIX_MEMBER_RING_PREFERENCE_WEIGHT,
    THREE_MEMBER_RING_STRAIN_WEIGHT,
)
from graph import Edge, HalfGraph, normalize_edge, normalize_edges


@dataclass(frozen=True)
class HalfGraphRingCounts:
    real_only_rings: int
    virtual_rings: int
    projected_total_rings: int
    projected_three_member_rings: int
    projected_four_member_rings: int
    projected_five_member_rings: int
    projected_six_member_rings: int
    strain_ring_score: int
    preferred_ring_score: int
    virtual_edges: tuple[Edge, ...]


def half_graph_ring_counts(graph: HalfGraph) -> HalfGraphRingCounts:
    """Return half-graph ring diagnostics.

    Adjacent n1 nodes are connected by virtual edges when the real half graph has
    no such edge. A minimal cycle basis is used so large cycles composed from
    smaller rings are not counted again. The projected total is not a substitute
    for the full-molecule cycle rank and must not be used to gate screening.
    """

    real_edges = set(graph.edges)
    virtual_edges = {
        normalize_edge((node, node + 1))
        for node in graph.n1_range
        if node + 1 < graph.n1 and normalize_edge((node, node + 1)) not in real_edges
    }
    augmented_edges = normalize_edges(real_edges | virtual_edges)
    cycles = _minimal_cycle_basis(
        nodes=graph.representative_nodes,
        edges=augmented_edges,
        virtual_edges=virtual_edges,
    )

    real_only = 0
    with_virtual = 0
    projected_three_member = 0
    projected_four_member = 0
    projected_five_member = 0
    projected_six_member = 0
    for cycle in cycles:
        has_virtual_edge = any(edge in virtual_edges for edge in cycle)
        multiplier = 1 if has_virtual_edge else 2
        cycle_size = len(_cycle_nodes(cycle))
        if cycle_size == 3:
            projected_three_member += multiplier
        elif cycle_size == 4:
            projected_four_member += multiplier
        elif cycle_size == 5:
            projected_five_member += multiplier
        elif cycle_size == 6:
            projected_six_member += multiplier

        if has_virtual_edge:
            with_virtual += 1
        else:
            real_only += 1

    return HalfGraphRingCounts(
        real_only_rings=real_only,
        virtual_rings=with_virtual,
        projected_total_rings=with_virtual + real_only * 2,
        projected_three_member_rings=projected_three_member,
        projected_four_member_rings=projected_four_member,
        projected_five_member_rings=projected_five_member,
        projected_six_member_rings=projected_six_member,
        # Ring-size preference of the design: 3-/4-membered rings are thermal
        # stability risks (raised on 2026-09-23: three-membered x6, four-membered
        # x3), while 5-/6-membered rings are the wanted motifs (x2 each).
        strain_ring_score=(
            projected_three_member * THREE_MEMBER_RING_STRAIN_WEIGHT
            + projected_four_member * FOUR_MEMBER_RING_STRAIN_WEIGHT
        ),
        preferred_ring_score=(
            projected_five_member * FIVE_MEMBER_RING_PREFERENCE_WEIGHT
            + projected_six_member * SIX_MEMBER_RING_PREFERENCE_WEIGHT
        ),
        virtual_edges=tuple(sorted(virtual_edges)),
    )


def _minimal_cycle_basis(
    nodes: tuple[int, ...],
    edges: tuple[Edge, ...],
    virtual_edges: set[Edge],
) -> tuple[tuple[Edge, ...], ...]:
    target_rank = _cycle_rank(nodes, edges)
    if target_rank <= 0:
        return ()

    all_cycles = _all_simple_cycles(nodes, edges)
    edge_index = {edge: index for index, edge in enumerate(edges)}
    ranked = sorted(
        all_cycles,
        key=lambda cycle: (
            len(cycle),
            sum(1 for edge in cycle if edge in virtual_edges),
            cycle,
        ),
    )

    basis: dict[int, int] = {}
    selected: list[tuple[Edge, ...]] = []
    for cycle in ranked:
        mask = _cycle_mask(cycle, edge_index)
        if _add_independent_vector(mask, basis):
            selected.append(cycle)
            if len(selected) == target_rank:
                break

    return tuple(selected)


def _all_simple_cycles(
    nodes: tuple[int, ...],
    edges: tuple[Edge, ...],
) -> tuple[tuple[Edge, ...], ...]:
    adjacency = {node: set() for node in nodes}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    cycles: set[tuple[Edge, ...]] = set()
    for start in sorted(nodes):
        _dfs_cycles(start, start, adjacency, [start], {start}, cycles)
    return tuple(sorted(cycles))


def _dfs_cycles(
    start: int,
    current: int,
    adjacency: dict[int, set[int]],
    path: list[int],
    seen: set[int],
    cycles: set[tuple[Edge, ...]],
) -> None:
    for neighbor in sorted(adjacency[current]):
        if neighbor == start and len(path) >= 3:
            cycles.add(_cycle_edges(path))
            continue
        if neighbor <= start or neighbor in seen:
            continue
        seen.add(neighbor)
        path.append(neighbor)
        _dfs_cycles(start, neighbor, adjacency, path, seen, cycles)
        path.pop()
        seen.remove(neighbor)


def _cycle_edges(path: list[int]) -> tuple[Edge, ...]:
    edges = [
        normalize_edge((path[index], path[(index + 1) % len(path)]))
        for index in range(len(path))
    ]
    return tuple(sorted(edges))


def _cycle_nodes(cycle: tuple[Edge, ...]) -> set[int]:
    nodes: set[int] = set()
    for u, v in cycle:
        nodes.add(u)
        nodes.add(v)
    return nodes


def _cycle_rank(nodes: tuple[int, ...], edges: tuple[Edge, ...]) -> int:
    return len(edges) - len(nodes) + _component_count(nodes, edges)


def _component_count(nodes: tuple[int, ...], edges: tuple[Edge, ...]) -> int:
    adjacency = {node: set() for node in nodes}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    remaining = set(nodes)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def _cycle_mask(cycle: tuple[Edge, ...], edge_index: dict[Edge, int]) -> int:
    mask = 0
    for edge in cycle:
        mask ^= 1 << edge_index[edge]
    return mask


def _add_independent_vector(mask: int, basis: dict[int, int]) -> bool:
    value = mask
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = value
            return True
        value ^= basis[pivot]
    return False
