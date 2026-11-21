from __future__ import annotations

import random
from itertools import combinations

from graph import Edge, HalfGraph


def completion_candidate_edges(
    base_graph: HalfGraph, max_carbon_degree: int = 4
) -> tuple[Edge, ...]:
    """List legal missing half-edges that a bond-completion model may predict."""

    existing = set(base_graph.edges)
    degree = base_graph.degrees()
    candidates: list[Edge] = []

    for u, v in combinations(base_graph.representative_nodes, 2):
        edge = (u, v)
        if edge in existing:
            continue
        if (
            degree[u] + 1 > base_graph.degree_limit(u, max_carbon_degree)
            or degree[v] + 1 > base_graph.degree_limit(v, max_carbon_degree)
        ):
            continue
        try:
            from symmetry import mirror_complete

            mirror_complete(
                HalfGraph(
                    base_graph.n1,
                    base_graph.n2,
                    base_graph.n3,
                    tuple(sorted(existing | {edge})),
                )
            )
        except ValueError:
            continue
        candidates.append(edge)

    return tuple(candidates)


def semi_base_candidate_edges(
    base_graph: HalfGraph, max_carbon_degree: int = 4
) -> tuple[Edge, ...]:
    """List legal missing half-edges that connect different base components."""

    if base_graph.is_connected():
        return ()

    component_index = _component_index(base_graph)
    return tuple(
        edge
        for edge in completion_candidate_edges(
            base_graph, max_carbon_degree=max_carbon_degree
        )
        if component_index[edge[0]] != component_index[edge[1]]
    )


def base_graph_to_semi_base_graph(
    base_graph: HalfGraph,
    rng: random.Random | None = None,
    max_carbon_degree: int = 4,
    max_attempts: int = 500,
    allowed_added_edges: set[Edge] | None = None,
) -> HalfGraph:
    """Connect a base graph into a single-component semi-base graph."""

    rng = rng or random.Random()
    base_graph.validate_base_graph(max_carbon_degree=max_carbon_degree)
    edges = set(base_graph.edges)

    for _ in range(max_attempts):
        graph = HalfGraph(base_graph.n1, base_graph.n2, base_graph.n3, tuple(edges))
        if graph.is_connected():
            graph.validate_semi_base_graph(max_carbon_degree=max_carbon_degree)
            return graph

        candidates = list(
            semi_base_candidate_edges(graph, max_carbon_degree=max_carbon_degree)
        )
        if allowed_added_edges is not None:
            candidates = [edge for edge in candidates if edge in allowed_added_edges]
        if not candidates:
            break
        edges.add(rng.choice(candidates))

    raise RuntimeError("failed to connect base graph into a semi-base graph")


def iter_count_partitions(total_carbons: int) -> tuple[tuple[int, int, int], ...]:
    """Enumerate `(n1, n2, n3)` for a fixed total carbon count.

    The degenerate `(n1=total_carbons, n2=0, n3=0)` partition (pure straight
    chain, no rings) is skipped because such molecules never pass the ring
    screen and only waste sampling budget.
    """

    if total_carbons <= 0:
        raise ValueError("total_carbons must be positive")
    partitions: list[tuple[int, int, int]] = []
    for n1 in range(total_carbons + 1):
        if n1 % 2 != total_carbons % 2:
            continue
        remaining = total_carbons - n1
        for n2 in range(0, remaining + 1, 2):
            n3 = remaining - n2
            # Pure linear alkanes (acyclic) are always rejected by
            # ring_count_below_min; skip them.
            if n2 == 0 and n3 == 0:
                continue
            partitions.append((n1, n2, n3))
    return tuple(partitions)


def generate_base_for_counts(
    n1: int,
    n2: int,
    n3: int,
    rng: random.Random | None = None,
    n3_group_continue_probability: float = 0.45,
    max_carbon_degree: int = 4,
    max_attempts: int = 500,
) -> HalfGraph:
    """Generate a base graph by processing n3 atoms into anchored groups."""

    rng = rng or random.Random()
    shape = HalfGraph(n1, n2, n3, ())
    if not shape.main_structure_nodes:
        raise ValueError("base graph generation requires at least one main node")

    for _ in range(max_attempts):
        try:
            graph = _generate_base_for_counts_once(
                shape,
                rng,
                n3_group_continue_probability,
                max_carbon_degree,
            )
            graph.validate_base_graph(max_carbon_degree=max_carbon_degree)
            return graph
        except ValueError:
            continue
    raise RuntimeError(f"failed to generate base graph for {(n1, n2, n3)}")


def iter_generated_bases_for_total(
    total_carbons: int,
    rng: random.Random | None = None,
    per_partition: int = 1,
    max_carbon_degree: int = 4,
) -> tuple[HalfGraph, ...]:
    """Generate base graphs for every `(n1, n2, n3)` partition of `total_carbons`."""

    rng = rng or random.Random()
    generated: list[HalfGraph] = []
    for n1, n2, n3 in iter_count_partitions(total_carbons):
        if n1 + n2 // 2 == 0:
            continue
        for _ in range(per_partition):
            generated.append(
                generate_base_for_counts(
                    n1,
                    n2,
                    n3,
                    rng=rng,
                    max_carbon_degree=max_carbon_degree,
                )
            )
    return tuple(generated)


def _generate_base_for_counts_once(
    shape: HalfGraph,
    rng: random.Random,
    n3_group_continue_probability: float,
    max_carbon_degree: int,
) -> HalfGraph:
    edges: set[Edge] = set()
    n3_nodes = list(shape.n3_original_range)
    index = 0

    while index < len(n3_nodes):
        anchor = rng.choice(list(shape.main_structure_nodes))
        group = [n3_nodes[index]]
        index += 1
        edges.add(_norm(anchor, group[0]))

        while index < len(n3_nodes) and rng.random() < n3_group_continue_probability:
            new_node = n3_nodes[index]
            parent = rng.choice(group)
            edges.add(_norm(parent, new_node))
            group.append(new_node)
            index += 1

    graph = HalfGraph(shape.n1, shape.n2, shape.n3, tuple(edges))
    graph.validate_base_graph(max_carbon_degree=max_carbon_degree)
    return graph


def base_graph_from_full_half(
    full_half: HalfGraph,
    rng: random.Random | None = None,
    deletion_probability: float = 0.5,
    max_carbon_degree: int = 4,
) -> HalfGraph:
    """Randomly delete bonds from a complete half graph until it is a base graph."""

    if not 0 <= deletion_probability <= 1:
        raise ValueError("deletion_probability must be between 0 and 1")

    rng = rng or random.Random()
    full_half.validate_complete_half_graph(max_carbon_degree=max_carbon_degree)
    kept = set(full_half.edges)

    for _ in range(5000):
        graph = HalfGraph(full_half.n1, full_half.n2, full_half.n3, tuple(kept))
        if _base_components_are_valid(graph):
            break

        candidates = _overfull_component_edges(graph)
        rng.shuffle(candidates)
        removed = False
        for edge in candidates:
            trial = set(kept)
            trial.remove(edge)
            trial_graph = HalfGraph(full_half.n1, full_half.n2, full_half.n3, tuple(trial))
            if _has_zero_main_component(trial_graph):
                continue
            kept = trial
            removed = True
            break
        if not removed:
            raise RuntimeError("failed to split full half graph into base components")
    else:
        raise RuntimeError("failed to derive a valid base graph from full half graph")

    # A base graph is a forest. Remove only cycle-closing edges, which preserves
    # each component and therefore its unique main-structure node.
    while not HalfGraph(
        full_half.n1,
        full_half.n2,
        full_half.n3,
        tuple(kept),
    ).is_acyclic():
        candidates = list(kept)
        rng.shuffle(candidates)
        removed = False
        for edge in candidates:
            trial = set(kept)
            trial.remove(edge)
            trial_graph = HalfGraph(
                full_half.n1,
                full_half.n2,
                full_half.n3,
                tuple(trial),
            )
            if not _base_components_are_valid(trial_graph):
                continue
            kept = trial
            removed = True
            break
        if not removed:
            raise RuntimeError("failed to remove cycles while deriving a base graph")

    edges = list(kept)
    rng.shuffle(edges)
    for edge in edges:
        if rng.random() >= deletion_probability:
            continue
        trial = set(kept)
        trial.remove(edge)
        trial_graph = HalfGraph(full_half.n1, full_half.n2, full_half.n3, tuple(trial))
        try:
            trial_graph.validate_base_graph(max_carbon_degree=max_carbon_degree)
        except ValueError:
            continue
        kept = trial

    base = HalfGraph(full_half.n1, full_half.n2, full_half.n3, tuple(kept))
    base.validate_base_graph(max_carbon_degree=max_carbon_degree)
    return base


def _component_main_counts(graph: HalfGraph) -> list[tuple[set[int], int]]:
    adjacency = {node: set() for node in graph.representative_nodes}
    for u, v in graph.edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    seen: set[int] = set()
    main_nodes = set(graph.main_structure_nodes)
    components: list[tuple[set[int], int]] = []
    for start in graph.representative_nodes:
        if start in seen:
            continue
        stack = [start]
        component: set[int] = set()
        seen.add(start)
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append((component, len(component & main_nodes)))
    return components


def _component_index(graph: HalfGraph) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for index, (component, _) in enumerate(_component_main_counts(graph)):
        for node in component:
            mapping[node] = index
    return mapping


def _base_components_are_valid(graph: HalfGraph) -> bool:
    return all(main_count == 1 for _, main_count in _component_main_counts(graph))


def _has_zero_main_component(graph: HalfGraph) -> bool:
    return any(main_count == 0 for _, main_count in _component_main_counts(graph))


def _overfull_component_edges(graph: HalfGraph) -> list[Edge]:
    edges: list[Edge] = []
    for component, main_count in _component_main_counts(graph):
        if main_count <= 1:
            continue
        component_edges = [
            edge for edge in graph.edges if edge[0] in component and edge[1] in component
        ]
        edges.extend(component_edges)
    return edges


def random_counts(
    rng: random.Random,
    min_carbons: int = 8,
    max_carbons: int = 16,
) -> tuple[int, int, int]:
    total = rng.randint(min_carbons, max_carbons)
    n1_options = [value for value in range(total + 1) if value % 2 == total % 2]
    n1 = rng.choice(n1_options)
    remaining = total - n1
    n2 = rng.randrange(0, remaining + 1, 2)
    n3 = remaining - n2
    return n1, n2, n3


def random_base_graph(
    rng: random.Random | None = None,
    min_carbons: int = 8,
    max_carbons: int = 16,
    edge_probability: float | None = None,
    max_carbon_degree: int = 4,
    max_attempts: int = 500,
) -> HalfGraph:
    rng = rng or random.Random()
    p = edge_probability if edge_probability is not None else rng.uniform(0.1, 0.3)

    for _ in range(max_attempts):
        n1, n2, n3 = random_counts(rng, min_carbons, max_carbons)
        half_graph = _sample_half_graph(rng, n1, n2, n3, p, max_carbon_degree)
        try:
            half_graph.validate_base_graph(max_carbon_degree=max_carbon_degree)
        except ValueError:
            continue
        return half_graph
    raise RuntimeError("failed to generate a valid base graph")


def _sample_half_graph(
    rng: random.Random,
    n1: int,
    n2: int,
    n3: int,
    edge_probability: float,
    max_carbon_degree: int,
) -> HalfGraph:
    shape = HalfGraph(n1=n1, n2=n2, n3=n3, edges=())
    parent = {node: node for node in shape.representative_nodes}
    degree = [0] * shape.total_carbons
    edges: list[Edge] = []
    # Precompute the per-node degree limit once instead of re-deriving it (via
    # kind()) for every candidate edge. degree_limit depends only on the node
    # kind, so this is exactly equivalent to the per-edge lookup.
    degree_limits = {
        node: shape.degree_limit(node, max_carbon_degree)
        for node in shape.representative_nodes
    }

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    candidates = list(combinations(shape.representative_nodes, 2))
    rng.shuffle(candidates)

    for u, v in candidates:
        if rng.random() > edge_probability:
            continue
        if (
            degree[u] >= degree_limits[u]
            or degree[v] >= degree_limits[v]
        ):
            continue
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        parent[ru] = rv
        degree[u] += 1
        degree[v] += 1
        edges.append((u, v))

    return HalfGraph(n1=n1, n2=n2, n3=n3, edges=tuple(edges))


def _norm(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)
