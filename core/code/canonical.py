from __future__ import annotations

import json
from itertools import permutations

from graph import Edge, HalfGraph


def base_signature(graph: HalfGraph) -> str:
    """Coarse structural signature used before exact base-graph equivalence.

    The signature follows the hand-checking rule:
    1. count n3 atoms directly connected to n1/n2 atoms;
    2. describe n3-only connected groups;
    3. include how each n3 group attaches to n1/n2 atoms.
    """

    adjacency = _adjacency(graph)
    n3_groups = _n3_components(graph, adjacency)
    n1_nodes = set(graph.n1_range)
    n2_nodes = set(graph.n2_original_range)

    n3_touching_n1 = 0
    n3_touching_n2 = 0
    n1_n3_edges = 0
    n2_n3_edges = 0
    for node in graph.n3_original_range:
        n1_count = len(adjacency[node] & n1_nodes)
        n2_count = len(adjacency[node] & n2_nodes)
        n1_n3_edges += n1_count
        n2_n3_edges += n2_count
        if n1_count:
            n3_touching_n1 += 1
        if n2_count:
            n3_touching_n2 += 1

    group_signatures = [
        _n3_group_signature(graph, adjacency, group) for group in n3_groups
    ]
    payload = {
        "n1_n3_edges": n1_n3_edges,
        "n2_n3_edges": n2_n3_edges,
        "n3_touching_n1": n3_touching_n1,
        "n3_touching_n2": n3_touching_n2,
        "n3_groups": sorted(group_signatures),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def are_base_graphs_equivalent(left: HalfGraph, right: HalfGraph) -> bool:
    """Return whether two base graphs are equivalent under type-preserving relabeling."""

    if (left.n1, left.n2, left.n3) != (right.n1, right.n2, right.n3):
        return False
    return canonical_base_key(left) == canonical_base_key(right)


def canonical_base_key(graph: HalfGraph) -> str:
    """Canonical key for a base graph under type-preserving relabeling."""

    return json.dumps(
        _canonical_base_encoding(graph),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def semi_base_added_edge_code(base_graph: HalfGraph, semi_base_graph: HalfGraph) -> str:
    """Canonical code for the edges added when connecting a base graph."""

    encoded = _canonical_base_extension_encoding(base_graph, semi_base_graph)
    return json.dumps(
        encoded["added_edges"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def semi_base_extension_key(base_graph: HalfGraph, semi_base_graph: HalfGraph) -> str:
    """Canonical key for `(base graph, added connection edges)`."""

    payload = {
        "base_key": canonical_base_key(base_graph),
        "added_edge_code": semi_base_added_edge_code(base_graph, semi_base_graph),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_base_encoding(graph: HalfGraph) -> tuple[object, ...]:
    return (
        graph.n1,
        graph.n2,
        graph.n3,
        base_signature(graph),
        _canonical_colored_incidence_smiles(graph, added_edges=()),
    )


def _canonical_base_extension_encoding(
    base_graph: HalfGraph, semi_base_graph: HalfGraph
) -> dict[str, object]:
    if (base_graph.n1, base_graph.n2, base_graph.n3) != (
        semi_base_graph.n1,
        semi_base_graph.n2,
        semi_base_graph.n3,
    ):
        raise ValueError("base and semi-base graphs must use the same counts")
    if not set(base_graph.edges).issubset(set(semi_base_graph.edges)):
        raise ValueError("semi-base graph must contain every base graph edge")

    added_edges = tuple(
        edge for edge in semi_base_graph.edges if edge not in set(base_graph.edges)
    )
    return {
        "base_encoding": _canonical_base_encoding(base_graph),
        "added_edges": _canonical_colored_incidence_smiles(
            semi_base_graph,
            added_edges=added_edges,
            node_color_graph=base_graph,
        ),
    }


def _main_degree_role(graph: HalfGraph, node: int) -> int:
    if node in graph.n1_range:
        return graph.n1_full_degree_from_half(node)
    if node in graph.n2_original_range:
        return graph.degrees()[node] + 1
    return graph.degrees()[node]


def _n3_group_signature(
    graph: HalfGraph, adjacency: dict[int, set[int]], group: tuple[int, ...]
) -> tuple[object, ...]:
    n1_nodes = set(graph.n1_range)
    n2_nodes = set(graph.n2_original_range)
    colors = {
        node: (len(adjacency[node] & n1_nodes), len(adjacency[node] & n2_nodes))
        for node in group
    }
    n3_edges = {
        _norm(a, b)
        for a in group
        for b in adjacency[a]
        if b in group and a < b
    }

    best: tuple[object, ...] | None = None
    for ordering in _color_preserving_orderings(group, colors):
        index = {node: idx for idx, node in enumerate(ordering)}
        encoded_edges = tuple(sorted((index[a], index[b]) for a, b in n3_edges))
        encoded_colors = tuple(colors[node] for node in ordering)
        encoded = (len(group), encoded_colors, encoded_edges)
        if best is None or encoded < best:
            best = encoded
    assert best is not None
    return best


def _canonical_colored_incidence_smiles(
    graph: HalfGraph,
    added_edges: tuple[Edge, ...],
    node_color_graph: HalfGraph | None = None,
) -> str:
    """Canonicalize a node/edge-colored graph without factorial relabeling.

    Edge colors are represented by labeled incidence nodes. RDKit then performs
    the exact canonical graph labeling in compiled code, including disconnected
    components and repeated isolated nodes.
    """

    from rdkit import Chem

    color_graph = node_color_graph or graph
    color_adjacency = _adjacency(color_graph)
    added = set(added_edges)
    editable = Chem.RWMol()
    node_indexes: dict[int, int] = {}

    for node in graph.representative_nodes:
        atom = Chem.Atom(6)
        atom.SetNoImplicit(True)
        atom.SetIsotope(
            _node_color_isotope(
                graph=color_graph,
                node=node,
                degree=len(color_adjacency[node]),
            )
        )
        node_indexes[node] = editable.AddAtom(atom)

    for u, v in graph.edges:
        edge_atom = Chem.Atom(6)
        edge_atom.SetNoImplicit(True)
        edge_atom.SetIsotope(901 if _norm(u, v) in added else 900)
        edge_index = editable.AddAtom(edge_atom)
        editable.AddBond(node_indexes[u], edge_index, Chem.BondType.SINGLE)
        editable.AddBond(edge_index, node_indexes[v], Chem.BondType.SINGLE)

    return Chem.MolToSmiles(
        editable.GetMol(),
        canonical=True,
        isomericSmiles=True,
    )


def _node_color_isotope(graph: HalfGraph, node: int, degree: int) -> int:
    kind_offset = {
        "n1": 100,
        "n2": 200,
        "n3": 300,
    }[graph.kind(node).value]
    return kind_offset + degree * 10 + _main_degree_role(graph, node)


def _color_preserving_orderings(
    nodes: tuple[int, ...], colors: dict[int, tuple[int, int]]
) -> list[tuple[int, ...]]:
    color_groups: dict[tuple[int, int], list[int]] = {}
    for node in nodes:
        color_groups.setdefault(colors[node], []).append(node)

    orderings: list[tuple[int, ...]] = [()]
    for color in sorted(color_groups):
        orderings = [
            prefix + permuted
            for prefix in orderings
            for permuted in permutations(color_groups[color])
        ]
    return orderings


def _n3_components(
    graph: HalfGraph, adjacency: dict[int, set[int]]
) -> list[tuple[int, ...]]:
    n3_nodes = set(graph.n3_original_range)
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in graph.n3_original_range:
        if start in seen:
            continue
        stack = [start]
        component: set[int] = set()
        seen.add(start)
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbor in adjacency[node] & n3_nodes:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return components


def _adjacency(graph: HalfGraph) -> dict[int, set[int]]:
    adjacency = {node: set() for node in graph.representative_nodes}
    for u, v in graph.edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def _is_consistent(
    node: int,
    candidate: int,
    mapping: dict[int, int],
    left_adj: dict[int, set[int]],
    right_adj: dict[int, set[int]],
) -> bool:
    for mapped_node, mapped_candidate in mapping.items():
        if (mapped_node in left_adj[node]) != (mapped_candidate in right_adj[candidate]):
            return False
    return True


def _norm(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)
