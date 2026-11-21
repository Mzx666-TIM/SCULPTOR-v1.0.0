from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


Edge = tuple[int, int]


class NodeKind(str, Enum):
    FIXED = "n1"
    MIRROR_BONDED = "n2"
    MOVING = "n3"


def normalize_edge(edge: Edge) -> Edge:
    u, v = edge
    if u == v:
        raise ValueError("self loops are not valid carbon-carbon bonds")
    if u < 0 or v < 0:
        raise ValueError("node indexes must be non-negative")
    return (u, v) if u < v else (v, u)


def normalize_edges(edges: Iterable[Edge]) -> tuple[Edge, ...]:
    return tuple(sorted({normalize_edge(edge) for edge in edges}))


def carbon_degree_limit(kind: NodeKind, max_carbon_degree: int = 4) -> int:
    if kind is NodeKind.MIRROR_BONDED:
        return max_carbon_degree - 1
    return max_carbon_degree


@dataclass(frozen=True)
class HalfGraph:
    """Half-side graph encoded as `(n1, n2, n3, edges)`.

    `n2` and `n3` are total full-molecule counts, so the half graph contains
    `n2 / 2` and `n3 / 2` representative moving nodes.
    """

    n1: int
    n2: int
    n3: int
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if self.n1 < 0 or self.n2 < 0 or self.n3 < 0:
            raise ValueError("n1, n2, and n3 must be non-negative")
        if self.n2 % 2 or self.n3 % 2:
            raise ValueError("n2 and n3 must be even")
        normalized = normalize_edges(self.edges)
        object.__setattr__(self, "edges", normalized)
        representative_nodes = set(self.representative_nodes)
        for u, v in normalized:
            if u not in representative_nodes or v not in representative_nodes:
                raise ValueError(
                    f"edge {(u, v)} uses a node outside the half graph originals"
                )

    @property
    def total_carbons(self) -> int:
        return self.n1 + self.n2 + self.n3

    @property
    def half_node_count(self) -> int:
        return self.n1 + self.n2 // 2 + self.n3 // 2

    @property
    def representative_nodes(self) -> tuple[int, ...]:
        return tuple(self.n1_range) + tuple(self.n2_original_range) + tuple(
            self.n3_original_range
        )

    @property
    def main_structure_nodes(self) -> tuple[int, ...]:
        return tuple(self.n1_range) + tuple(self.n2_original_range)

    @property
    def n1_range(self) -> range:
        return range(0, self.n1)

    @property
    def n2_original_range(self) -> range:
        return range(self.n1, self.n1 + self.n2 // 2)

    @property
    def n2_mirror_range(self) -> range:
        return range(self.n1 + self.n2 // 2, self.n1 + self.n2)

    @property
    def n3_original_range(self) -> range:
        start = self.n1 + self.n2
        return range(start, start + self.n3 // 2)

    @property
    def n3_mirror_range(self) -> range:
        start = self.n1 + self.n2 + self.n3 // 2
        return range(start, self.total_carbons)

    def mirror_node(self, node: int) -> int:
        if node in self.n1_range:
            return node
        if node in self.n2_original_range:
            return node + self.n2 // 2
        if node in self.n3_original_range:
            return node + self.n3 // 2
        raise ValueError(f"node {node} is outside half graph originals")

    def kind(self, node: int) -> NodeKind:
        if node in self.n1_range:
            return NodeKind.FIXED
        if node in self.n2_original_range:
            return NodeKind.MIRROR_BONDED
        if node in self.n3_original_range:
            return NodeKind.MOVING
        raise ValueError(f"node {node} is outside half graph originals")

    def degrees(self) -> list[int]:
        degree = [0] * self.total_carbons
        for u, v in self.edges:
            degree[u] += 1
            degree[v] += 1
        return degree

    def degree_limit(self, node: int, max_carbon_degree: int = 4) -> int:
        return carbon_degree_limit(self.kind(node), max_carbon_degree)

    def n1_full_degree_from_half(self, node: int) -> int:
        if node not in self.n1_range:
            raise ValueError(f"node {node} is not an n1 node")
        full_degree = 0
        for u, v in self.edges:
            if node not in (u, v):
                continue
            other = v if u == node else u
            if other in self.n1_range:
                full_degree += 1
            elif other in self.n2_original_range or other in self.n3_original_range:
                full_degree += 2
            else:
                raise ValueError(f"node {other} is outside half graph originals")
        return full_degree

    def is_acyclic(self) -> bool:
        parent = {node: node for node in self.representative_nodes}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in self.edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            parent[ru] = rv
        return True

    def validate_complete_half_graph(self, max_carbon_degree: int = 4) -> None:
        self.validate_n1_linear_forest()
        degrees = self.degrees()
        # Accumulate the *completed* (mirrored) degree for every n1 node in one
        # pass, instead of re-scanning all edges once per n1 node. An n1 node's
        # full degree counts an n1 neighbour once and an n2/n3 original twice.
        n1_full_degree = dict.fromkeys(self.n1_range, 0)
        for u, v in self.edges:
            if u in n1_full_degree:
                n1_full_degree[u] += 1 if v in self.n1_range else 2
            if v in n1_full_degree:
                n1_full_degree[v] += 1 if u in self.n1_range else 2
        for node in self.representative_nodes:
            if node in self.n1_range:
                full_degree = n1_full_degree[node]
                if full_degree > max_carbon_degree:
                    raise ValueError(
                        f"n1 node {node} mirrored degree {full_degree} exceeds "
                        f"{max_carbon_degree}"
                    )
                continue

            degree = degrees[node]
            limit = self.degree_limit(node, max_carbon_degree)
            if degree > limit:
                raise ValueError(f"node {node} degree {degree} exceeds {limit}")

    def validate_n1_linear_forest(self) -> None:
        """Require the n1-induced graph to be a disjoint union of paths."""

        parent = {node: node for node in self.n1_range}
        n1_degree = {node: 0 for node in self.n1_range}

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for u, v in self.edges:
            if u not in parent or v not in parent:
                continue
            n1_degree[u] += 1
            n1_degree[v] += 1
            if n1_degree[u] > 2 or n1_degree[v] > 2:
                raise ValueError("the n1-induced graph must not branch")
            root_u = find(u)
            root_v = find(v)
            if root_u == root_v:
                raise ValueError("the n1-induced graph must be acyclic")
            parent[root_u] = root_v

    def validate_half_degrees(self, max_carbon_degree: int = 4) -> None:
        self.validate_complete_half_graph(max_carbon_degree=max_carbon_degree)

    def validate_base_graph(self, max_carbon_degree: int = 4) -> None:
        self.validate_complete_half_graph(max_carbon_degree=max_carbon_degree)
        if not self.is_acyclic():
            raise ValueError("a base graph must be acyclic")
        self.validate_base_components()

    def validate_semi_base_graph(self, max_carbon_degree: int = 4) -> None:
        self.validate_complete_half_graph(max_carbon_degree=max_carbon_degree)
        if not self.is_connected():
            raise ValueError("a semi-base graph must contain exactly one component")

    def is_connected(self) -> bool:
        if not self.representative_nodes:
            return True
        return len(self._connected_components()) == 1

    def validate_base_components(self) -> None:
        main_nodes = set(self.main_structure_nodes)
        for component in self._connected_components():
            main_count = len(component & main_nodes)
            if main_count != 1:
                raise ValueError(
                    "each base graph connected component must contain exactly "
                    f"one main structure node; component {sorted(component)} "
                    f"contains {main_count}"
                )

    def _adjacency(self) -> dict[int, set[int]]:
        adjacency = {node: set() for node in self.representative_nodes}
        for u, v in self.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        return adjacency

    def _connected_components(self) -> list[set[int]]:
        """Return the connected components over the representative nodes."""

        adjacency = self._adjacency()
        seen: set[int] = set()
        components: list[set[int]] = []
        for start in self.representative_nodes:
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
            components.append(component)
        return components

    def as_record(self) -> dict[str, object]:
        return {
            "n1": self.n1,
            "n2": self.n2,
            "n3": self.n3,
            "edges": list(self.edges),
        }
