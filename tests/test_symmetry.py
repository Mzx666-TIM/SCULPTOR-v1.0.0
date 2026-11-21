"""Lock-down tests for the pure symmetry layer (core/code/symmetry.py).

`symmetry.py` is a dependency-free graph helper that `generator.py` relies on.
These tests lock down its deterministic behaviour so future optimisation stays
safe. Only the non-random, deterministic functions are covered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCULPTOR_ROOT = Path(__file__).resolve().parents[1]
for code_dir in (SCULPTOR_ROOT / "core/code",):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

from graph import HalfGraph
from symmetry import (
    find_reflection_decomposition,
    mirror_complete,
    validate_full_carbon_degrees,
)


class TestMirrorComplete:
    def test_completes_n2_cross_edge(self) -> None:
        # n1=2 (0,1), n2=2 (rep node 2, mirror 3). Edge (0,2).
        graph = HalfGraph(n1=2, n2=2, n3=0, edges=((0, 2),))
        full = mirror_complete(graph)
        # mirror copies (0,2) -> (0,3); n2 cross-edge (2,3)
        assert set(full) == {(0, 2), (0, 3), (2, 3)}

    def test_normalizes_output_edges(self) -> None:
        graph = HalfGraph(n1=2, n2=2, n3=0, edges=((2, 0),))
        full = mirror_complete(graph)
        assert set(full) == {(0, 2), (0, 3), (2, 3)}


class TestValidateFullCarbonDegrees:
    def test_ok_within_limit(self) -> None:
        validate_full_carbon_degrees(4, ((0, 1), (1, 2), (2, 3)))

    def test_raises_on_overflow(self) -> None:
        with pytest.raises(ValueError):
            validate_full_carbon_degrees(6, ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5)))


class TestFindReflectionDecomposition:
    def test_straight_chain_has_reflection(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        dec = find_reflection_decomposition(6, edges)
        assert dec.n1 == 0
        # A 6-chain decomposes into 2 n2 pairs + 4? Actually computed earlier:
        # n1=0, n2=2, n3=4. Verify by round-tripping.
        half = dec.to_half_graph()
        completed = mirror_complete(half)
        assert len(completed) == len(edges)

    def test_empty_two_node_is_swap(self) -> None:
        # An edgeless 2-node graph still admits the swap (1,0) as a reflection.
        dec = find_reflection_decomposition(2, [])
        assert dec.permutation == (1, 0)
        assert dec.n1 == 0
        assert dec.n2 == 0
        assert dec.n3 == 2

    def test_star_tree_decomposition(self) -> None:
        # Star centred on node 1 (0-1, 1-2, 1-3) is symmetric about node 1.
        edges = [(0, 1), (1, 2), (1, 3)]
        dec = find_reflection_decomposition(4, edges)
        assert dec.permutation == (0, 1, 3, 2)
        assert dec.n1 == 2  # fixed nodes 0 and 1
