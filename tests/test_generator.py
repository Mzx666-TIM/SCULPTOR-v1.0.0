"""Lock-down tests for the pure generation helpers (generator.py).

`generator.py` imports `graph` and `symmetry` through flat imports, so both
code dirs must be on sys.path. Only the pure / deterministic helpers are tested
here; the random graph samplers are exercised for invariants, not exact output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCULPTOR_ROOT = Path(__file__).resolve().parents[1]
for code_dir in (
    SCULPTOR_ROOT / "core/code",
    SCULPTOR_ROOT / "Generation/sym_molecule_generation/code",
):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

from graph import HalfGraph
from generator import (
    _component_main_counts,
    _norm,
    iter_count_partitions,
)


class TestNorm:
    def test_orders_smaller_first(self) -> None:
        assert _norm(3, 1) == (1, 3)
        assert _norm(1, 3) == (1, 3)


class TestIterCountPartitions:
    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            iter_count_partitions(0)
        with pytest.raises(ValueError):
            iter_count_partitions(-4)

    def test_skips_degenerate_pure_linear(self) -> None:
        parts = iter_count_partitions(8)
        assert (8, 0, 0) not in parts
        assert (0, 0, 8) in parts  # even though n1=0, this passes the n1%2 parity check

    def test_all_partitions_sum_to_total(self) -> None:
        for total in (6, 7, 8, 12, 15):
            for n1, n2, n3 in iter_count_partitions(total):
                assert n1 + n2 + n3 == total

    def test_n2_and_n3_always_even(self) -> None:
        for total in (6, 9, 14):
            for _, n2, n3 in iter_count_partitions(total):
                assert n2 % 2 == 0
                assert n3 % 2 == 0


class TestComponentMainCounts:
    def test_counts_main_nodes_per_component(self) -> None:
        # n1=2 (0,1), n2=2 (2,3 rep=2). edges (0,2). Components: {0,2} and {1}.
        graph = HalfGraph(n1=2, n2=2, n3=0, edges=((0, 2),))
        counts = _component_main_counts(graph)
        by_size = sorted(counts, key=lambda item: len(item[0]))
        # {1} -> 1 main; {0,2} -> 2 main
        assert by_size == [({1}, 1), ({0, 2}, 2)]
