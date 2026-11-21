"""Lock-down tests for the pure graph layer (core/code/graph.py).

These tests assert the *current* behavior so that any optimisation that must
not change results has a safety net. They import the modules through the same
flat-import mechanism the runtime uses (each code dir is on sys.path).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCULPTOR_ROOT = Path(__file__).resolve().parents[1]
for code_dir in (SCULPTOR_ROOT / "core/code",):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

from graph import HalfGraph, NodeKind, carbon_degree_limit, normalize_edge, normalize_edges


class TestNormalizeEdge:
    def test_orders_smaller_first(self) -> None:
        assert normalize_edge((3, 1)) == (1, 3)
        assert normalize_edge((1, 3)) == (1, 3)

    def test_rejects_self_loop(self) -> None:
        with pytest.raises(ValueError):
            normalize_edge((2, 2))

    def test_rejects_negative_index(self) -> None:
        with pytest.raises(ValueError):
            normalize_edge((-1, 3))


class TestNormalizeEdges:
    def test_dedups_and_sorts(self) -> None:
        assert normalize_edges([(3, 1), (2, 4), (1, 3)]) == ((1, 3), (2, 4))

    def test_returns_empty_for_empty(self) -> None:
        assert normalize_edges([]) == ()


class TestDegreeLimit:
    def test_mirror_bonded_gets_one_less(self) -> None:
        assert carbon_degree_limit(NodeKind.MIRROR_BONDED) == 3

    def test_others_get_full(self) -> None:
        assert carbon_degree_limit(NodeKind.FIXED) == 4
        assert carbon_degree_limit(NodeKind.MOVING) == 4


class TestHalfGraph:
    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValueError):
            HalfGraph(n1=-1, n2=0, n3=0, edges=())

    def test_rejects_odd_n2_n3(self) -> None:
        with pytest.raises(ValueError):
            HalfGraph(n1=1, n2=1, n3=0, edges=())

    def test_normalizes_edges_in_post_init(self) -> None:
        graph = HalfGraph(n1=2, n2=2, n3=0, edges=((2, 0),))
        assert graph.edges == ((0, 2),)

    def test_rejects_edge_outside_originals(self) -> None:
        with pytest.raises(ValueError):
            HalfGraph(n1=2, n2=2, n3=0, edges=((0, 5),))

    def test_counts(self) -> None:
        graph = HalfGraph(n1=2, n2=2, n3=2, edges=())
        assert graph.total_carbons == 6
        assert graph.half_node_count == 4

    def test_representative_nodes_order(self) -> None:
        graph = HalfGraph(n1=1, n2=4, n3=2, edges=())
        # n1: 0; n2 originals: range(1, 3) -> (1, 2); n3 originals: range(5, 6) -> (5,)
        assert graph.representative_nodes == (0, 1, 2, 5)

    def test_mirror_node_for_n1_is_itself(self) -> None:
        graph = HalfGraph(n1=3, n2=0, n3=0, edges=())
        assert graph.mirror_node(0) == 0

    def test_kind_classification(self) -> None:
        graph = HalfGraph(n1=2, n2=2, n3=2, edges=())
        assert graph.kind(0) is NodeKind.FIXED
        assert graph.kind(2) is NodeKind.MIRROR_BONDED
        assert graph.kind(4) is NodeKind.MOVING

    def test_degree_limit_delegates_to_kind(self) -> None:
        graph = HalfGraph(n1=2, n2=2, n3=0, edges=())
        assert graph.degree_limit(2) == 3

    def test_acyclic_true_for_forest(self) -> None:
        graph = HalfGraph(n1=2, n2=2, n3=0, edges=((0, 2),))
        assert graph.is_acyclic() is True


class TestValidate:
    def test_validate_base_graph_requires_acyclic(self) -> None:
        graph = HalfGraph(n1=2, n2=2, n3=0, edges=((0, 2), (2, 1), (0, 1)))
        with pytest.raises(ValueError):
            graph.validate_base_graph()

    def test_validate_complete_half_graph_ok(self) -> None:
        graph = HalfGraph(n1=2, n2=2, n3=0, edges=((0, 2),))
        graph.validate_complete_half_graph()

    def test_validate_complete_half_graph_rejects_n1_overflow(self) -> None:
        # Node 0 is n1, connected to three n2 originals -> full degree 3*2=6 > 4.
        graph = HalfGraph(n1=2, n2=6, n3=0, edges=((0, 2), (0, 3), (0, 4)))
        with pytest.raises(ValueError):
            graph.validate_complete_half_graph()

    def test_n1_full_degree_weights_moving_bonds_double(self) -> None:
        # n1 node 0 has one n2-neighbour (counts 2) and one n1-neighbour (counts 1).
        graph = HalfGraph(n1=2, n2=2, n3=2, edges=((0, 2), (0, 1)))
        assert graph.n1_full_degree_from_half(0) == 3

    def test_validate_half_graph_branched_n1_rejected(self) -> None:
        # Node 0 has three n1 neighbours -> n1-induced graph branches.
        graph = HalfGraph(n1=5, n2=0, n3=0, edges=((0, 1), (0, 2), (0, 3)))
        with pytest.raises(ValueError):
            graph.validate_n1_linear_forest()

    def test_validate_n1_linear_forest_accepts_plain_path(self) -> None:
        # A forest of paths (0-1-2) is valid even though it touches one node twice.
        graph = HalfGraph(n1=3, n2=0, n3=0, edges=((0, 1), (1, 2)))
        graph.validate_n1_linear_forest()  # must not raise


class TestAsRecord:
    def test_round_trip_shape(self) -> None:
        graph = HalfGraph(n1=2, n2=2, n3=2, edges=((0, 2),))
        record = graph.as_record()
        assert record["n1"] == 2
        assert record["n2"] == 2
        assert record["n3"] == 2
        assert record["edges"] == [(0, 2)]
