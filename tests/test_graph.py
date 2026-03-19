"""Tests for IntentGraph."""

from ftg.graph import IntentGraph
from ftg.models import Dimension, Edge, EdgeSource, Node, RelationType


def _node(nid: str, dim: Dimension = Dimension.WHAT) -> Node:
    return Node(id=nid, content=f"content_{nid}", dimension=dim)


class TestCausalInvariant:
    """CAUSAL edges MUST be USER_EXPLICIT. Non-compliant edges auto-downgrade to SUPPORTS."""

    def test_user_explicit_causal_accepted(self):
        g = IntentGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))
        edge = Edge(
            source="a", target="b",
            relation_type=RelationType.CAUSAL,
            source_type=EdgeSource.USER_EXPLICIT,
        )
        assert g.add_edge(edge) is True
        edges = g.get_all_edges()
        assert len(edges) == 1
        assert edges[0].relation_type == RelationType.CAUSAL

    def test_user_implied_causal_downgraded(self):
        g = IntentGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))
        edge = Edge(
            source="a", target="b",
            relation_type=RelationType.CAUSAL,
            source_type=EdgeSource.USER_IMPLIED,
        )
        assert g.add_edge(edge) is True
        edges = g.get_all_edges()
        assert len(edges) == 1
        assert edges[0].relation_type == RelationType.SUPPORTS

    def test_algorithm_inferred_causal_downgraded(self):
        g = IntentGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))
        edge = Edge(
            source="a", target="b",
            relation_type=RelationType.CAUSAL,
            source_type=EdgeSource.ALGORITHM_INFERRED,
        )
        g.add_edge(edge)
        edges = g.get_all_edges()
        assert edges[0].relation_type == RelationType.SUPPORTS


class TestDAGEnforcement:
    """Causal/dependency edges must not create cycles."""

    def test_no_self_loop(self):
        g = IntentGraph()
        g.add_node(_node("a"))
        edge = Edge(source="a", target="a", relation_type=RelationType.SUPPORTS)
        assert g.add_edge(edge) is False

    def test_cycle_rejected(self):
        g = IntentGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))
        g.add_node(_node("c"))

        assert g.add_edge(Edge(
            source="a", target="b",
            relation_type=RelationType.DEPENDENCY,
            source_type=EdgeSource.USER_EXPLICIT,
        ))
        assert g.add_edge(Edge(
            source="b", target="c",
            relation_type=RelationType.DEPENDENCY,
            source_type=EdgeSource.USER_EXPLICIT,
        ))
        # c -> a would create a cycle
        result = g.add_edge(Edge(
            source="c", target="a",
            relation_type=RelationType.DEPENDENCY,
            source_type=EdgeSource.USER_EXPLICIT,
        ))
        assert result is False
        assert g.edge_count() == 2

    def test_supports_edges_allow_cycles(self):
        """Non-causal edges (SUPPORTS) don't need DAG property."""
        g = IntentGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))
        assert g.add_edge(Edge(
            source="a", target="b",
            relation_type=RelationType.SUPPORTS,
            source_type=EdgeSource.USER_IMPLIED,
        ))
        assert g.add_edge(Edge(
            source="b", target="a",
            relation_type=RelationType.SUPPORTS,
            source_type=EdgeSource.USER_IMPLIED,
        ))
        assert g.edge_count() == 2


class TestEdgeRejection:
    def test_missing_nodes_rejected(self):
        g = IntentGraph()
        g.add_node(_node("a"))
        edge = Edge(source="a", target="nonexistent", relation_type=RelationType.SUPPORTS)
        assert g.add_edge(edge) is False

    def test_connectivity_score_empty(self):
        g = IntentGraph()
        assert g.connectivity_score() == 0.0

    def test_connectivity_score_connected(self):
        g = IntentGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))
        g.add_edge(Edge(
            source="a", target="b",
            relation_type=RelationType.SUPPORTS,
            source_type=EdgeSource.USER_IMPLIED,
        ))
        assert g.connectivity_score() == 0.8
        assert g.connectivity_score(mode="all") == 1.0

    def test_dimension_node_counts(self):
        g = IntentGraph()
        g.add_node(_node("a", Dimension.WHAT))
        g.add_node(_node("b", Dimension.WHY))
        g.add_node(_node("c", Dimension.WHAT))
        counts = g.dimension_node_counts()
        assert counts["what"] == 2
        assert counts["why"] == 1
        assert counts["how"] == 0
