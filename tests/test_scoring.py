"""Tests for Fathom score computation and saturation gates."""

from ftg.scoring import (
    DEPTH_CAP,
    compute_fathom_breakdown,
    evaluate_fathom_gates,
    information_gain,
)
from ftg.graph import IntentGraph
from ftg.models import CausalHypothesis, Dimension, Edge, EdgeSource, Node, NodeType, RelationType


def _make_node(node_id: str, dim: Dimension, content: str = "") -> Node:
    return Node(id=node_id, content=content or f"{dim.value}-{node_id}", dimension=dim)


class TestFathomBreakdownV2:
    def test_secondary_deeper_than_primary_does_not_score(self):
        g = IntentGraph()
        g.add_node(
            Node(
                id="n1",
                content="classical language curriculum",
                dimension=Dimension.WHAT,
                secondary_dimensions=["how"],
            )
        )

        primary_only = compute_fathom_breakdown(
            g,
            {"what", "how"},
            creditable_dimensions={"what"},
        )
        blocked = compute_fathom_breakdown(
            g,
            {"what", "how"},
            creditable_dimensions={"what", "how"},
        )

        assert blocked.fathom_score == primary_only.fathom_score
        assert blocked.blocked_secondary_count == 1
        assert blocked.blocked_secondary_dimensions == ["how"]

    def test_secondary_same_depth_can_still_score(self):
        g = IntentGraph()
        g.add_node(
            Node(
                id="n1",
                content="curriculum in Western Europe",
                dimension=Dimension.WHAT,
                secondary_dimensions=["where"],
            )
        )

        primary_only = compute_fathom_breakdown(
            g,
            {"what", "where"},
            creditable_dimensions={"what"},
        )
        with_secondary = compute_fathom_breakdown(
            g,
            {"what", "where"},
            creditable_dimensions={"what", "where"},
        )

        assert with_secondary.fathom_score > primary_only.fathom_score
        assert with_secondary.blocked_secondary_count == 0

    def test_missing_primary_with_secondary_contributes_no_score(self):
        g = IntentGraph()
        g.add_node(
            Node(
                id="n1",
                content="uncertain dimension signal",
                dimension=None,
                secondary_dimensions=["how"],
            )
        )

        breakdown = compute_fathom_breakdown(
            g,
            {"how"},
            creditable_dimensions={"how"},
        )

        assert breakdown.atom_count == 0
        assert breakdown.fathom_score == 0.0
        assert breakdown.blocked_secondary_count == 1
        assert breakdown.blocked_secondary_dimensions == ["how"]

    def test_volunteered_inquiry_nodes_are_not_capped(self):
        intent_graph = IntentGraph()
        intent_graph.add_node(
            Node(
                id="intent_1",
                content="want to know what these languages were mainly used for",
                raw_quote="mainly used for what purposes",
                node_type=NodeType.INTENT,
                dimension=Dimension.WHY,
            )
        )

        belief_graph = IntentGraph()
        belief_graph.add_node(
            Node(
                id="belief_1",
                content="the practical value matters because it affects my child",
                raw_quote="this will affect my child, so practical value matters",
                node_type=NodeType.BELIEF,
                dimension=Dimension.WHY,
            )
        )

        intent_breakdown = compute_fathom_breakdown(
            intent_graph,
            {"why"},
            creditable_dimensions={"why"},
        )
        belief_breakdown = compute_fathom_breakdown(
            belief_graph,
            {"why"},
            creditable_dimensions={"why"},
        )

        assert intent_breakdown.fathom_score == belief_breakdown.fathom_score

    def test_promoted_dimension_response_does_not_add_extra_score(self):
        g = IntentGraph()
        g.add_node(_make_node("n1", Dimension.WHAT, "task object"))

        volunteered = compute_fathom_breakdown(g, {"what"})
        elicited = compute_fathom_breakdown(
            g,
            {"what"},
            dimension_bound_responses={
                "what": {
                    "raw_text": "I want the comparison to focus on practical value",
                    "round": 2,
                    "node_ids": ["n1"],
                }
            },
        )

        assert elicited.fathom_score == volunteered.fathom_score
        assert elicited.depth_penetration == volunteered.depth_penetration

    def test_unpromoted_dimension_bound_response_still_contributes(self):
        g = IntentGraph()

        breakdown = compute_fathom_breakdown(
            g,
            {"when"},
            dimension_bound_responses={
                "when": {
                    "raw_text": "I do not know the current French education system, please look it up",
                    "round": 1,
                    "node_ids": [],
                }
            },
        )

        assert breakdown.atom_count == 1
        assert breakdown.fathom_score > 0.0

    def test_verified_grounding_scores_higher_than_explicit_edge_only(self):
        g = IntentGraph()
        g.add_node(_make_node("n1", Dimension.WHY, "people think it is useless"))
        g.add_node(_make_node("n2", Dimension.WHAT, "questioning practical value"))
        g.add_edge(
            Edge(
                source="n1",
                target="n2",
                relation_type=RelationType.CAUSAL,
                source_type=EdgeSource.USER_EXPLICIT,
            )
        )

        explicit_only = compute_fathom_breakdown(g, {"why", "what"})
        verified = compute_fathom_breakdown(
            g,
            {"why", "what"},
            causal_hypotheses=[
                CausalHypothesis(
                    id="hyp1",
                    source_node_id="n1",
                    target_node_id="n2",
                    status="confirmed",
                )
            ],
        )

        assert verified.grounded_pair_count == 1
        assert verified.grounding_mass > explicit_only.grounding_mass
        assert verified.bedrock_grounding > explicit_only.bedrock_grounding
        assert verified.fathom_score > explicit_only.fathom_score


class TestInformationGain:
    def test_positive_gain(self):
        assert information_gain(0.3, 0.5) == 0.2

    def test_no_negative_gain(self):
        assert information_gain(0.5, 0.3) == 0.0


# ---------------------------------------------------------------------------
# Saturation gates
# ---------------------------------------------------------------------------

def _make_full_score_graph() -> IntentGraph:
    graph = IntentGraph()
    for dim in Dimension:
        for depth in range(3):
            graph.add_node(_make_node(f"{dim.value}_{depth}", dim))
    return graph


class TestSaturation:
    def test_saturation_is_diagnostic_only_even_for_high_score_graphs(self):
        graph = _make_full_score_graph()

        result = evaluate_fathom_gates(
            graph=graph,
            fathom_history=[],
            round_count=1,

            dimension_assessment={dim.value: "covered" for dim in Dimension},
        )

        assert result["fathom_score"] >= 0.80
        assert result["is_fathomed"] is False
        assert result["fathom_type"] == "not_fathomed"
        assert result["reason"] == "manual fathom required"

    def test_historical_max_is_preserved_without_auto_fathoming(self):
        graph = IntentGraph()

        result = evaluate_fathom_gates(
            graph=graph,
            fathom_history=[0.979],
            round_count=1,

            dimension_assessment={dim.value: "covered" for dim in Dimension},
        )

        assert result["fathom_score"] == 0.979
        assert result["is_fathomed"] is False
        assert result["fathom_type"] == "not_fathomed"
        assert result["reason"] == "manual fathom required"

    def test_breakdown_fields_are_exposed_in_gate_state(self):
        graph = IntentGraph()
        graph.add_node(_make_node("what_0", Dimension.WHAT))
        graph.add_node(_make_node("why_0", Dimension.WHY))

        result = evaluate_fathom_gates(
            graph=graph,
            fathom_history=[],
            round_count=2,

            dimension_assessment={
                "what": "covered",
                "why": "covered",
                "how": "missing",
                "who": "missing",
                "when": "missing",
                "where": "missing",
            },
            dimension_bound_responses={
                "why": {
                    "raw_text": "because practical value matters",
                    "round": 1,
                    "node_ids": ["why_0"],
                }
            },
        )

        assert "surface_coverage" in result
        assert "depth_penetration" in result
        assert "bedrock_grounding" in result
        assert "utility_mass" in result
        assert "creditable_dimensions" in result
        assert "blocked_secondary_dimensions" in result
        assert 0.0 <= result["surface_coverage"] <= 1.0
        assert 0.0 <= result["depth_penetration"] <= 1.0
        assert 0.0 <= result["bedrock_grounding"] <= 1.0
        assert "why" in result["creditable_dimensions"]

    def test_missing_dimension_can_remain_relevant_but_not_creditable(self):
        graph = IntentGraph()
        graph.add_node(
            Node(
                id="what_0",
                content="curriculum topic",
                dimension=Dimension.WHAT,
                secondary_dimensions=["how"],
            )
        )

        result = evaluate_fathom_gates(
            graph=graph,
            fathom_history=[],
            round_count=1,

            dimension_assessment={
                "what": "covered",
                "why": "missing",
                "how": "missing",
                "who": "missing",
                "when": "missing",
                "where": "missing",
            },
        )

        assert "how" in result["score_relevant_dimensions"]
        assert "how" not in result["creditable_dimensions"]
        assert "how" in result["blocked_secondary_dimensions"]


