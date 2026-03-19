from __future__ import annotations

from ftg.causal import CausalTracker
from ftg.graph import IntentGraph
from ftg.models import CausalHypothesis, Dimension, Node, NodeType


def _node(
    node_id: str,
    content: str,
    *,
    node_type: NodeType,
    dimension: Dimension,
) -> Node:
    return Node(
        id=node_id,
        content=content,
        raw_quote=content,
        node_type=node_type,
        dimension=dimension,
    )


def test_generate_hypotheses_dedupes_mirrored_pairs(monkeypatch):
    graph = IntentGraph()
    new_goal = _node(
        "goal",
        "The user wants to know the practical uses of these languages",
        node_type=NodeType.GOAL,
        dimension=Dimension.WHY,
    )
    existing_belief = _node(
        "belief",
        "The user considers these languages to be niche",
        node_type=NodeType.BELIEF,
        dimension=Dimension.WHAT,
    )
    graph.add_node(new_goal)
    graph.add_node(existing_belief)

    tracker = CausalTracker()
    monkeypatch.setattr(tracker, "_compute_ambiguity", lambda *args, **kwargs: 0.8)

    hypotheses = tracker.generate_hypotheses(
        new_nodes=[new_goal],
        graph=graph,
        current_round=2,
    )

    assert len(hypotheses) == 1
    pair = hypotheses[0]
    assert {pair.source_node_id, pair.target_node_id} == {"goal", "belief"}


def test_resolve_hypothesis_expires_pending_mirror():
    tracker = CausalTracker()
    tracker._hypotheses = [
        CausalHypothesis(
            id="forward",
            source_node_id="goal",
            target_node_id="belief",
            source_content="The user wants to know the practical uses of these languages",
            target_content="The user considers these languages to be niche",
            ambiguity_score=0.9,
            status="pending",
            created_at_round=2,
        ),
        CausalHypothesis(
            id="reverse",
            source_node_id="belief",
            target_node_id="goal",
            source_content="The user considers these languages to be niche",
            target_content="The user wants to know the practical uses of these languages",
            ambiguity_score=0.9,
            status="pending",
            created_at_round=2,
        ),
    ]

    tracker.resolve_hypothesis("forward", "confirmed")

    statuses = {h.id: h.status for h in tracker.hypotheses}
    assert statuses["forward"] == "confirmed"
    assert statuses["reverse"] == "expired"


def test_get_next_hypothesis_skips_verified_mirror_pair():
    tracker = CausalTracker()
    tracker._verified_pairs = {tuple(sorted(("goal", "belief")))}
    tracker._hypotheses = [
        CausalHypothesis(
            id="reverse",
            source_node_id="belief",
            target_node_id="goal",
            source_content="The user considers these languages to be niche",
            target_content="The user wants to know the practical uses of these languages",
            ambiguity_score=0.95,
            status="pending",
            created_at_round=2,
        ),
    ]

    assert tracker.get_next_hypothesis() is None
