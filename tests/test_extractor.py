import json

from ftg.extractor import (
    _build_extraction_system_prompt,
    extract,
    validate_raw_quote,
)
from ftg.graph import IntentGraph
from ftg.models import Dimension, Node, NodeType


def _llm_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_validate_raw_quote_accepts_light_paraphrase():
    node = Node(
        id="sell_memory",
        content="User is considering whether to sell their RAM sticks",
        raw_quote="sell my RAM sticks",
        node_type=NodeType.GOAL,
        dimension=Dimension.WHAT,
    )
    assert validate_raw_quote(node, "Should I sell my RAM sticks?") is True


def test_extract_rescues_root_anchor_when_initial_decision_round_has_no_nodes():
    graph = IntentGraph()

    def llm_fn(_req):
        return _llm_json(
            {
                "task_type": "general",
                "nodes": [],
                "edges": [],
                "dimension_assessment": {
                    "who": "missing",
                    "what": "missing",
                    "why": "missing",
                    "when": "missing",
                    "where": "missing",
                    "how": "missing",
                },
            }
        )

    nodes, edges, task_type, dim_assessment, dim_states, dim_semantics, hints = extract(
        user_text="Should I sell my RAM sticks?",
        graph=graph,
        llm_fn=llm_fn,
    )

    assert task_type == "general"
    assert edges == []
    assert dim_states["what"]["coverage_level"] == "partial"
    assert dim_semantics == {}
    assert hints == {}
    assert any(node.node_type == NodeType.GOAL for node in nodes)
    assert any("sell" in node.content and "RAM sticks" in node.content for node in nodes)


def test_extract_does_not_add_duplicate_root_anchor_when_llm_already_returns_goal():
    graph = IntentGraph()

    def llm_fn(_req):
        return _llm_json(
            {
                "task_type": "general",
                "nodes": [
                    {
                        "id": "sell_memory",
                        "content": "User is considering whether to sell their RAM sticks",
                        "raw_quote": "sell my RAM sticks",
                        "confidence": 0.9,
                        "node_type": "goal",
                        "dimension": "what",
                        "secondary_dimensions": ["how"],
                    }
                ],
                "edges": [],
                "dimension_assessment": {
                    "who": "missing",
                    "what": "covered",
                    "why": "missing",
                    "when": "missing",
                    "where": "missing",
                    "how": "missing",
                },
            }
        )

    nodes, *_ = extract(
        user_text="Should I sell my RAM sticks?",
        graph=graph,
        llm_fn=llm_fn,
    )

    assert len(nodes) == 1
    assert nodes[0].id == "sell_memory"


def test_extract_rescues_root_anchor_for_non_punctuated_interrogative_query():
    graph = IntentGraph()

    def llm_fn(_req):
        return _llm_json(
            {
                "task_type": "thinking",
                "nodes": [],
                "edges": [],
                "dimension_assessment": {
                    "who": "missing",
                    "what": "missing",
                    "why": "missing",
                    "when": "missing",
                    "where": "missing",
                    "how": "missing",
                },
            }
        )

    nodes, edges, task_type, dim_assessment, dim_states, dim_semantics, hints = extract(
        user_text="In Western European countries after WWI which aspects were Latin and Greek mainly used for in middle and high school",
        graph=graph,
        llm_fn=llm_fn,
    )

    assert task_type == "thinking"
    assert edges == []
    assert dim_states["what"]["coverage_level"] == "partial"
    assert dim_semantics == {}
    assert hints == {}
    assert any(node.node_type == NodeType.INTENT for node in nodes)
    assert any("User's question:" in node.content for node in nodes)


def test_initial_round_prompt_prioritizes_root_anchor_over_advisory_hints():
    prompt = _build_extraction_system_prompt(initial_round=True, hard_causal_edges=[])

    assert "Initial round / empty graph priority rules:" in prompt
    assert "you must prioritize keeping at least one intent or goal node that represents the main task" in prompt
    assert "clarification_hints requirements:" not in prompt


def test_initial_round_prompt_can_opt_into_clarification_hints():
    prompt = _build_extraction_system_prompt(
        initial_round=True,
        hard_causal_edges=[],
        include_clarification_hints=True,
    )

    assert "clarification_hints requirements:" in prompt


def test_extract_rescues_followup_question_as_goal_when_llm_returns_no_nodes():
    graph = IntentGraph()
    graph.add_node(
        Node(
            id="root_question",
            content="User's question: In Western European countries after WWI which aspects were Latin and Greek mainly used for",
            raw_quote="In Western European countries after WWI which aspects were Latin and Greek mainly used for",
            node_type=NodeType.INTENT,
            dimension=Dimension.WHAT,
        )
    )

    def llm_fn(_req):
        return _llm_json(
            {
                "task_type": "general",
                "nodes": [],
                "edges": [],
                "dimension_assessment": {
                    "who": "missing",
                    "what": "missing",
                    "why": "missing",
                    "when": "missing",
                    "where": "missing",
                    "how": "missing",
                },
            }
        )

    nodes, edges, task_type, _, _, _, _ = extract(
        user_text="What is the use of them learning these niche languages?",
        graph=graph,
        llm_fn=llm_fn,
    )

    assert task_type == "general"
    assert edges
    assert len(nodes) == 1
    assert nodes[0].node_type == NodeType.GOAL
    assert nodes[0].dimension == Dimension.WHY
    assert "What is the use of them learning these niche languages" in nodes[0].content


def test_extract_dimension_states_keep_partial_coverage_visible():
    graph = IntentGraph()

    def llm_fn(_req):
        return _llm_json(
            {
                "task_type": "thinking",
                "nodes": [
                    {
                        "id": "no_feedback_received",
                        "content": "After the promotion was denied, the user received no clear reason or feedback",
                        "raw_quote": "no clear reason given",
                        "confidence": 0.95,
                        "node_type": "fact",
                        "dimension": "how",
                        "secondary_dimensions": ["what"],
                    }
                ],
                "edges": [],
                "dimension_assessment": {
                    "who": "missing",
                    "what": "missing",
                    "why": "missing",
                    "when": "missing",
                    "where": "missing",
                    "how": "missing",
                },
                "dimension_states": {
                    "who": {"evidence_present": False, "coverage_level": "none", "supporting_node_ids": [], "open_gap": ""},
                    "what": {"evidence_present": False, "coverage_level": "none", "supporting_node_ids": [], "open_gap": ""},
                    "why": {"evidence_present": False, "coverage_level": "none", "supporting_node_ids": [], "open_gap": ""},
                    "when": {"evidence_present": False, "coverage_level": "none", "supporting_node_ids": [], "open_gap": ""},
                    "where": {"evidence_present": False, "coverage_level": "not_relevant", "supporting_node_ids": [], "open_gap": ""},
                    "how": {
                        "evidence_present": True,
                        "coverage_level": "partial",
                        "supporting_node_ids": ["no_feedback_received"],
                        "open_gap": "Feedback is missing, still need next steps and execution plan",
                    },
                },
            }
        )

    nodes, edges, task_type, dim_assessment, dim_states, _, _ = extract(
        user_text="no clear reason given",
        graph=graph,
        llm_fn=llm_fn,
    )

    assert task_type == "thinking"
    assert edges == []
    assert len(nodes) == 1
    assert dim_assessment["how"] == "covered_implicitly"
    assert dim_states["how"]["coverage_level"] == "partial"
    assert dim_states["how"]["evidence_present"] is True
    assert dim_states["how"]["supporting_node_ids"] == ["no_feedback_received"]
    assert "Feedback is missing" in dim_states["how"]["open_gap"]


def test_extract_builds_partial_dimension_state_from_nodes_when_llm_only_returns_missing():
    graph = IntentGraph()

    def llm_fn(_req):
        return _llm_json(
            {
                "task_type": "thinking",
                "nodes": [
                    {
                        "id": "no_feedback_received",
                        "content": "After the promotion was denied, the user received no clear reason or feedback",
                        "raw_quote": "no clear reason given",
                        "confidence": 0.95,
                        "node_type": "fact",
                        "dimension": "how",
                        "secondary_dimensions": [],
                    }
                ],
                "edges": [],
                "dimension_assessment": {
                    "who": "missing",
                    "what": "missing",
                    "why": "missing",
                    "when": "missing",
                    "where": "missing",
                    "how": "missing",
                },
                "dimension_semantics": {
                    "how": "Feedback is missing, but next concrete steps are still unknown",
                },
            }
        )

    _, _, _, dim_assessment, dim_states, _, _ = extract(
        user_text="no clear reason given",
        graph=graph,
        llm_fn=llm_fn,
    )

    assert dim_assessment["how"] == "covered_implicitly"
    assert dim_states["how"]["coverage_level"] == "partial"
    assert dim_states["how"]["evidence_present"] is True
    assert dim_states["how"]["supporting_node_ids"] == ["no_feedback_received"]
