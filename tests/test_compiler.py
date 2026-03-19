"""Tests for Prompt Compiler."""

from ftg.compiler import compile_intent_graph
from ftg.graph import IntentGraph
from ftg.models import (
    Dimension, Edge, EdgeSource, Node, NodeType, RelationType,
)

TASK_ANCHOR = "=== [1] Task Anchor ==="
USER_EXPR = "=== [2] User Expression & System Understanding ==="
CAUSAL_SECTION = "=== [3] User-Explicit Causal Relationships ==="
CONSTRAINT_SECTION = "=== [4] Constraints & Conflicts ==="
SYSTEM_INFER = "=== [5] System-Inferred Supplements ==="
ATTACHMENT_CONTEXT = "=== [0a] User-Provided Attachment Context ==="
ORIGINAL_REQUEST = "Original user request: "
USER_PREFIX = "User: "
UNDERSTANDING_PREFIX = "- Understanding: "
DIMENSION_PREFIX = "Dimension: "
BIAS_PREFIX = "Bias signal: "
LOW_CONF = "Low confidence: node confidence is 30%."
CONFLICT = "tension or conflict"
CAUSAL_3A = "[3a] User-Explicit Causal Relationships"
CAUSAL_3B = "[3b] Structural Dependencies"
NO_HARD_CAUSAL = "No user-explicit hard causal constraints found."
NO_ANCHORED_INFO = "No graph information that can be reliably anchored to the user's own words."
NO_SYSTEM_INFER = "No system-inferred supplements that cannot be anchored to the user's own words."
FRAME_BIAS = "Framing Effect"
FRENCH_PDF = "French curriculum PDF"
FRENCH_SUMMARY = "A French middle school course schedule document"
BASE_ON_ATTACHMENT = "Please answer based on the attachment"
OTHER_HIDDEN = "additional"


def _build_rich_graph() -> IntentGraph:
    g = IntentGraph()

    g.add_node(Node(
        id="goal_job",
        content="Find a new job with better pay",
        dimension=Dimension.WHAT,
        node_type=NodeType.GOAL,
        raw_quote="I want to find a new job",
    ))
    g.add_node(Node(
        id="reason",
        content="Current salary is below market",
        dimension=Dimension.WHY,
        node_type=NodeType.FACT,
        raw_quote="my salary is too low",
    ))
    g.add_node(Node(
        id="constraint_time",
        content="Must transition within 3 months",
        dimension=Dimension.WHEN,
        node_type=NodeType.CONSTRAINT,
        raw_quote="I need to switch within 3 months",
    ))
    g.add_node(Node(
        id="stakeholder",
        content="Family depends on stable income",
        dimension=Dimension.WHO,
        node_type=NodeType.FACT,
        raw_quote="my family depends on me",
    ))
    g.add_node(Node(
        id="method",
        content="Apply through LinkedIn and referrals",
        dimension=Dimension.HOW,
        node_type=NodeType.INTENT,
    ))
    g.add_node(Node(
        id="biased_node",
        content="My current company is terrible",
        dimension=Dimension.WHAT,
        node_type=NodeType.BELIEF,
        bias_flags=["framing_effect"],
        raw_quote="my company is terrible",
    ))
    g.add_node(Node(
        id="low_conf",
        content="Maybe consider freelancing",
        dimension=Dimension.HOW,
        node_type=NodeType.ASSUMPTION,
        confidence=0.3,
    ))

    g.add_edge(Edge(
        source="reason",
        target="goal_job",
        relation_type=RelationType.CAUSAL,
        source_type=EdgeSource.USER_EXPLICIT,
    ))
    g.add_edge(Edge(
        source="constraint_time",
        target="method",
        relation_type=RelationType.CONTRADICTION,
        source_type=EdgeSource.ALGORITHM_INFERRED,
    ))
    g.add_edge(Edge(
        source="stakeholder",
        target="goal_job",
        relation_type=RelationType.DEPENDENCY,
        source_type=EdgeSource.USER_IMPLIED,
    ))

    return g


class TestCompilerStructure:
    def test_all_sections_present(self):
        g = _build_rich_graph()
        output = compile_intent_graph(g, "I want to change jobs", "thinking")

        assert TASK_ANCHOR in output
        assert USER_EXPR in output
        assert "=== [2a]" not in output
        assert CAUSAL_SECTION in output
        assert CONSTRAINT_SECTION in output
        assert SYSTEM_INFER in output
        assert "=== [6]" not in output
        assert "=== [7]" not in output

    def test_task_anchor_keeps_original_request(self):
        g = _build_rich_graph()
        output = compile_intent_graph(g, "I want to change jobs", "thinking")

        assert f"{ORIGINAL_REQUEST}I want to change jobs" in output
        assert "response anchor" not in output
        assert "integration requirements" not in output

    def test_user_expression_section_groups_by_raw_quote(self):
        g = _build_rich_graph()
        output = compile_intent_graph(g, "change jobs", "thinking")

        assert f"{USER_PREFIX}I want to find a new job" in output
        assert f"{UNDERSTANDING_PREFIX}Find a new job with better pay" in output

    def test_causal_model_section(self):
        g = _build_rich_graph()
        output = compile_intent_graph(g, "change jobs", "thinking")

        assert "Current salary is below market -> causes -> Find a new job with better pay" in output

    def test_bias_notes_are_inlined_under_anchor(self):
        g = _build_rich_graph()
        output = compile_intent_graph(g, "change jobs", "thinking")

        assert f"{USER_PREFIX}my company is terrible" in output
        assert BIAS_PREFIX in output
        assert FRAME_BIAS in output

    def test_unanchored_low_confidence_information_moves_to_system_inference(self):
        g = _build_rich_graph()
        output = compile_intent_graph(g, "change jobs", "thinking")

        assert SYSTEM_INFER in output
        assert "Maybe consider freelancing" in output
        assert LOW_CONF in output

    def test_constraint_appears(self):
        g = _build_rich_graph()
        output = compile_intent_graph(g, "change jobs", "thinking")

        assert "Must transition within 3 months" in output

    def test_output_format_decision_removed(self):
        g = _build_rich_graph()
        output = compile_intent_graph(g, "change jobs", "thinking")

        assert "provide decision analysis" not in output
        assert "response format requirements" not in output

    def test_attachment_contexts_are_rendered_without_entering_graph(self):
        g = IntentGraph()
        output = compile_intent_graph(
            g,
            BASE_ON_ATTACHMENT,
            "general",
            attachment_contexts=[
                {
                    "label": FRENCH_PDF,
                    "summary": FRENCH_SUMMARY,
                    "raw_ref": "upload://course.pdf",
                }
            ],
        )

        assert ATTACHMENT_CONTEXT in output
        assert FRENCH_PDF in output
        assert "upload://course.pdf" in output

    def test_dimension_states_are_inlined_under_matching_anchor(self):
        g = IntentGraph()
        g.add_node(Node(
            id="no_feedback_received",
            content="Didn't get a clear reason for the missed promotion",
            dimension=Dimension.HOW,
            node_type=NodeType.FACT,
            raw_quote="no clear reason was given either",
        ))
        output = compile_intent_graph(
            g,
            "Is getting an MBA in 2026 still worth it?",
            "thinking",
            dimension_states={
                "how": {
                    "evidence_present": True,
                    "coverage_level": "partial",
                    "supporting_node_ids": ["no_feedback_received"],
                    "open_gap": "We know feedback was missing, but still lack the next-step path and execution plan",
                },
                "where": {
                    "evidence_present": False,
                    "coverage_level": "not_relevant",
                    "supporting_node_ids": [],
                    "open_gap": "",
                },
            },
        )

        assert "=== [2a]" not in output
        assert "User: no clear reason was given either" in output
        assert "Dimension states: " not in output


class TestCompilerEdgeCases:
    def test_empty_graph(self):
        g = IntentGraph()
        output = compile_intent_graph(g, "test", "general")
        assert TASK_ANCHOR in output
        assert USER_EXPR in output
        assert NO_ANCHORED_INFO in output
        assert "=== [7]" not in output

    def test_no_causal_edges_skips_section(self):
        g = IntentGraph()
        g.add_node(Node(id="n1", content="something", dimension=Dimension.WHAT))
        output = compile_intent_graph(g, "test", "general")
        assert CAUSAL_3A not in output
        assert "=== [3]" not in output

    def test_dependency_chains_are_capped_in_output(self):
        g = IntentGraph()
        for idx in range(3):
            g.add_node(Node(id=f"r{idx}", content=f"root {idx}", dimension=Dimension.WHY))
        for idx in range(3):
            g.add_node(Node(id=f"m1_{idx}", content=f"mid1 {idx}", dimension=Dimension.WHAT))
        for idx in range(3):
            g.add_node(Node(id=f"m2_{idx}", content=f"mid2 {idx}", dimension=Dimension.WHAT))
        g.add_node(Node(id="leaf", content="leaf", dimension=Dimension.HOW))

        for r_idx in range(3):
            for m1_idx in range(3):
                g.add_edge(Edge(
                    source=f"r{r_idx}",
                    target=f"m1_{m1_idx}",
                    relation_type=RelationType.DEPENDENCY,
                    source_type=EdgeSource.ALGORITHM_INFERRED,
                ))
        for m1_idx in range(3):
            for m2_idx in range(3):
                g.add_edge(Edge(
                    source=f"m1_{m1_idx}",
                    target=f"m2_{m2_idx}",
                    relation_type=RelationType.DEPENDENCY,
                    source_type=EdgeSource.ALGORITHM_INFERRED,
                ))
        for m2_idx in range(3):
            g.add_edge(Edge(
                source=f"m2_{m2_idx}",
                target="leaf",
                relation_type=RelationType.DEPENDENCY,
                source_type=EdgeSource.ALGORITHM_INFERRED,
            ))

        output = compile_intent_graph(g, "test", "thinking")

        assert CAUSAL_3B not in output
