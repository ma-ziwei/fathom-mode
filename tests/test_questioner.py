from __future__ import annotations

import json

from ftg.graph import IntentGraph
from ftg.models import CausalHypothesis, Dimension, Node, NodeType
from ftg.questioner import (
    RoundConstraints,
    RoundMaterials,
    RoundPacket,
    _build_round_packet_prompt,
    generate_question,
)


def _llm_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _node(
    node_id: str,
    content: str,
    *,
    dimension: Dimension,
    node_type: NodeType = NodeType.FACT,
    bias_flags: list[str] | None = None,
) -> Node:
    return Node(
        id=node_id,
        content=content,
        raw_quote=content,
        dimension=dimension,
        node_type=node_type,
        bias_flags=bias_flags or [],
    )


def test_generate_question_preserves_specific_llm_question():
    result = generate_question(
        graph=IntentGraph(),
        conversation_history="",
        task_type="general",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Which restaurants are better suited for hosting guests?",
                "target_dimension": "what",
                "target_gap": "want to narrow down to the hosting scenario first",
                "target_types": ["restaurant", "hosting"],
            }
        ),
        round_count=1,
        target_dimension="what",
    )

    assert result["question"] == "Which restaurants are better suited for hosting guests?"
    assert result["target_gap"] == "want to narrow down to the hosting scenario first"
    assert result["target_types"] == ["restaurant", "hosting"]


def test_generate_question_rewrites_generic_image_question_with_scene_specific_fallback():
    result = generate_question(
        graph=IntentGraph(),
        conversation_history="",
        task_type="creation",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Can you tell me more about that?",
                "target_dimension": "who",
                "target_gap": "",
                "target_types": ["character", "figure"],
            }
        ),
        round_count=1,
        target_dimension="who",
    )

    # With unified fallback, generic LLM question is replaced by gap-based or dimension fallback
    assert "can you tell me more" not in result["question"].lower()
    assert result["question"].endswith("?")


def test_generate_question_uses_target_gap_for_generic_decision_question():
    result = generate_question(
        graph=IntentGraph(),
        conversation_history="",
        task_type="thinking",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Could you elaborate on that?",
                "target_dimension": "why",
                "target_gap": "whether you care more about saving money or the experience when hosting friends",
                "target_types": ["cost", "experience"],
            }
        ),
        round_count=1,
        target_dimension="why",
    )

    assert "saving money" in result["question"]
    assert result["question"].endswith("?")
    assert "could you elaborate" not in result["question"].lower()


def test_dimension_round_no_longer_returns_focus_governor_metadata():
    graph = IntentGraph()
    graph.add_node(_node("n1", "Uses of Latin and Greek courses in French schools", dimension=Dimension.WHAT))
    graph.add_node(_node("n2", "Making educational decisions for the child later", dimension=Dimension.WHY, node_type=NodeType.GOAL))
    graph.add_node(
        _node(
            "n3",
            "These languages seem niche and their value is unclear",
            dimension=Dimension.WHAT,
            node_type=NodeType.BELIEF,
            bias_flags=["framing_effect"],
        )
    )

    result = generate_question(
        graph=graph,
        conversation_history="Q: Which part would you like to explore first?\nA: I want to first understand what these courses are actually useful for.",
        task_type="thinking",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Could you elaborate on that?",
                "target_dimension": "why",
                "target_gap": "",
                "target_types": [],
            }
        ),
        round_count=2,
        target_dimension="why",
        dimension_semantics={"why": "the practical significance of these courses for the child's growth and educational decisions"},
        latest_user_response="I mainly want to know what these courses are actually useful for.",
        root_question="What were the main purposes of learning Latin in middle school and Greek in high school in Western European countries after WWI",
        previous_current_answer_object="Uses of Latin and Greek courses in French schools",
        previous_focus_layer="task_object",
    )

    assert "current_answer_object" not in result
    assert "focus_layer" not in result
    assert "handoff_applied" not in result
    assert result["target_gap"] == "the practical significance of these courses for the child's growth and educational decisions"


def test_dimension_round_still_tracks_target_gap_without_focus_handoff():
    graph = IntentGraph()
    graph.add_node(_node("n1", "Uses of Latin and Greek courses in French schools", dimension=Dimension.WHAT))
    graph.add_node(
        _node(
            "n2",
            "These languages seem niche, maybe they have no practical value",
            dimension=Dimension.WHAT,
            node_type=NodeType.BELIEF,
            bias_flags=["framing_effect"],
        )
    )

    result = generate_question(
        graph=graph,
        conversation_history="Q: Which part would you like to explore first?\nA: I actually care more about whether these niche languages have any value.",
        task_type="thinking",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Could you elaborate on that?",
                "target_dimension": "why",
                "target_gap": "",
                "target_types": [],
            }
        ),
        round_count=2,
        target_dimension="why",
        dimension_semantics={"why": "the real concern behind this judgment about the value of these courses"},
        latest_user_response="What I actually care about is whether these niche languages have any value at all.",
        root_question="What were the main purposes of learning Latin in middle school and Greek in high school in Western European countries after WWI",
        previous_current_answer_object="Uses of Latin and Greek courses in French schools",
        previous_focus_layer="task_object",
    )

    assert "focus_layer" not in result
    assert "current_answer_object" not in result
    assert result["target_gap"] == "the real concern behind this judgment about the value of these courses"


def test_redirect_confirm_question_is_shorter_and_not_raw_quote_dump():
    result = generate_question(
        graph=IntentGraph(),
        conversation_history="",
        task_type="thinking",
        llm_fn=lambda req: "{}",
        round_count=2,
        question_mode="redirect_confirm",
        redirect_context={
            "original_source": "the user is choosing between pour-over coffee and instant coffee for hosting friends",
            "original_target": "using coffee mainly for hosting friends, a social scenario",
            "user_response": "if it's cheap I'll buy one, I can also drink it myself later; if it's expensive forget it, I could just buy good instant coffee to serve friends",
            "hypothesis_id": "h-1",
        },
    )

    assert result["question_mode"] == "causal"
    assert "you mentioned" not in result["question"].lower()
    assert "Do you agree" in result["question"] or "key factor" in result["question"]
    assert len(result["question"]) < 200


def test_causal_question_rewrites_generic_fallback_with_explicit_relation():
    hypothesis = CausalHypothesis(
        id="h-2",
        source_node_id="a",
        target_node_id="b",
        source_content="buying pour-over equipment",
        target_content="better suited for hosting friends",
    )

    result = generate_question(
        graph=IntentGraph(),
        conversation_history="",
        task_type="thinking",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Can you tell me more about why these two are related?",
            }
        ),
        round_count=2,
        question_mode="causal",
        causal_hypothesis=hypothesis,
    )

    # Simple causal question uses compact text; verify source/target are both present
    assert "pour-over" in result["question"].lower()
    assert "suited for host" in result["question"].lower() or "hosting" in result["question"].lower()
    assert "why these two are related" not in result["question"]


def test_generate_question_first_round_keeps_progress_and_tension():
    result = generate_question(
        graph=IntentGraph(),
        conversation_history="",
        task_type="general",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Can you tell me more about that?",
                "target_dimension": "what",
                "target_gap": "whether you want to know the main uses or the historical reasons for learning these two languages",
                "target_types": ["uses", "historical background"],
            }
        ),
        round_count=0,
        target_dimension="what",
    )

    assert result["response"]
    assert result["insight"]
    assert "whether you want" in result["insight"] or "main uses" in result["insight"]
    assert "can you tell me more" not in result["question"].lower()


def test_causal_question_fills_progress_and_tension_when_llm_is_empty():
    hypothesis = CausalHypothesis(
        id="h-3",
        source_node_id="a",
        target_node_id="b",
        source_content="whether to buy Switch 2",
        target_content="buying Switch 2 is mainly for playing party games with friends at gatherings",
    )

    result = generate_question(
        graph=IntentGraph(),
        conversation_history="",
        task_type="thinking",
        llm_fn=lambda req: "{}",
        round_count=2,
        question_mode="causal",
        causal_hypothesis=hypothesis,
    )

    assert result["response"]
    assert result["insight"]
    assert "causal" in result["insight"].lower() or "causation" in result["insight"].lower() or "correlation" in result["insight"].lower()


def test_dimension_round_system_prompt_no_longer_injects_focus_protocol():
    graph = IntentGraph()
    graph.add_node(_node("n1", "Latin courses in French schools", dimension=Dimension.WHAT))
    graph.add_node(_node("n2", "Making educational decisions for the child later", dimension=Dimension.WHY, node_type=NodeType.GOAL))
    graph.add_node(_node("n3", "Somewhat uneasy about the French education system", dimension=Dimension.WHY, node_type=NodeType.BELIEF))

    captured: dict[str, str] = {}

    def llm_fn(req):
        captured["system_prompt"] = req.system_prompt
        return _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Let me first confirm: is your main priority right now to understand the practical role of these courses in the French education system?",
                "target_dimension": "what",
                "target_gap": "the practical role of these courses in the French education system",
                "target_types": ["course role"],
            }
        )

    generate_question(
        graph=graph,
        conversation_history="",
        task_type="thinking",
        llm_fn=llm_fn,
        round_count=0,
        target_dimension="what",
        dimension_semantics={"what": "the practical role of these courses in the French education system"},
    )

    assert "explicit protocol for this round" not in captured["system_prompt"]
    assert "current understanding snapshot" not in captured["system_prompt"]
    assert "current structural tension" not in captured["system_prompt"]
    assert "current primary gap" not in captured["system_prompt"]


def test_dimension_round_keeps_progress_and_tension_without_focus_plan():
    graph = IntentGraph()
    graph.add_node(_node("n1", "Latin courses in French schools", dimension=Dimension.WHAT))
    graph.add_node(_node("n2", "Making educational decisions for the child later", dimension=Dimension.WHY, node_type=NodeType.GOAL))
    graph.add_node(_node("n3", "Somewhat uneasy about the French education system", dimension=Dimension.WHY, node_type=NodeType.BELIEF))

    result = generate_question(
        graph=graph,
        conversation_history="Q: Which part do you care about more?\nA: I mainly want to make decisions for the child later.",
        task_type="thinking",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Could you elaborate on that?",
                "target_dimension": "why",
                "target_gap": "",
                "target_types": [],
            }
        ),
        round_count=1,
        target_dimension="why",
        dimension_semantics={"why": "what you really want to judge based on this"},
        latest_user_response="I mainly want to make decisions for the child later.",
    )

    assert result["response"]
    assert "zeroing in on" in result["response"].lower() or "what you really want" in result["response"]
    assert result["insight"]
    assert "what you really want to judge" in result["question"]
    assert "current_read_focus" not in result
    assert result["target_gap"] == "what you really want to judge based on this"


def test_causal_question_preserves_source_to_target_direction_for_analysis():
    hypothesis = CausalHypothesis(
        id="h-4",
        source_node_id="belief",
        target_node_id="goal",
        source_content="the user judges Latin and Greek as 'useless languages'",
        target_content="the user wants to understand the practical uses and value of children and grandchildren learning Latin and Greek",
    )

    result = generate_question(
        graph=IntentGraph(),
        conversation_history="",
        task_type="thinking",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Why do you think these two are related?",
            }
        ),
        round_count=4,
        question_mode="causal",
        causal_hypothesis=hypothesis,
    )

    # Simple causal question uses compact text; verify source appears before target
    assert "latin and greek" in result["question"].lower() or "judges" in result["question"].lower()
    assert "understand" in result["question"].lower() or "wants to" in result["question"].lower()
    assert "directly affects" in result["question"] or "coincidentally" in result["question"]


def test_dimension_round_result_no_longer_carries_task_object_metadata():
    graph = IntentGraph()
    graph.add_node(
        _node(
            "root",
            "The problem the user wants to solve: main purposes of classical language education in Western European countries after WWI",
            dimension=Dimension.WHAT,
            node_type=NodeType.INTENT,
        )
    )
    graph.add_node(
        _node(
            "family",
            "The user's children and grandchildren live in France",
            dimension=Dimension.WHO,
        )
    )
    graph.add_node(
        _node(
            "task",
            "Uses of Latin and Greek courses in French schools",
            dimension=Dimension.WHAT,
        )
    )

    result = generate_question(
        graph=graph,
        conversation_history="",
        task_type="thinking",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Could you first confirm the practical uses of these courses?",
                "target_dimension": "what",
                "target_gap": "the practical uses of these courses",
                "target_types": ["course uses"],
            }
        ),
        round_count=1,
        target_dimension="what",
        latest_user_response="I want to know the practical uses of these courses.",
    )

    assert "current_answer_object" not in result
    assert result["target_gap"] == "the practical uses of these courses"


def test_dimension_round_follows_latest_goal_via_target_gap_without_theme_metadata():
    graph = IntentGraph()
    graph.add_node(
        _node(
            "task",
            "Uses of Latin and Greek courses in French schools",
            dimension=Dimension.WHAT,
        )
    )
    graph.add_node(
        _node(
            "goal",
            "What the user currently most wants to know: what use is there in learning these niche languages",
            dimension=Dimension.WHY,
            node_type=NodeType.GOAL,
        )
    )

    result = generate_question(
        graph=graph,
        conversation_history="Q: Which part would you like to discuss first?\nA: My children and grandchildren live in France.",
        task_type="thinking",
        llm_fn=lambda req: _llm_json(
            {
                "response": "",
                "insight": "",
                "question": "Could you elaborate on that?",
                "target_dimension": "why",
                "target_gap": "what use is there in learning these niche languages",
                "target_types": [],
            }
        ),
        round_count=2,
        target_dimension="why",
        latest_user_response="What use is there in learning these niche languages?",
        root_question="What were the main purposes of learning Latin in middle school and Greek in high school in Western European countries after WWI",
        previous_current_answer_object="Uses of Latin and Greek courses in French schools",
        previous_focus_layer="task_object",
    )

    assert "focus_layer" not in result
    assert "current_answer_object" not in result
    assert "root question" not in result["response"].lower()
    assert "niche languages" in result["question"]


def test_round_packet_prompt_surfaces_dimension_states_for_partial_coverage():
    packet = RoundPacket(
        constraints=RoundConstraints(task_type="thinking", round_count=2, allowed_modes=["dimension"]),
        materials=RoundMaterials(
            root_question="Is it still worth doing an MBA in 2026?",
            latest_user_response="no clear reason given",
            graph_summary="Covered dimensions: Who(1), What(1), Why(1)",
            dimension_states={
                "how": {
                    "evidence_present": True,
                    "coverage_level": "partial",
                    "supporting_node_ids": ["no_feedback_received"],
                    "open_gap": "already know feedback is missing, but still lacking next-step path and execution plan",
                }
            },
        ),
    )

    prompt = _build_round_packet_prompt(packet)

    assert "[information_gaps]" in prompt
    assert "How: already know feedback is missing, but still lacking next-step path and execution plan" in prompt
    # Internal metadata should NOT leak into the prompt
    assert "coverage_level" not in prompt
    assert "supporting_node_ids" not in prompt
    assert "evidence_present" not in prompt
