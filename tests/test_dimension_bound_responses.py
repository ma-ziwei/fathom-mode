from __future__ import annotations

import json

from ftg.compiler import compile_intent_graph
from ftg.graph import IntentGraph
from ftg.questioner import generate_question


def _llm_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_generate_question_includes_dimension_bound_response_and_skips_same_dimension():
    captured_prompt: dict[str, str] = {}

    def fake_llm(req):
        captured_prompt["user_prompt"] = req.user_prompt
        return _llm_json(
            {
                "ask_mode": "dimension",
                "response": "",
                "insight": "",
                "question": "Let's first talk about the purpose of these courses",
                "target_gap": "the main purpose of these courses",
                "target_types": ["course purpose"],
            }
        )

    result = generate_question(
        graph=IntentGraph(),
        conversation_history="",
        task_type="general",
        llm_fn=fake_llm,
        round_count=2,
        target_dimension="when",
        dimension_bound_responses={
            "when": {
                "raw_text": "I don't know the current French education system, please look it up for me",
                "round": 1,
            }
        },
    )

    assert "already_answered" in captured_prompt["user_prompt"]
    assert "I don't know the current French education system, please look it up for me" in captured_prompt["user_prompt"]
    assert result["target_dimension"] != "when"


def test_compile_intent_graph_surfaces_dimension_bound_responses():
    output = compile_intent_graph(
        IntentGraph(),
        "What were Latin in middle school and Greek in high school mainly used for in Western European countries after WWI",
        "thinking",
        dimension_bound_responses={
            "when": {
                "raw_text": "I don't know the current French education system, please look it up for me",
            },
            "who": {
                "raw_text": "This is a scene diagram, there are no people",
            },
        },
    )

    assert "=== [2] User Expression & System Understanding ===" in output
    assert "direct response to a dimension follow-up" not in output
    assert "I don't know the current French education system" not in output
    assert "This is a scene diagram, there are no people" not in output
