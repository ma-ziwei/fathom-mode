from __future__ import annotations

from types import SimpleNamespace

import ftg.fathom as fathom
from ftg.models import Dimension, Node, NodeType


def test_fathom_session_uses_understanding_context_for_understanding_and_execution_context_for_compile(monkeypatch):
    seen: dict[str, str] = {}

    def fake_extract(
        *,
        user_text,
        graph,
        llm_fn,
        embed_fn,
        conversation_context,
        task_type,
        causal_markers_fn,
        match_markers_fn,
        user_context="",
        include_clarification_hints=False,
    ):
        seen["extract"] = user_context
        return [], [], "general", {}, {}, {}, {}

    def fake_generate_round_output(packet, llm_fn):
        seen["question"] = packet.materials.user_context
        return {"question": "Q", "insight": None, "response": "R", "ask_mode": "dimension"}

    def fake_compile_intent_graph(
        graph,
        user_input,
        task_type,
        user_context="",
        dimension_bound_responses=None,
        dimension_states=None,
        attachment_contexts=None,
        **kwargs,
    ):
        seen["compile"] = user_context
        return "compiled prompt"

    monkeypatch.setattr(fathom, "extract", fake_extract)
    monkeypatch.setattr(fathom, "generate_round_output", fake_generate_round_output)
    monkeypatch.setattr(fathom, "compile_intent_graph", fake_compile_intent_graph)
    monkeypatch.setattr(fathom, "find_target_dimension", lambda *args, **kwargs: "what")
    monkeypatch.setattr(
        fathom,
        "evaluate_fathom_gates",
        lambda **kwargs: {
            "is_fathomed": False,
            "fathom_score": 0.25,
            "fathom_type": "not_fathomed",
            "reason": "keep asking",
        },
    )

    factory = fathom.Fathom(llm_fn=lambda req: "")
    session = factory.start(
        user_input="compare these options",
        dialogue_fn=lambda q, insight=None: "",
        understanding_context="[understanding]",
        execution_context="[execution]",
    )

    session._extract_and_update("compare these options")
    session._generate_next_question()
    compiled = session.compile()

    assert seen == {
        "extract": "[understanding]",
        "question": "[understanding]",
        "compile": "[execution]",
    }
    assert compiled.compiled_prompt == "compiled prompt"


def test_fathom_attachment_context_round_trip_and_compile(monkeypatch):
    seen: dict[str, object] = {}

    def fake_compile_intent_graph(
        graph,
        user_input,
        task_type,
        user_context="",
        dimension_bound_responses=None,
        dimension_states=None,
        attachment_contexts=None,
        **kwargs,
    ):
        seen["attachments"] = attachment_contexts
        return "compiled prompt"

    monkeypatch.setattr(fathom, "compile_intent_graph", fake_compile_intent_graph)

    factory = fathom.Fathom(llm_fn=lambda req: "")
    session = factory.start(
        user_input="Please answer based on the attachment",
        dialogue_fn=lambda q, insight=None: "",
    )
    session.add_attachment_context(
        label="French curriculum PDF",
        summary="A French curriculum description",
        raw_ref="upload://course.pdf",
    )

    state = session.to_state()
    restored = fathom.FathomSession.from_state(state, llm_fn=lambda req: "")
    compiled = restored.compile()

    assert compiled.compiled_prompt == "compiled prompt"
    assert restored._attachment_score_bonus == 0.0
    assert seen["attachments"] == [
        {
            "label": "French curriculum PDF",
            "summary": "A French curriculum description",
            "raw_ref": "upload://course.pdf",
            "metadata": {},
        }
    ]


def test_fathom_binds_latest_user_reply_to_pending_dimension(monkeypatch):
    def fake_extract(
        *,
        user_text,
        graph,
        llm_fn,
        embed_fn,
        conversation_context,
        task_type,
        causal_markers_fn,
        match_markers_fn,
        user_context="",
        include_clarification_hints=False,
    ):
        return [
            Node(
                id="lookup_gap",
                content="User does not understand the current French education system",
                raw_quote=user_text,
                dimension=Dimension.WHAT,
                node_type=NodeType.CONSTRAINT,
            )
        ], [], "general", {}, {}, {}, {}

    monkeypatch.setattr(fathom, "extract", fake_extract)
    monkeypatch.setattr(
        fathom,
        "evaluate_fathom_gates",
        lambda **kwargs: {
            "is_fathomed": False,
            "fathom_score": 0.25,
            "fathom_type": "not_fathomed",
            "reason": "keep asking",
        },
    )

    factory = fathom.Fathom(llm_fn=lambda req: "")
    session = factory.start(
        user_input="test question",
        dialogue_fn=lambda q, insight=None: "",
    )
    session._pending_dimension = "when"
    session._last_question_mode = "dimension"
    session._pending_question = "Which period of the French education system?"

    session._advance("I don't understand the current French education system, please look it up for me", ask_question=False)

    assert session._dimension_bound_responses["when"]["raw_text"] == "I don't understand the current French education system, please look it up for me"
    assert session._dimension_bound_responses["when"]["node_ids"] == ["lookup_gap"]


def test_fathom_relay_command_fathom_compiles_current_session(monkeypatch):
    monkeypatch.setattr(
        fathom.FathomSession,
        "compile",
        lambda self: SimpleNamespace(compiled_prompt="compiled prompt", task_type="thinking"),
    )

    factory = fathom.Fathom(llm_fn=lambda req: "")
    session = factory.start(
        user_input="test question",
        dialogue_fn=lambda q, insight=None: "",
    )
    session._fathom_score = 0.42

    response = session.relay("fathom")

    assert response.action == "review"
    assert response.compiled_prompt == "compiled prompt"
    assert session._phase == "compiled_review"
    assert session._fathom_type == "not_fathomed"


def test_fathom_relay_review_executes_after_confirmation(monkeypatch):
    monkeypatch.setattr(
        fathom.FathomSession,
        "compile",
        lambda self: SimpleNamespace(compiled_prompt="compiled prompt", task_type="thinking"),
    )

    factory = fathom.Fathom(llm_fn=lambda req: "")
    session = factory.start(
        user_input="test question",
        dialogue_fn=lambda q, insight=None: "",
    )
    session._fathom_score = 0.42

    session.relay("fathom")
    response = session.relay("execute")

    assert response.action == "execute"
    assert response.compiled_prompt == "compiled prompt"
    assert session._phase == "compiled"
    assert session._fathom_type == "manual"


def test_fathom_relay_command_stop_stops_current_session():
    factory = fathom.Fathom(llm_fn=lambda req: "")
    session = factory.start(
        user_input="test question",
        dialogue_fn=lambda q, insight=None: "",
    )
    session._fathom_score = 0.31

    response = session.relay("stop")

    assert response.action == "stop"
    assert session._phase == "stopped"
    assert session._pending_round_action == "stop"
