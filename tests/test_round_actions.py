import json

from ftg import Fathom
from ftg.graph import IntentGraph
from ftg.models import Dimension, Node, NodeType
from ftg.questioner import generate_question


def test_generate_question_can_return_answer_now():
    graph = IntentGraph()
    graph.add_node(
        Node(
            id="goal",
            content="The user wants to know the practical uses of learning Latin and Greek",
            raw_quote="what is the use of learning these languages",
            node_type=NodeType.GOAL,
            dimension=Dimension.WHAT,
        )
    )

    def fake_llm(_req):
        return json.dumps(
            {
                "round_action": "answer_now",
                "ask_mode": "dimension",
                "response": "Here is a preliminary assessment.",
                "insight": "If we continue, we can explore further.",
                "question": "",
                "target_gap": "",
                "target_types": [],
                "hypothesis_id": "",
            },
            ensure_ascii=False,
        )

    result = generate_question(
        graph=graph,
        conversation_history="",
        task_type="thinking",
        llm_fn=fake_llm,
        round_count=1,
        complexity=0.4,
        waived_dimensions=set(),
        question_mode="dimension",
        target_dimension="what",
        root_question="what is the use of learning these languages",
    )

    assert result["round_action"] == "answer_now"
    assert result["question"] == ""


def test_fathom_round_relay_can_surface_answer_now():
    fathom = Fathom(llm_fn=lambda _req: "{}")
    session = fathom.start(user_input="I want to know what Latin is good for", dialogue_fn=lambda q, i=None: "")
    session._fathom_score = 0.42

    relay = session._make_round_relay("Current assessment: Here is a preliminary answer.", None, "answer_now")

    assert relay.action == "answer_now"
    assert "Current assessment" in relay.display
