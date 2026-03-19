"""
Basic usage example of Fathom-then-Generate.

This example uses a MOCK LLM (no API key needed) to demonstrate
the full fathom loop. Replace mock_llm_fn with a real provider
(e.g., make_openai_llm) for production use.
"""

import json
import sys

from ftg import Fathom, FathomedIntent, LLMRequest


# ---------------------------------------------------------------------------
# Mock LLM — returns canned JSON responses for demonstration
# ---------------------------------------------------------------------------

ROUND_COUNTER = 0


def mock_llm_fn(req: LLMRequest) -> str:
    """Mock LLM that returns deterministic JSON responses."""
    global ROUND_COUNTER

    if req.json_mode and "understanding engine" in req.system_prompt:
        ROUND_COUNTER += 1
        if ROUND_COUNTER == 1:
            return json.dumps({
                "task_type": "thinking",
                "nodes": [
                    {
                        "id": "current_job",
                        "content": "Currently working as a software engineer",
                        "raw_quote": "I'm a software engineer",
                        "confidence": 0.9,
                        "node_type": "fact",
                        "dimension": "what",
                        "secondary_dimensions": ["who"],
                    },
                    {
                        "id": "want_change",
                        "content": "Considering changing jobs for better pay",
                        "raw_quote": "should I change jobs",
                        "confidence": 0.8,
                        "node_type": "intent",
                        "dimension": "what",
                        "secondary_dimensions": ["why"],
                    },
                ],
                "edges": [
                    {
                        "source": "want_change",
                        "target": "current_job",
                        "relation_type": "dependency",
                        "source_type": "user_implied",
                    }
                ],
                "bias_updates": [],
                "dimension_assessment": {
                    "who": "covered_implicitly",
                    "what": "covered",
                    "why": "missing",
                    "when": "missing",
                    "where": "not_relevant",
                    "how": "missing",
                },
            })
        elif ROUND_COUNTER == 2:
            return json.dumps({
                "task_type": "thinking",
                "nodes": [
                    {
                        "id": "salary_low",
                        "content": "Current salary is below market rate",
                        "raw_quote": "my salary is too low",
                        "confidence": 0.85,
                        "node_type": "fact",
                        "dimension": "why",
                    },
                    {
                        "id": "growth_limited",
                        "content": "Limited career growth at current company",
                        "raw_quote": "not much room for growth",
                        "confidence": 0.7,
                        "node_type": "belief",
                        "dimension": "why",
                    },
                ],
                "edges": [
                    {
                        "source": "salary_low",
                        "target": "want_change",
                        "relation_type": "supports",
                        "source_type": "user_explicit",
                    }
                ],
                "bias_updates": [],
                "dimension_assessment": {
                    "who": "covered_implicitly",
                    "what": "covered",
                    "why": "covered",
                    "when": "missing",
                    "where": "not_relevant",
                    "how": "missing",
                },
            })
        elif ROUND_COUNTER == 3:
            return json.dumps({
                "task_type": "thinking",
                "nodes": [
                    {
                        "id": "timeline_3mo",
                        "content": "Want to transition within 3 months",
                        "raw_quote": "within 3 months ideally",
                        "confidence": 0.8,
                        "node_type": "constraint",
                        "dimension": "when",
                    },
                    {
                        "id": "approach_linkedin",
                        "content": "Plan to use LinkedIn and referrals",
                        "raw_quote": "through LinkedIn and asking friends",
                        "confidence": 0.75,
                        "node_type": "intent",
                        "dimension": "how",
                    },
                ],
                "edges": [],
                "bias_updates": [],
                "dimension_assessment": {
                    "who": "covered_implicitly",
                    "what": "covered",
                    "why": "covered",
                    "when": "covered",
                    "where": "not_relevant",
                    "how": "covered",
                },
            })
        else:
            return json.dumps({
                "task_type": "thinking",
                "nodes": [],
                "edges": [],
                "bias_updates": [],
                "dimension_assessment": {
                    "who": "covered", "what": "covered", "why": "covered",
                    "when": "covered", "where": "not_relevant", "how": "covered",
                },
            })

    if req.json_mode and "conversational partner" in req.system_prompt:
        return json.dumps({
            "response": "I understand you're considering a career change.",
            "insight": "When evaluating job opportunities, consider not just salary but total compensation including equity, benefits, and work-life balance.",
            "question": "What is the main reason you're considering changing jobs right now?",
            "target_dimension": "why",
            "target_gap": "motivation for job change",
            "target_types": ["belief", "value"],
        })

    return json.dumps({"verdict": "confirmed"})


# ---------------------------------------------------------------------------
# Mock dialogue — simulates user responses
# ---------------------------------------------------------------------------

RESPONSES = [
    "My salary is too low and there's not much room for growth here.",
    "I'd like to transition within 3 months ideally, through LinkedIn and asking friends.",
    "Yes, that looks correct.",
]
RESPONSE_IDX = 0


def mock_dialogue(question: str, insight=None) -> str:
    global RESPONSE_IDX
    print(f"\n{'='*60}")
    print(f"[FtG] {question}")
    if insight:
        print(f"[Insight] {insight}")

    if RESPONSE_IDX < len(RESPONSES):
        answer = RESPONSES[RESPONSE_IDX]
        RESPONSE_IDX += 1
    else:
        answer = "yes"

    print(f"[User] {answer}")
    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Fathom-then-Generate — Basic Usage Example")
    print("  (Using mock LLM — no API key needed)")
    print("=" * 60)

    fathom = Fathom(llm_fn=mock_llm_fn)
    session = fathom.start(
        "Should I change jobs? I'm a software engineer.",
        dialogue_fn=mock_dialogue,
    )
    result: FathomedIntent = session.run()

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"\nFathom Score: {result.fathom_score:.3f}")
    print(f"Fathom Type:  {result.fathom_type}")
    print(f"Task Type:    {result.task_type}")
    print(f"Rounds:       {result.rounds}")
    print(f"Nodes:        {len(result.nodes)}")
    print(f"Edges:        {len(result.edges)}")
    print(f"Biases:       {result.bias_flags or 'None detected'}")

    print(f"\n{'='*60}")
    print("  COMPILED PROMPT")
    print("=" * 60)
    print(result.compiled_prompt)

    print(f"\n{'='*60}")
    print("  MERMAID DIAGRAM")
    print("=" * 60)
    print(result.to_mermaid())


if __name__ == "__main__":
    main()
