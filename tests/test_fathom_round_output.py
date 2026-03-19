from __future__ import annotations

from ftg.fathom import FathomSession


def test_combine_response_formats_progress_tension_and_next_question():
    text = FathomSession._combine_response(
        {
            "response": "At first glance, this is not just looking up information, but making a judgment call.",
            "insight": "What will really change the conclusion is whether you value long-term use or short-term gains more.",
            "question": "If you sell it now, is the main reason to cash out, or are you worried the price will drop later?",
        }
    )

    # Response and insight are merged into one paragraph
    assert "judgment call" in text
    assert "long-term use" in text
    # Question appears as its own paragraph
    assert "main reason to cash out" in text
    assert 'Say "fathom" to compile, or "stop" to exit Fathom Mode.' in text


def test_combine_response_keeps_footer_when_question_is_empty():
    text = FathomSession._combine_response(
        {
            "response": "The information is sufficient, we can proceed directly to execution.",
            "insight": "Continuing to ask questions would not significantly change the execution result.",
            "question": "",
        }
    )

    assert "sufficient" in text
    assert 'Say "fathom" to compile, or "stop" to exit Fathom Mode.' in text
