from __future__ import annotations

from types import SimpleNamespace

from ftg import Fathom, FathomSession, LLMRequest
from ftg.backend import CompositeBackend, FunctionBackend
from ftg.contrib.openclaw import OpenClawBackend


def test_openclaw_backend_uses_phase_scoped_session_keys():
    calls: list[tuple[str, str]] = []

    backend = OpenClawBackend(
        chat_runner=lambda session_key, message: calls.append((session_key, message)) or "{}",
    )
    req = LLMRequest(system_prompt="SYSTEM", user_prompt="USER", json_mode=True)

    extract = backend.make_extract_llm_fn("sess123")
    question = backend.make_question_llm_fn("sess123")
    execute = backend.make_execute_llm_fn("sess123")

    extract(req)
    extract(req)
    question(req)
    question(req)
    execute(req)

    assert calls[0][0].startswith("ftg-extract-sess123-")
    assert calls[1][0].startswith("ftg-extract-sess123-")
    assert calls[0][0] != calls[1][0]
    assert calls[2][0] == "ftg-session-sess123"
    assert calls[3][0] == "ftg-session-sess123"
    assert calls[4][0] == "ftg-session-sess123"
    assert "Return strict JSON only." in calls[0][1]


def test_composite_backend_uses_phase_override_without_extra_provider_config():
    calls: list[str] = []

    default_backend = FunctionBackend(lambda _req: "default")
    question_backend = OpenClawBackend(
        chat_runner=lambda session_key, _message: calls.append(session_key) or "question",
    )
    backend = CompositeBackend(
        default_backend=default_backend,
        question_backend=question_backend,
    )

    req = LLMRequest(system_prompt="S", user_prompt="U")

    assert backend.make_extract_llm_fn("abc")(req) == "default"
    assert backend.make_question_llm_fn("abc")(req) == "question"
    assert backend.make_execute_llm_fn("abc")(req) == "default"
    assert calls == ["ftg-session-abc"]


class _FakeBackend:
    def make_extract_llm_fn(self, session_id: str):
        return lambda _req, sid=session_id: f"extract:{sid}"

    def make_question_llm_fn(self, session_id: str):
        return lambda _req, sid=session_id: f"question:{sid}"

    def make_execute_llm_fn(self, session_id: str):
        return lambda _req, sid=session_id: f"execute:{sid}"


def test_fathom_backend_binds_phase_callbacks_once_and_restores_same_scope():
    backend = _FakeBackend()
    fathom = Fathom(backend=backend)
    session = fathom.start(user_input="test", dialogue_fn=lambda q, i=None: "")

    backend_session_id = session._backend_session_id
    req = LLMRequest(system_prompt="", user_prompt="")

    assert session._llm_fn(req) == f"extract:{backend_session_id}"
    assert session._question_llm_fn(req) == f"question:{backend_session_id}"
    assert session._execute_llm_fn(req) == f"execute:{backend_session_id}"

    restored = FathomSession.from_state(session.to_state(), backend=backend)

    assert restored._backend_session_id == backend_session_id
    assert restored._llm_fn(req) == f"extract:{backend_session_id}"
    assert restored._question_llm_fn(req) == f"question:{backend_session_id}"
    assert restored._execute_llm_fn(req) == f"execute:{backend_session_id}"
