from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from ftg import cli


class _FakeSession:
    def __init__(self) -> None:
        self.relay_calls: list[str] = []

    def relay(self, user_message: str = ""):
        self.relay_calls.append(user_message)
        return SimpleNamespace(
            action="ask_user",
            display="First round follow-up question",
            fathom_score=25,
            compiled_prompt=None,
            task_type="thinking",
        )

    def to_state(self) -> dict:
        return {}


class _FakeFathom:
    def __init__(self, llm_fn=None, backend=None, **_kwargs) -> None:
        self.llm_fn = llm_fn
        self.backend = backend

    def start(self, *, user_input: str, dialogue_fn):
        assert user_input == "Should I change jobs"
        assert callable(dialogue_fn)
        return _FakeSession()


def test_cmd_relay_new_session_does_not_double_feed_initial_message(monkeypatch, capsys):
    created_sessions: list[_FakeSession] = []

    class _CapturingFathom(_FakeFathom):
        def start(self, *, user_input: str, dialogue_fn):
            session = super().start(user_input=user_input, dialogue_fn=dialogue_fn)
            created_sessions.append(session)
            return session

    monkeypatch.setattr(cli, "_make_routing_llm_fn", lambda: object())
    monkeypatch.setattr(cli, "Fathom", _CapturingFathom)
    monkeypatch.setattr(cli, "_save_session", lambda session_id, session, meta: None)
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: SimpleNamespace(hex="abc123def4567890"))

    cli.cmd_relay(None, "Should I change jobs")

    assert len(created_sessions) == 1
    assert created_sessions[0].relay_calls == [""]

    out = capsys.readouterr().out
    assert "abc123def456" in out
    assert "First round follow-up question" in out


def test_cmd_relay_saves_phase_for_new_session(monkeypatch):
    saved: dict[str, object] = {}

    class _ConfirmSession(_FakeSession):
        def relay(self, user_message: str = ""):
            self.relay_calls.append(user_message)
            return SimpleNamespace(
                action="confirm",
                display="Confirmation summary",
                fathom_score=100,
                compiled_prompt="Task prompt",
                task_type="thinking",
            )

    class _ConfirmFathom(_FakeFathom):
        def start(self, *, user_input: str, dialogue_fn):
            return _ConfirmSession()

    def _capture_save(session_id, session, meta):
        saved["session_id"] = session_id
        saved["meta"] = dict(meta)

    monkeypatch.setattr(cli, "_make_routing_llm_fn", lambda: object())
    monkeypatch.setattr(cli, "Fathom", _ConfirmFathom)
    monkeypatch.setattr(cli, "_save_session", _capture_save)
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: SimpleNamespace(hex="phasecase1234567890"))

    cli.cmd_relay(None, "Should I change jobs")

    assert saved["session_id"] == "phasecase123"
    assert saved["meta"]["phase"] == "confirming"


def test_cmd_fathom_uses_relay_protocol_and_persists_review_phase(monkeypatch, capsys):
    saved: dict[str, object] = {}

    class _FathomCommandSession:
        def relay(self, user_message: str = ""):
            assert user_message == "fathom"
            return SimpleNamespace(
                action="review",
                display="manual fathom",
                fathom_score=91,
                compiled_prompt="compiled prompt",
                task_type="thinking",
            )

    monkeypatch.setattr(
        cli,
        "_load_session",
        lambda session_id: (_FathomCommandSession(), {"question": "q", "phase": "questioning"}),
    )
    monkeypatch.setattr(cli, "_touch_meta", lambda meta: meta)
    monkeypatch.setattr(cli, "_save_session", lambda session_id, session, meta: saved.update({"session_id": session_id, "meta": dict(meta)}))

    cli.cmd_fathom("sess123")

    assert saved["session_id"] == "sess123"
    assert saved["meta"]["phase"] == "compiled_review"
    out = capsys.readouterr().out
    assert "\"action\": \"review\"" in out
    assert "\"compiled_prompt\": \"compiled prompt\"" in out


def test_cmd_stop_uses_relay_protocol_and_persists_stopped_phase(monkeypatch, capsys):
    saved: dict[str, object] = {}

    class _StopCommandSession:
        def relay(self, user_message: str = ""):
            assert user_message == "stop"
            return SimpleNamespace(
                action="stop",
                display="stopped",
                fathom_score=44,
                compiled_prompt=None,
                task_type="thinking",
            )

    monkeypatch.setattr(
        cli,
        "_load_session",
        lambda session_id: (_StopCommandSession(), {"question": "q", "phase": "questioning"}),
    )
    monkeypatch.setattr(cli, "_touch_meta", lambda meta: meta)
    monkeypatch.setattr(cli, "_save_session", lambda session_id, session, meta: saved.update({"session_id": session_id, "meta": dict(meta)}))

    cli.cmd_stop("sess123")

    assert saved["session_id"] == "sess123"
    assert saved["meta"]["phase"] == "stopped"
    out = capsys.readouterr().out
    assert "\"action\": \"stop\"" in out


def test_session_persistence_uses_utf8(monkeypatch, tmp_path: Path):
    class _Utf8Session:
        def to_state(self) -> dict:
            return {"topic": "home buying analysis"}

    class _FakeFathomSession:
        @staticmethod
        def from_state(state: dict, llm_fn):
            return {"state": state, "llm_fn": llm_fn}

    monkeypatch.setattr(cli, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(cli, "FathomSession", _FakeFathomSession)
    monkeypatch.setattr(cli, "_make_routing_llm_fn", lambda: "fake-llm")

    meta = {
        "question": "I want to buy a house",
        "phase": "questioning",
        "created_at": time.time(),
        "last_active": time.time(),
    }
    cli._save_session("utf8case", _Utf8Session(), meta)

    raw = (tmp_path / "utf8case.json").read_bytes()
    assert "I want to buy a house".encode("utf-8") in raw

    session, loaded_meta = cli._load_session("utf8case")

    assert session["state"] == {"topic": "home buying analysis"}
    assert session["llm_fn"] == "fake-llm"
    assert loaded_meta["question"] == "I want to buy a house"

    saved = json.loads((tmp_path / "utf8case.json").read_text(encoding="utf-8"))
    assert saved["meta"]["question"] == "I want to buy a house"
