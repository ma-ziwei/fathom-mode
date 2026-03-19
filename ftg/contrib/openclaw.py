"""OpenClaw backend for FtG.

Requires: pip install fathom-mode[openclaw]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# OpenClaw Gateway Client (formerly openclaw_gateway.py)
# ---------------------------------------------------------------------------

OPENCLAW_WSL_DISTRO = os.environ.get("FTG_OPENCLAW_WSL_DISTRO", "Ubuntu")
OPENCLAW_WSL_HOST = os.environ.get("FTG_OPENCLAW_WSL_HOST", "wsl.localhost")
OPENCLAW_CONNECT_TIMEOUT_SECONDS = 15.0
OPENCLAW_CHAT_TIMEOUT_SECONDS = 180.0
OPENCLAW_HISTORY_POLL_INTERVAL_SECONDS = 2.0


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _openclaw_candidate_paths(*parts: str) -> list[Path]:
    candidates = [Path.home() / ".openclaw"]
    if sys.platform == "win32":
        candidates.append(
            Path(rf"\\{OPENCLAW_WSL_HOST}\{OPENCLAW_WSL_DISTRO}\home\{os.environ.get('FTG_OPENCLAW_WSL_USER', 'ubuntu')}\.openclaw")
        )
    paths: list[Path] = []
    for base in candidates:
        candidate = base.joinpath(*parts)
        if candidate.exists():
            paths.append(candidate)
    return paths


def _detect_openclaw_wsl_host() -> str | None:
    try:
        completed = subprocess.run(
            ["wsl", "-d", OPENCLAW_WSL_DISTRO, "bash", "-lc", "hostname -I"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    for token in (completed.stdout or "").split():
        if re.match(r"^\d+\.\d+\.\d+\.\d+$", token):
            return token
    return None


def read_openclaw_gateway_config() -> tuple[str, str]:
    for path in _openclaw_candidate_paths("openclaw.json"):
        data = _load_json_if_exists(path)
        if not isinstance(data, dict):
            continue
        gateway = data.get("gateway", {})
        auth = gateway.get("auth", {}) if isinstance(gateway, dict) else {}
        token = auth.get("token") if isinstance(auth, dict) else None
        port = gateway.get("port", 18789) if isinstance(gateway, dict) else 18789
        bind = str(gateway.get("bind", "loopback")).strip().lower() if isinstance(gateway, dict) else "loopback"
        custom_bind_host = (
            str(gateway.get("customBindHost") or "").strip() if isinstance(gateway, dict) else ""
        )
        if token:
            if bind == "custom" and custom_bind_host:
                return f"ws://{custom_bind_host}:{port}", str(token)
            if bind in {"lan", "tailnet"}:
                wsl_host = _detect_openclaw_wsl_host()
                if wsl_host:
                    return f"ws://{wsl_host}:{port}", str(token)
            return f"ws://127.0.0.1:{port}", str(token)
    raise RuntimeError("OpenClaw gateway config/token not found")


def _is_timeout_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


def _openclaw_message_text(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    text = message.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    content = message.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            chunk = item.get("text")
            if isinstance(chunk, str) and chunk.strip():
                parts.append(chunk.strip())
    joined = "\n".join(parts).strip()
    return joined or None


def _latest_assistant_timestamp(messages: list[dict[str, Any]]) -> int:
    latest = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        timestamp = int(message.get("timestamp") or 0)
        latest = max(latest, timestamp)
    return latest


def _latest_assistant_text_from_history(
    messages: list[dict[str, Any]],
    *,
    min_timestamp: int,
) -> str | None:
    latest_message: dict[str, Any] | None = None
    latest_timestamp = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        timestamp = int(message.get("timestamp") or 0)
        if timestamp < min_timestamp or timestamp < latest_timestamp:
            continue
        text = _openclaw_message_text(message)
        if not text:
            continue
        latest_message = message
        latest_timestamp = timestamp
    if latest_message is None:
        return None
    return _openclaw_message_text(latest_message)


class OpenClawGatewayClient:
    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self.token = token
        self.ws = None

    def __enter__(self) -> "OpenClawGatewayClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _send_json(self, payload: dict[str, Any]) -> None:
        assert self.ws is not None
        self.ws.send(json.dumps(payload, ensure_ascii=False))

    def _recv_json(self, timeout_seconds: float) -> dict[str, Any]:
        import websocket

        assert self.ws is not None
        self.ws.settimeout(timeout_seconds)
        raw = self.ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        message = json.loads(raw)
        if not isinstance(message, dict):
            raise RuntimeError("OpenClaw gateway returned a non-object message")
        return message

    def _wait_for_response(self, req_id: str, timeout_seconds: float) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                msg = self._recv_json(max(0.1, deadline - time.time()))
            except Exception as exc:
                if _is_timeout_error(exc):
                    continue
                raise
            if msg.get("type") != "res" or msg.get("id") != req_id:
                continue
            if not msg.get("ok"):
                error = msg.get("error") or {}
                raise RuntimeError(str(error.get("message") or "OpenClaw request failed"))
            payload = msg.get("payload")
            return payload if isinstance(payload, dict) else {}
        raise RuntimeError("Timed out waiting for OpenClaw response")

    def connect(self) -> None:
        try:
            import websocket
        except ImportError:
            raise ImportError(
                "websocket-client not installed. Run: pip install fathom-mode[openclaw]"
            ) from None

        self.ws = websocket.create_connection(self.url, timeout=OPENCLAW_CONNECT_TIMEOUT_SECONDS)

        deadline = time.time() + OPENCLAW_CONNECT_TIMEOUT_SECONDS
        nonce = None
        while time.time() < deadline:
            msg = self._recv_json(max(0.1, deadline - time.time()))
            if msg.get("type") == "event" and msg.get("event") == "connect.challenge":
                payload = msg.get("payload") or {}
                nonce = payload.get("nonce")
                break
        if not nonce:
            raise RuntimeError("OpenClaw gateway did not send a connect challenge")

        req_id = uuid.uuid4().hex
        self._send_json(
            {
                "type": "req",
                "id": req_id,
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": "cli",
                        "version": "ftg",
                        "platform": sys.platform,
                        "mode": "cli",
                        "instanceId": uuid.uuid4().hex,
                    },
                    "role": "operator",
                    "scopes": ["operator.admin", "operator.approvals", "operator.pairing"],
                    "auth": {"token": self.token},
                    "caps": ["tool-events"],
                    "userAgent": "ftg-cli",
                    "locale": "en",
                },
            }
        )
        self._wait_for_response(req_id, OPENCLAW_CONNECT_TIMEOUT_SECONDS)

    def request(self, method: str, params: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        req_id = uuid.uuid4().hex
        self._send_json(
            {
                "type": "req",
                "id": req_id,
                "method": method,
                "params": params,
            }
        )
        return self._wait_for_response(req_id, timeout_seconds)

    def chat_history(self, session_key: str, *, limit: int = 20) -> list[dict[str, Any]]:
        history = self.request(
            "chat.history",
            {"sessionKey": session_key, "limit": limit},
            OPENCLAW_CONNECT_TIMEOUT_SECONDS,
        )
        messages = history.get("messages")
        return messages if isinstance(messages, list) else []

    def run_chat(self, session_key: str, message: str, *, timeout_seconds: float = OPENCLAW_CHAT_TIMEOUT_SECONDS) -> str:
        baseline_messages = self.chat_history(session_key, limit=10)
        last_assistant_timestamp = _latest_assistant_timestamp(baseline_messages)
        min_reply_timestamp = max(int(time.time() * 1000), last_assistant_timestamp + 1)
        idempotency_key = uuid.uuid4().hex
        final_text: str | None = None
        latest_history_messages: list[dict[str, Any]] = []
        deadline = time.time() + timeout_seconds
        next_history_poll_at = time.time()

        send_result = self.request(
            "chat.send",
            {
                "sessionKey": session_key,
                "message": message,
                "deliver": False,
                "idempotencyKey": idempotency_key,
            },
            OPENCLAW_CONNECT_TIMEOUT_SECONDS,
        )
        run_id = str(send_result.get("runId") or idempotency_key)

        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            recv_timeout = min(remaining, OPENCLAW_HISTORY_POLL_INTERVAL_SECONDS)
            try:
                msg = self._recv_json(recv_timeout)
            except Exception as exc:
                if not _is_timeout_error(exc):
                    raise
            else:
                if msg.get("type") == "event" and msg.get("event") == "chat":
                    payload = msg.get("payload") or {}
                    if payload.get("runId") == run_id:
                        state = payload.get("state")
                        text = _openclaw_message_text(payload.get("message"))
                        if state == "final" and text:
                            final_text = text
                            break
                        if state == "aborted":
                            raise RuntimeError("OpenClaw chat aborted")
                        if state == "error":
                            raise RuntimeError(str(payload.get("errorMessage") or "OpenClaw chat failed"))

            if time.time() < next_history_poll_at:
                continue

            history_messages = self.chat_history(session_key, limit=20)
            latest_history_messages = history_messages
            final_text = _latest_assistant_text_from_history(
                history_messages,
                min_timestamp=min_reply_timestamp,
            )
            if final_text:
                break
            next_history_poll_at = time.time() + OPENCLAW_HISTORY_POLL_INTERVAL_SECONDS

        if not final_text and latest_history_messages:
            final_text = _latest_assistant_text_from_history(
                latest_history_messages,
                min_timestamp=min_reply_timestamp,
            )

        if not final_text:
            raise RuntimeError("OpenClaw did not return a final assistant message")
        return final_text.strip()

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None


def make_openclaw_round_llm(session_key: str) -> Callable[[Any], str]:
    ws_url, token = read_openclaw_gateway_config()

    def llm_fn(req: Any) -> str:
        system_prompt = getattr(req, "system_prompt", "") or ""
        user_prompt = getattr(req, "user_prompt", "") or ""
        json_mode = bool(getattr(req, "json_mode", False))
        parts = [
            "You are the FtG second-pass response generator.",
            "You may call tools as needed to supplement facts, but your final output must be a single JSON object — no Markdown code blocks, no explanations.",
        ]
        if json_mode:
            parts.append("This output must be strict JSON.")
        if system_prompt:
            parts.append(f"[System Protocol]\n{system_prompt}")
        if user_prompt:
            parts.append(f"[FtG Materials]\n{user_prompt}")
        message = "\n\n".join(parts)
        with OpenClawGatewayClient(ws_url, token) as client:
            return client.run_chat(session_key=session_key, message=message)

    return llm_fn


# ---------------------------------------------------------------------------
# OpenClaw Backend
# ---------------------------------------------------------------------------

@dataclass
class OpenClawBackend:
    """
    FtGBackend implementation for OpenClaw.

    call() uses an ephemeral session key (no memory).
    send_to_session() uses a persistent session key (with memory).
    """

    session_prefix: str = "ftg"
    fallback_llm: Callable[[Any], str] | None = None
    chat_runner: Callable[[str, str], str] | None = None
    persistent_session_key_factory: Callable[[str], str] | None = None
    execute_session_key_factory: Callable[[str], str] | None = None

    def call(self, req: Any) -> str:
        """Stateless LLM call — ephemeral session key, no memory."""
        call_id = uuid.uuid4().hex[:8]
        session_key = f"{self.session_prefix}-stateless-{call_id}"
        return self._invoke(req, session_key=session_key)

    def send_to_session(self, message: str, *, session_id: str) -> str:
        """Stateful call — persistent session key, with memory."""
        session_key = f"{self.session_prefix}-session-{_clean_session_component(session_id)}"
        return self._run_chat(session_key=session_key, message=message)

    # --- Legacy interface ---

    def make_extract_llm_fn(self, session_id: str) -> Callable[[Any], str]:
        def llm_fn(req: Any) -> str:
            call_id = uuid.uuid4().hex[:8]
            session_key = f"{self.session_prefix}-extract-{_clean_session_component(session_id)}-{call_id}"
            return self._invoke(req, session_key=session_key)
        return llm_fn

    def make_question_llm_fn(self, session_id: str) -> Callable[[Any], str]:
        session_key = self._persistent_session_key(session_id)
        def llm_fn(req: Any) -> str:
            return self._invoke(req, session_key=session_key)
        return llm_fn

    def make_execute_llm_fn(self, session_id: str) -> Callable[[Any], str]:
        stable_session_id = _clean_session_component(session_id)
        session_key = (
            self.execute_session_key_factory(stable_session_id)
            if self.execute_session_key_factory
            else self._persistent_session_key(stable_session_id)
        )
        def llm_fn(req: Any) -> str:
            return self._invoke(req, session_key=session_key)
        return llm_fn

    def _persistent_session_key(self, session_id: str) -> str:
        stable_session_id = _clean_session_component(session_id)
        if self.persistent_session_key_factory is not None:
            return self.persistent_session_key_factory(stable_session_id)
        return f"{self.session_prefix}-session-{stable_session_id}"

    # --- Internal ---

    def _invoke(self, req: Any, *, session_key: str) -> str:
        message = _format_openclaw_message(req)
        try:
            return self._run_chat(session_key=session_key, message=message)
        except Exception:
            if self.fallback_llm is None:
                raise
            return self.fallback_llm(req)

    def _run_chat(self, *, session_key: str, message: str) -> str:
        if self.chat_runner is not None:
            return self.chat_runner(session_key, message)
        ws_url, token = read_openclaw_gateway_config()
        with OpenClawGatewayClient(ws_url, token) as client:
            return client.run_chat(session_key=session_key, message=message)


def _clean_session_component(value: str) -> str:
    return (
        "".join(
            ch for ch in str(value or "").strip()
            if ch.isalnum() or ch in {"-", "_"}
        ).strip("-_")
        or "session"
    )


def _format_openclaw_message(req: Any) -> str:
    system_prompt = str(getattr(req, "system_prompt", "") or "").strip()
    user_prompt = str(getattr(req, "user_prompt", "") or "").strip()
    json_mode = bool(getattr(req, "json_mode", False))
    parts: list[str] = []
    if system_prompt:
        parts.append(system_prompt)
    if user_prompt:
        parts.append(user_prompt)
    if json_mode:
        parts.append("Return strict JSON only.")
    return "\n\n".join(parts).strip()
