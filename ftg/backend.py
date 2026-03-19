"""
Backend abstractions for FtG.

FtGBackend is the primary interface: three methods, one credential.
- call(): stateless LLM call (no memory) — used for extract
- call_in_session(): structured call with session history — used for questioner
- send_to_session(): plain-text call with session history — used for execute/downstream

LegacyBackend is the old 3-method interface, kept for backward compatibility.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Protocol


LLMCallable = Callable[[Any], str]


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class FtGBackend(Protocol):
    """
    Unified backend interface for FtG.

    Three methods — one credential:
      - call()            → stateless (extract uses this)
      - call_in_session() → structured + session history (questioner uses this)
      - send_to_session() → plain text + session history (downstream uses this)

    Questioner and downstream share the same session_id.
    Extract is isolated — no cross-call memory.
    """

    def call(self, req: Any) -> str:
        """Stateless LLM call. No memory between calls.

        Used by FtG internally for extraction and causal verification.
        Each call is independent.
        """
        ...

    def call_in_session(self, req: Any, *, session_id: str) -> str:
        """Structured LLM call with session history.

        Like call(), but maintains conversation history per session_id.
        Used by the questioner — supports json_mode, temperature, system_prompt.
        Shares the same session history as send_to_session().
        """
        ...

    def send_to_session(self, message: str, *, session_id: str) -> str:
        """Plain-text message to a downstream session with memory/context.

        The compiled prompt is sent here. Same session_id = same conversation.
        Shares the same session history as call_in_session().
        """
        ...


# ---------------------------------------------------------------------------
# Legacy 3-method protocol — backward compatibility
# ---------------------------------------------------------------------------

class LegacyBackend(Protocol):
    """Old phase-aware backend interface. Use FtGBackend instead."""

    def make_extract_llm_fn(self, session_id: str) -> LLMCallable:
        ...

    def make_question_llm_fn(self, session_id: str) -> LLMCallable:
        ...

    def make_execute_llm_fn(self, session_id: str) -> LLMCallable:
        ...


def is_legacy_backend(backend: Any) -> bool:
    """Check if a backend implements the old 3-method interface."""
    return (
        hasattr(backend, "make_extract_llm_fn")
        and not hasattr(backend, "call")
    )


# ---------------------------------------------------------------------------
# Adapter: FtGBackend → internal 3-method shape (used by Fathom factory)
# ---------------------------------------------------------------------------

class _UnifiedAdapter:
    """Wraps a FtGBackend into the internal 3-method interface."""

    def __init__(self, backend: FtGBackend) -> None:
        self._backend = backend

    def make_extract_llm_fn(self, session_id: str) -> LLMCallable:  # noqa: ARG002
        return self._backend.call

    def make_question_llm_fn(self, session_id: str) -> LLMCallable:
        backend = self._backend

        def question_fn(req: Any) -> str:
            return backend.call_in_session(req, session_id=session_id)

        return question_fn

    def make_execute_llm_fn(self, session_id: str) -> LLMCallable:
        backend = self._backend

        def execute_fn(req: Any) -> str:
            message = _format_req_as_message(req)
            return backend.send_to_session(message, session_id=session_id)

        return execute_fn


def _format_req_as_message(req: Any) -> str:
    """Convert an LLMRequest-like object to a plain text message."""
    system_prompt = str(getattr(req, "system_prompt", "") or "").strip()
    user_prompt = str(getattr(req, "user_prompt", "") or "").strip()
    parts: list[str] = []
    if system_prompt:
        parts.append(system_prompt)
    if user_prompt:
        parts.append(user_prompt)
    return "\n\n".join(parts).strip()


def resolve_backend(backend: Any) -> Any:
    """Normalize a backend to the internal 3-method interface.

    Accepts either FtGBackend (new) or LegacyBackend (old).
    Returns an object with make_extract_llm_fn / make_question_llm_fn / make_execute_llm_fn.
    """
    if is_legacy_backend(backend):
        return backend
    return _UnifiedAdapter(backend)


# ---------------------------------------------------------------------------
# FunctionBackend — simple callable wrapper (implements both protocols)
# ---------------------------------------------------------------------------

@dataclass
class FunctionBackend:
    """
    Wrap a raw LLM callable into an FtG backend.

    Simplest integration: provide one function, FtG uses it for everything.
    Implements both FtGBackend (new) and LegacyBackend (old) interfaces.
    """

    llm_fn: LLMCallable
    question_llm_fn: LLMCallable | None = None
    execute_llm_fn: LLMCallable | None = None

    # --- New FtGBackend interface ---

    def call(self, req: Any) -> str:
        return self.llm_fn(req)

    def call_in_session(self, req: Any, *, session_id: str) -> str:  # noqa: ARG002
        """Structured call — falls back to stateless call() for plain functions."""
        return self.llm_fn(req)

    def send_to_session(self, message: str, *, session_id: str) -> str:  # noqa: ARG002
        from ftg.fathom import LLMRequest
        fn = self.execute_llm_fn or self.question_llm_fn or self.llm_fn
        return fn(LLMRequest(system_prompt="", user_prompt=message))

    # --- Legacy interface ---

    def make_extract_llm_fn(self, session_id: str) -> LLMCallable:  # noqa: ARG002
        return self.llm_fn

    def make_question_llm_fn(self, session_id: str) -> LLMCallable:  # noqa: ARG002
        return self.question_llm_fn or self.llm_fn

    def make_execute_llm_fn(self, session_id: str) -> LLMCallable:  # noqa: ARG002
        return self.execute_llm_fn or self.question_llm_fn or self.llm_fn


# ---------------------------------------------------------------------------
# CompositeBackend — phase-specific override (legacy interface)
# ---------------------------------------------------------------------------

@dataclass
class CompositeBackend:
    """Compose one default backend with optional phase-specific overrides."""

    default_backend: Any
    extract_backend: Any | None = None
    question_backend: Any | None = None
    execute_backend: Any | None = None

    def make_extract_llm_fn(self, session_id: str) -> LLMCallable:
        backend = self.extract_backend or self.default_backend
        return backend.make_extract_llm_fn(session_id)

    def make_question_llm_fn(self, session_id: str) -> LLMCallable:
        backend = self.question_backend or self.default_backend
        return backend.make_question_llm_fn(session_id)

    def make_execute_llm_fn(self, session_id: str) -> LLMCallable:
        backend = self.execute_backend or self.default_backend
        return backend.make_execute_llm_fn(session_id)
