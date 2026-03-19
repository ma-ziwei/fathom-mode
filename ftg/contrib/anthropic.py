"""Anthropic backend for FtG.

Requires: pip install fathom-mode[anthropic]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnthropicBackend:
    """
    FtGBackend implementation for Anthropic Claude.

    call() is stateless — fresh messages each time.
    send_to_session() maintains in-memory conversation history per session_id.
    """

    api_key: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    _session_histories: dict[str, list[dict]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )
    _cached_client: Any = field(default=None, repr=False)

    def _client(self) -> Any:
        if self._cached_client is not None:
            return self._cached_client
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install fathom-mode[anthropic]"
            ) from None
        self._cached_client = anthropic.Anthropic(api_key=self.api_key)
        return self._cached_client

    def call(self, req: Any) -> str:
        """Stateless LLM call — no memory."""
        system_prompt = getattr(req, "system_prompt", "") or ""
        user_prompt = getattr(req, "user_prompt", "") or ""
        temperature = getattr(req, "temperature", 0.3)

        response = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return response.content[0].text

    def call_in_session(self, req: Any, *, session_id: str) -> str:
        """Structured call with session history — used by questioner."""
        system_prompt = getattr(req, "system_prompt", "") or ""
        user_prompt = getattr(req, "user_prompt", "") or ""
        temperature = getattr(req, "temperature", 0.3)

        history = self._session_histories[session_id]
        messages = list(history) + [{"role": "user", "content": user_prompt}]

        response = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
        )
        reply = response.content[0].text

        history.append({"role": "user", "content": user_prompt})
        history.append({"role": "assistant", "content": reply})
        return reply

    def send_to_session(self, message: str, *, session_id: str) -> str:
        """Plain-text call with session history — used by execute/downstream."""
        history = self._session_histories[session_id]
        history.append({"role": "user", "content": message})

        response = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=list(history),
        )
        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})
        return reply


def make_anthropic_llm(api_key: str, model: str = "claude-sonnet-4-20250514") -> Any:
    """Create an llm_fn for Anthropic Claude (legacy helper)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic SDK not installed. Run: pip install fathom-mode[anthropic]"
        ) from None
    client = anthropic.Anthropic(api_key=api_key)

    def llm_fn(req: Any) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=req.system_prompt,
            messages=[{"role": "user", "content": req.user_prompt}],
            temperature=req.temperature,
        )
        return response.content[0].text

    return llm_fn
