"""OpenAI backend for FtG.

Requires: pip install fathom-mode[openai]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpenAIBackend:
    """
    FtGBackend implementation for OpenAI.

    call() is stateless — fresh messages each time.
    send_to_session() maintains in-memory conversation history per session_id.
    """

    api_key: str
    model: str = "gpt-4o"
    max_tokens: int = 4096
    _session_histories: dict[str, list[dict]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )
    _cached_client: Any = field(default=None, repr=False)

    def _client(self) -> Any:
        if self._cached_client is not None:
            return self._cached_client
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai SDK not installed. Run: pip install fathom-mode[openai]"
            ) from None
        self._cached_client = openai.OpenAI(api_key=self.api_key)
        return self._cached_client

    def call(self, req: Any) -> str:
        """Stateless LLM call — no memory."""
        system_prompt = getattr(req, "system_prompt", "") or ""
        user_prompt = getattr(req, "user_prompt", "") or ""
        temperature = getattr(req, "temperature", 0.3)
        json_mode = bool(getattr(req, "json_mode", False))

        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client().chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def call_in_session(self, req: Any, *, session_id: str) -> str:
        """Structured call with session history — used by questioner."""
        system_prompt = getattr(req, "system_prompt", "") or ""
        user_prompt = getattr(req, "user_prompt", "") or ""
        temperature = getattr(req, "temperature", 0.3)
        json_mode = bool(getattr(req, "json_mode", False))

        history = self._session_histories[session_id]
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client().chat.completions.create(**kwargs)
        reply = response.choices[0].message.content or ""

        history.append({"role": "user", "content": user_prompt})
        history.append({"role": "assistant", "content": reply})
        return reply

    def send_to_session(self, message: str, *, session_id: str) -> str:
        """Plain-text call with session history — used by execute/downstream."""
        history = self._session_histories[session_id]
        history.append({"role": "user", "content": message})

        response = self._client().chat.completions.create(
            model=self.model,
            messages=list(history),
            max_tokens=self.max_tokens,
        )
        reply = response.choices[0].message.content or ""
        history.append({"role": "assistant", "content": reply})
        return reply


def make_openai_llm(api_key: str, model: str = "gpt-4o") -> Any:
    """Create an llm_fn for OpenAI (legacy helper)."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai SDK not installed. Run: pip install fathom-mode[openai]"
        ) from None
    client = openai.OpenAI(api_key=api_key)

    def llm_fn(req: Any) -> str:
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.user_prompt},
            ],
            "temperature": req.temperature,
        }
        if req.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    return llm_fn


def make_openai_embed(api_key: str, model: str = "text-embedding-3-small") -> Any:
    """Create an embed_fn for OpenAI (legacy helper)."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai SDK not installed. Run: pip install fathom-mode[openai]"
        ) from None
    client = openai.OpenAI(api_key=api_key)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in response.data]

    return embed_fn
