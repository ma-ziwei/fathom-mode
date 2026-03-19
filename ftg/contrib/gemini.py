"""Gemini backend for FtG.

Requires: pip install fathom-mode[gemini]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeminiBackend:
    """
    FtGBackend implementation for Google Gemini.

    call() is stateless — fresh messages each time.
    send_to_session() maintains in-memory conversation history per session_id.
    """

    api_key: str
    model: str = "gemini-2.0-flash"
    _session_histories: dict[str, list[dict]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )
    _cached_client: Any = field(default=None, repr=False)

    def _client(self) -> Any:
        if self._cached_client is not None:
            return self._cached_client
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai SDK not installed. Run: pip install fathom-mode[gemini]"
            ) from None
        self._cached_client = genai.Client(api_key=self.api_key)
        return self._cached_client

    def call(self, req: Any) -> str:
        """Stateless LLM call — no memory."""
        from google.genai import types

        system_prompt = getattr(req, "system_prompt", "") or ""
        user_prompt = getattr(req, "user_prompt", "") or ""
        temperature = getattr(req, "temperature", 0.3)
        json_mode = bool(getattr(req, "json_mode", False))

        config = types.GenerateContentConfig(temperature=temperature)
        if system_prompt:
            config.system_instruction = system_prompt
        if json_mode:
            config.response_mime_type = "application/json"

        response = self._client().models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        return response.text or ""

    def call_in_session(self, req: Any, *, session_id: str) -> str:
        """Structured call with session history — used by questioner."""
        from google.genai import types

        system_prompt = getattr(req, "system_prompt", "") or ""
        user_prompt = getattr(req, "user_prompt", "") or ""
        temperature = getattr(req, "temperature", 0.3)
        json_mode = bool(getattr(req, "json_mode", False))

        config = types.GenerateContentConfig(temperature=temperature)
        if system_prompt:
            config.system_instruction = system_prompt
        if json_mode:
            config.response_mime_type = "application/json"

        history = self._session_histories[session_id]
        contents = [{"role": h["role"], "parts": h["parts"]} for h in history]
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})

        response = self._client().models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        reply = response.text or ""

        history.append({"role": "user", "parts": [{"text": user_prompt}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        return reply

    def send_to_session(self, message: str, *, session_id: str) -> str:
        """Plain-text call with session history — used by execute/downstream."""
        history = self._session_histories[session_id]
        history.append({"role": "user", "parts": [{"text": message}]})

        response = self._client().models.generate_content(
            model=self.model,
            contents=[{"role": h["role"], "parts": h["parts"]} for h in history],
        )
        reply = response.text or ""
        history.append({"role": "model", "parts": [{"text": reply}]})
        return reply
