"""
Utilities — JSON parsing, math helpers.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_outer_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _repair_inner_quotes(text: str) -> str:
    """
    Repair common malformed-JSON cases where a model emits unescaped inner quotes
    inside string values, e.g.:
      "raw_quote": "learning "Latin" in "high school""
    """
    result: list[str] = []
    in_string = False
    escape = False
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]
        if not in_string:
            result.append(ch)
            if ch == '"':
                in_string = True
                escape = False
            i += 1
            continue

        if escape:
            result.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\":
            result.append(ch)
            escape = True
            i += 1
            continue

        if ch == '"':
            j = i + 1
            while j < length and text[j].isspace():
                j += 1
            next_char = text[j] if j < length else ""
            if next_char in {",", "}", "]", ":"} or j >= length:
                result.append(ch)
                in_string = False
            else:
                result.append('\\"')
            i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def parse_llm_json(raw: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response with robustness against:
    - Markdown code fences
    - Leading/trailing text outside JSON
    - Trailing commas
    """
    if not raw:
        return {}

    text = _extract_outer_json_object(_strip_code_fences(raw.strip()))

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    repaired = _repair_inner_quotes(cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    logger.warning("Failed to parse LLM JSON: %s...", raw[:200])
    return {}


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
