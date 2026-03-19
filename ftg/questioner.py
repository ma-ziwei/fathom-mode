"""
Question Generation — produces contextual follow-up questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ftg.dimensions import DIMENSION_DESCRIPTIONS, DIMENSION_LABELS, find_target_dimension
from ftg.graph import IntentGraph
from ftg.models import CausalHypothesis, RelationType
from ftg.utils import parse_llm_json

if TYPE_CHECKING:
    from ftg.causal import CausalTracker
    from ftg.fathom import LLMRequest

LLMFunction = Callable[["LLMRequest"], str]


@dataclass
class DimensionCandidate:
    dimension: str
    reason: str = ""
    gap: str = ""
    target_types: list[str] = field(default_factory=list)
    related_quotes: list[str] = field(default_factory=list)


@dataclass
class CausalCandidate:
    hypothesis_id: str
    source_node_id: str
    target_node_id: str
    source_content: str
    target_content: str
    reason: str = ""


@dataclass
class RoundConstraints:
    task_type: str
    round_count: int
    allowed_modes: list[str] = field(default_factory=lambda: ["dimension"])
    one_question_only: bool = True
    must_output_response: bool = True
    must_output_insight: bool = True


@dataclass
class RoundMaterials:
    root_question: str
    latest_user_response: str = ""
    conversation_history: str = ""
    graph_summary: str = ""
    dimension_candidate: DimensionCandidate | None = None
    causal_candidate: CausalCandidate | None = None
    dimension_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    dimension_bound_responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    user_context: str = ""


@dataclass
class RoundPacket:
    constraints: RoundConstraints
    materials: RoundMaterials


# ---------------------------------------------------------------------------
# Question Generation Prompt
# ---------------------------------------------------------------------------

# Legacy prompt blocks removed. RoundPacket is the single active LLM prompt path.


ROUND_PROTOCOL_SYSTEM_PROMPT = """\
Generate one round of output based on the given materials.
Requirements:
- Ask only one question
- ask_mode must be chosen from allowed_modes
- Talk directly TO the user (use "you/your", never "the user")
- Your "response" MUST directly reference what the user just said in [latest_user_response] — never give a generic acknowledgment like "I understand your request"
- Your response must be about the user's situation, not a system status report
- Do not repeat directions the user already answered or declined
- Never say "unknown" or "currently unknown" — state what you CAN infer
- Never quote or paraphrase from [graph_summary] — it is internal context only
- Adapt your questioning style to the task_type and questioning_hint in [constraints] — these reflect the user's collaboration need
- Ask a natural, scene-specific question — not a generic or abstract one
- Return JSON: {
  "ask_mode": "dimension or causal",
  "response": "brief acknowledgment",
  "insight": "key tension or gap",
  "question": "one natural question",
  "round_action": "ask_user or answer_now"
}
"""


# ---------------------------------------------------------------------------
# Task-aware dimension semantics — tells the LLM what each dimension means
# in the context of a specific task type, so it asks better questions.
# ---------------------------------------------------------------------------

TASK_DIMENSION_SEMANTICS: dict[str, dict[str, str]] = {
    "thinking": {
        "what": "the core problem, question, or decision at hand",
        "how": "constraints, evaluation criteria, and available approaches",
        "why": "underlying motivation, values, fears, and priorities",
        "who": "who is affected, whose opinion matters, stakeholders",
        "when": "decision timeline, urgency, reversibility window",
        "where": "context, environment, or domain that constrains options",
    },
    "creation": {
        "what": "the deliverable — what to create and its core content",
        "how": "style, tone, format, medium, and technical constraints",
        "why": "the purpose — who will see it and what effect it should produce",
        "who": "the audience or recipient",
        "when": "deadline or context timing",
        "where": "the platform, channel, or medium where it will be used",
    },
    "execution": {
        "what": "the desired end state after the operation completes",
        "how": "the specific steps, tools, or path to follow",
        "why": "the underlying goal — what prompted this action now",
        "who": "whose data, account, or workspace is involved",
        "when": "timing, urgency, or sequencing constraints",
        "where": "which application, system, or environment to operate in",
    },
    "learning": {
        "what": "the skill, knowledge domain, or concept to understand",
        "how": "preferred learning style, available resources, depth needed",
        "why": "the career goal or motivation driving this",
        "who": "current level, background, and relevant experience",
        "when": "timeline and time commitment available",
        "where": "learning context — self-study, on-the-job, formal education",
    },
}

# ---------------------------------------------------------------------------
# Task-aware questioning hints — high-level guidance for the questioner LLM
# on how to approach each collaboration mode.
# ---------------------------------------------------------------------------

TASK_QUESTIONING_HINTS: dict[str, str] = {
    "thinking": (
        "- Explore the user's values, fears, and assumptions — not just facts.\n"
        "- If the user presents a binary choice, check whether there are options they haven't considered.\n"
        "- When the user faces a selection or comparison, ask about the ONE differentiating factor that would actually change the answer.\n"
        "- Ask what they've already tried or ruled out before suggesting new directions."
    ),
    "creation": (
        "- Clarify purpose and audience before style and format.\n"
        "- The user usually knows WHAT to create — help them articulate the nuances that make it effective.\n"
        "- Ask about constraints: what must NOT be included, length limits, tone boundaries.\n"
        "- If the medium is ambiguous (email vs doc vs message), ask."
    ),
    "execution": (
        "- Clarify the desired end state before discussing steps.\n"
        "- Identify which parts the user is certain about and which they're unsure of.\n"
        "- If the action seems misaligned with the stated goal, flag it before proceeding.\n"
        "- Ask about rollback: what happens if this goes wrong."
    ),
    "learning": (
        "- Assess the user's current level before recommending paths.\n"
        "- Understand whether the goal is breadth or depth.\n"
        "- Ask what 'success' looks like — certification, job readiness, or personal understanding.\n"
        "- Ask about time constraints and preferred learning format."
    ),
}

# DIMENSION_LABELS imported from ftg.dimensions


# ---------------------------------------------------------------------------
# Question post-processing helpers
# ---------------------------------------------------------------------------

GENERIC_QUESTION_PATTERNS = (
    "can you tell me more",
    "could you elaborate",
    "can you be more specific",
    "could you be more specific",
    "any more details",
    "anything else to add",
    "can you add more",
    "do you have anything else",
    "is there anything else",
    "tell me more",
    "could you expand on that",
)

ABSTRACT_DIMENSION_QUESTION_PATTERNS = (
    "people involved",
    "roles or stakeholders",
    "stakeholders",
    "which part of the information",
    "which aspect to explore",
    "which dimension",
    "this part of the information",
)

GENERIC_INSIGHT_PATTERNS = (
    "some information still needs to be filled in",
    "need more information",
    "need to confirm",
    "need to think about it",
    "let me think",
)

QUESTION_LIKE_PATTERN = re.compile(
    r"(\?|should|would|could|can|do|does|is|are|will|how|why|what|which|who|where|when)",
    re.IGNORECASE,
)

_QUESTION_START_PATTERN = re.compile(
    r"^\s*(should|would|could|can|do|does|is|are|will|how|why|what|which|who|where|when)\b",
    re.IGNORECASE,
)


def _is_likely_question_text(text: str) -> bool:
    """Return True only when *text* looks like an actual question, not a description that happens to contain a question word."""
    if "?" in text:
        return True
    return bool(_QUESTION_START_PATTERN.match(text))


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    # Strip residual XML tags from LLM output (e.g. </final_answer>, </final)
    text = re.sub(r"</?[a-zA-Z_][a-zA-Z0-9_]*>?|</", "", text)
    return text.strip(".,;:!? ")


def _ensure_question_mark(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return normalized
    if normalized.endswith(("?",)):
        return normalized
    return f"{normalized}?"


def _normalize_target_types(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\s,/]+", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            cleaned = _clean_text(item)
            if cleaned:
                items.append(cleaned)
        return items
    return []


def _compact_focus_text(text: str, max_len: int = 18) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(for example|such as|like|mainly|the key is)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[.,;:].*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}..."


def _select_focus_text(target_gap: str, target_types: list[str], target_dimension: str) -> str:
    if target_gap:
        return _compact_focus_text(target_gap)
    if target_types:
        joined = ", ".join(_clean_text(item) for item in target_types if _clean_text(item))
        return _compact_focus_text(joined)
    return f"the {DIMENSION_LABELS.get(target_dimension, target_dimension)} aspect"


def _reframe_to_second_person(text: str) -> str:
    """Convert graph-internal third-person language to natural second-person."""
    normalized = _clean_text(text)
    if not normalized:
        return ""
    # Remove meta-descriptions appended by extractor
    normalized = re.sub(
        r",?\s*(?:which is|this is|this involves) a [\w\s]+ (?:scenario|decision|question|task)\.?$",
        "", normalized, flags=re.IGNORECASE,
    )
    # "the user's X" → "your X"
    normalized = re.sub(r"\b[Tt]he user'?s\b", "your", normalized)
    # "the user is/wants/..." → "you are/want/..."
    normalized = re.sub(r"\b[Tt]he user\b", "you", normalized)
    return normalized


def _compact_relation_text(text: object, max_len: int = 22) -> str:
    cleaned = _reframe_to_second_person(str(text) if text else "")
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(you\s*(are|is)?|this matter)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}..."


def _looks_generic_question(question: str) -> bool:
    normalized = _clean_text(question).lower()
    if not normalized:
        return True
    for pattern in GENERIC_QUESTION_PATTERNS:
        if pattern in normalized:
            return True
    if len(normalized) <= 30 and any(token in normalized.lower() for token in ("specific", "elaborate", "more detail", "expand")):
        return True
    return False


def _looks_abstract_dimension_question(question: str) -> bool:
    normalized = _clean_text(question).lower()
    if not normalized:
        return True
    return any(pattern in normalized for pattern in ABSTRACT_DIMENSION_QUESTION_PATTERNS)


def _looks_generic_insight(insight: str) -> bool:
    normalized = _clean_text(insight).lower()
    if not normalized:
        return True
    for pattern in GENERIC_INSIGHT_PATTERNS:
        if pattern in normalized:
            return True
    if len(normalized) <= 8:
        return True
    return False


def _infer_minimal_progress(
    target_dimension: str,
    target_gap: str,
    target_types: list[str],
    question_mode: str = "dimension",
    causal_hypothesis: CausalHypothesis | None = None,
    redirect_context: dict | None = None,
) -> str:
    if question_mode == "redirect_confirm":
        return "Let me consolidate what you just added into a new direction for analysis."

    if question_mode == "causal" and causal_hypothesis is not None:
        source = _compact_relation_text(causal_hypothesis.source_content)
        target = _compact_relation_text(causal_hypothesis.target_content)
        if source and target:
            return f'I\'m now verifying whether "{source}" and "{target}" have a key causal relationship.'
        return "I'm now verifying whether these two factors have a key causal relationship."

    focus = _select_focus_text(target_gap, target_types, target_dimension)
    return f'So far I\'ve got the main direction. Now I\'m zeroing in on "{focus}" — that\'s the part that will actually change the answer.'


def _infer_structural_tension(
    target_dimension: str,
    target_gap: str,
    target_types: list[str],
    question_mode: str = "dimension",
    causal_hypothesis: CausalHypothesis | None = None,
    redirect_context: dict | None = None,
) -> str:
    if question_mode == "redirect_confirm":
        return "The key here isn't adding more background — it's confirming whether I've correctly identified what you're really focused on."

    if question_mode == "causal" and causal_hypothesis is not None:
        return "What really needs to be clarified here is whether this is causation or just correlation — it directly affects the direction of analysis."

    focus = _select_focus_text(target_gap, target_types, target_dimension)
    return f'What would actually change the answer isn\'t more background, but first confirming "{focus}".'


_MODE_FALLBACK_QUESTIONS: dict[str, dict[str, str]] = {
    "thinking": {
        "who": "Is this mainly about you, or are other people involved too?",
        "what": "What's the single most important thing you're trying to figure out here?",
        "why": "What's driving this — what outcome matters most to you?",
        "when": "Is there any time pressure or deadline here?",
        "where": "Does location or context matter for this?",
        "how": "Do you have a preferred approach, or are you open to suggestions?",
    },
    "creation": {
        "who": "Who is the primary audience for this?",
        "what": "What's the core message or purpose of what you're creating?",
        "why": "What effect should this have on the reader or viewer?",
        "when": "Is there a deadline or timing constraint?",
        "where": "Where will this be published or delivered?",
        "how": "Do you have preferences on tone, style, or format?",
    },
    "execution": {
        "who": "Whose account or workspace will this affect?",
        "what": "What should the end state look like when this is done?",
        "why": "What prompted this action right now?",
        "when": "Is there a window or deadline for this operation?",
        "where": "Which system or environment should this happen in?",
        "how": "Are there specific steps you want followed, or should I choose the approach?",
    },
    "learning": {
        "who": "What's your current level of experience with this?",
        "what": "What specifically do you want to understand or be able to do?",
        "why": "What's the goal — career change, project need, or personal interest?",
        "when": "How much time can you commit to this?",
        "where": "Are you learning on your own, or in a structured program?",
        "how": "Do you prefer hands-on practice, reading, or guided tutorials?",
    },
    "general": {
        "who": "Is this mainly about you, or are other people involved too?",
        "what": "What's the single most important thing you need help with here?",
        "why": "What's driving this — what outcome matters most to you?",
        "when": "Is there any time pressure or deadline here?",
        "where": "Does location or context matter for this?",
        "how": "Do you have a preferred approach, or are you open to suggestions?",
    },
}

# Universal default (used when task_type is unknown)
_DIMENSION_FALLBACK_QUESTIONS: dict[str, str] = _MODE_FALLBACK_QUESTIONS["general"]


def _infer_scene_specific_question(
    target_dimension: str,
    target_gap: str,
    target_types: list[str],
    task_type: str = "general",
) -> str:
    gap = _clean_text(target_gap)

    # Bare dimension labels (e.g. "How", "Why") are not valid gaps
    if gap and gap in DIMENSION_LABELS.values():
        gap = ""

    if gap and _is_likely_question_text(gap):
        return _ensure_question_mark(gap)

    if gap:
        return _ensure_question_mark(f"I still need to know: {gap}")

    effective_dim = target_dimension or "what"
    mode_questions = _MODE_FALLBACK_QUESTIONS.get(task_type, _DIMENSION_FALLBACK_QUESTIONS)
    return mode_questions.get(
        effective_dim,
        "What's the single most important thing you're trying to figure out here?",
    )


def _looks_overquoted_causal_question(question: str) -> bool:
    normalized = _clean_text(question)
    if not normalized:
        return True
    if "these two" in normalized.lower() or "why are they related" in normalized.lower():
        return True
    if normalized.lower().startswith("you mentioned"):
        return True
    if normalized.count('"') >= 4 and len(normalized) >= 40:
        return True
    return False


def _infer_causal_verification_question(
    source_content: str,
    target_content: str,
) -> str:
    source = _compact_relation_text(source_content)
    target = _compact_relation_text(target_content)

    if source and target:
        return _ensure_question_mark(
            f'Do you think "{source}" directly affects "{target}", or are they just coincidentally related'
        )
    if target:
        return _ensure_question_mark(
            f'What do you think is the key reason that affects "{target}"'
        )
    return "Do you think this is a genuine causal relationship, or just a coincidence?"


def _render_redirect_confirm_question(
    original_source: str,
    original_target: str,
    user_alt: str,
) -> str:
    source = _compact_relation_text(original_source)
    target = _compact_relation_text(original_target)
    alt = _compact_relation_text(user_alt)

    if target and alt and source:
        return _ensure_question_mark(
            f'Based on what you said, the key factor affecting "{target}" is more like "{alt}" rather than "{source}". Do you agree'
        )
    if target and alt:
        return _ensure_question_mark(
            f'Based on what you said, the key factor affecting "{target}" is more like "{alt}". Do you agree'
        )
    return "Based on what you said, I need to confirm whether this new causal assessment is correct — do you agree?"


# ---------------------------------------------------------------------------
# Round packet rendering
# ---------------------------------------------------------------------------

def build_graph_summary(graph: IntentGraph) -> str:
    nodes = graph.get_all_nodes()
    if not nodes:
        return ""
    parts: list[str] = ["What we know so far:"]
    for node in nodes[:8]:
        display_text = _reframe_to_second_person(node.content)
        parts.append(f"- {display_text}")
    return "\n".join(parts)


def _render_dimension_bound_responses(responses: dict[str, dict[str, Any]]) -> str:
    if not responses:
        return ""
    lines = []
    for dimension, payload in responses.items():
        raw_text = _clean_text(payload.get("raw_text", ""))
        if raw_text:
            lines.append(f"- {DIMENSION_LABELS.get(dimension, dimension)}: {raw_text}")
    return "\n".join(lines)


def _render_dimension_states(states: dict[str, dict[str, Any]]) -> str:
    if not states:
        return ""
    lines: list[str] = []
    for dimension in [key for key in DIMENSION_LABELS.keys() if key in states]:
        payload = states.get(dimension, {})
        if not isinstance(payload, dict):
            continue
        open_gap = _clean_text(payload.get("open_gap", ""))
        if open_gap:
            lines.append(f"- {DIMENSION_LABELS.get(dimension, dimension)}: {open_gap}")
    return "\n".join(lines)


def _build_round_packet_prompt(packet: RoundPacket) -> str:
    parts: list[str] = []
    constraints = packet.constraints
    materials = packet.materials
    parts.append(f"[root_question]\n{materials.root_question}")
    if materials.latest_user_response:
        parts.append(f"[latest_user_response]\n{materials.latest_user_response}")
    if materials.conversation_history:
        parts.append(f"[conversation_history]\n{materials.conversation_history}")
    if materials.user_context:
        parts.append(f"[external_context]\n{materials.user_context}")
    if materials.graph_summary:
        parts.append(f"[graph_summary]\n{materials.graph_summary}")
    dimension_states = _render_dimension_states(materials.dimension_states)
    if dimension_states:
        parts.append(f"[information_gaps]\n{dimension_states}")
    if materials.dimension_candidate:
        candidate = materials.dimension_candidate
        focus = TASK_DIMENSION_SEMANTICS.get(constraints.task_type, {}).get(candidate.dimension, "")
        candidate_lines = [
            "[dimension_candidate]",
            f"dimension={candidate.dimension}",
        ]
        if focus:
            candidate_lines.append(f"focus={focus}")
        candidate_lines.append(f"gap={candidate.gap}")
        if candidate.target_types:
            candidate_lines.append(f"target_types={' | '.join(candidate.target_types)}")
        if candidate.related_quotes:
            candidate_lines.append(f"related_quotes={' | '.join(candidate.related_quotes)}")
        parts.append(
            "\n".join(candidate_lines)
        )
    if materials.causal_candidate:
        candidate = materials.causal_candidate
        parts.append(
            "\n".join(
                [
                    "[causal_candidate]",
                    f"source={candidate.source_content}",
                    f"target={candidate.target_content}",
                ]
            )
        )
    bound = _render_dimension_bound_responses(materials.dimension_bound_responses)
    if bound:
        parts.append(f"[already_answered — do NOT re-ask these directions]\n{bound}")
    constraint_lines = [
        "[constraints]",
        f"task_type={constraints.task_type}",
        f"round_count={constraints.round_count}",
        f"allowed_modes={' | '.join(constraints.allowed_modes)}",
        f"one_question_only={'true' if constraints.one_question_only else 'false'}",
    ]
    hint = TASK_QUESTIONING_HINTS.get(constraints.task_type, "")
    if hint:
        constraint_lines.append(f"questioning_hint={hint}")
    parts.append("\n".join(constraint_lines))
    return "\n\n".join(part for part in parts if part.strip())


def _fallback_round_output(packet: RoundPacket) -> dict:
    dimension_candidate = packet.materials.dimension_candidate
    causal_candidate = packet.materials.causal_candidate

    if causal_candidate and "causal" in packet.constraints.allowed_modes and not dimension_candidate:
        hyp = CausalHypothesis(
            id=causal_candidate.hypothesis_id,
            source_node_id=causal_candidate.source_node_id,
            target_node_id=causal_candidate.target_node_id,
            source_content=causal_candidate.source_content,
            target_content=causal_candidate.target_content,
        )
        return {
            "ask_mode": "causal",
            "response": _infer_minimal_progress(
                target_dimension="",
                target_gap="",
                target_types=[],
                question_mode="causal",
                causal_hypothesis=hyp,
            ),
            "insight": _infer_structural_tension(
                target_dimension="",
                target_gap="",
                target_types=[],
                question_mode="causal",
                causal_hypothesis=hyp,
            ),
            "question": _infer_causal_verification_question(
                causal_candidate.source_content,
                causal_candidate.target_content,
            ),
        }

    if not dimension_candidate:
        return {
            "ask_mode": "dimension",
            "response": "I've captured the main direction of your question so far.",
            "insight": "The most important next step is to fill in the piece of information that would actually change the answer.",
            "question": "Which part would you most like me to explore further first?",
        }

    return {
        "ask_mode": "dimension",
        "response": _infer_minimal_progress(
            target_dimension=dimension_candidate.dimension,
            target_gap=dimension_candidate.gap,
            target_types=dimension_candidate.target_types,
        ),
        "insight": _infer_structural_tension(
            target_dimension=dimension_candidate.dimension,
            target_gap=dimension_candidate.gap,
            target_types=dimension_candidate.target_types,
        ),
        "question": _infer_scene_specific_question(
            target_dimension=dimension_candidate.dimension,
            target_gap=dimension_candidate.gap,
            target_types=dimension_candidate.target_types,
            task_type=packet.constraints.task_type,
        ),
    }


def generate_round_output(packet: RoundPacket, llm_fn: LLMFunction) -> dict:
    from ftg.fathom import LLMRequest

    try:
        raw = llm_fn(
            LLMRequest(
                system_prompt=ROUND_PROTOCOL_SYSTEM_PROMPT,
                user_prompt=_build_round_packet_prompt(packet),
                json_mode=True,
                temperature=0.4,
            )
        )
        result = parse_llm_json(raw)
    except Exception:
        result = {}

    fallback = _fallback_round_output(packet)
    ask_mode = _clean_text(result.get("ask_mode", "")).lower()
    if ask_mode not in packet.constraints.allowed_modes:
        ask_mode = fallback["ask_mode"]

    target_gap = _clean_text(result.get("target_gap", ""))
    if not target_gap and packet.materials.dimension_candidate:
        target_gap = packet.materials.dimension_candidate.gap

    target_types = _normalize_target_types(result.get("target_types"))
    if not target_types and packet.materials.dimension_candidate:
        target_types = packet.materials.dimension_candidate.target_types

    response = _reframe_to_second_person(_clean_text(result.get("response", ""))) or fallback["response"]
    insight_candidate = _reframe_to_second_person(_clean_text(result.get("insight", "")))
    insight = insight_candidate if not _looks_generic_insight(insight_candidate) else ""
    if not insight:
        if ask_mode == "causal" and packet.materials.causal_candidate:
            hyp = CausalHypothesis(
                id=packet.materials.causal_candidate.hypothesis_id,
                source_node_id=packet.materials.causal_candidate.source_node_id,
                target_node_id=packet.materials.causal_candidate.target_node_id,
                source_content=packet.materials.causal_candidate.source_content,
                target_content=packet.materials.causal_candidate.target_content,
            )
            insight = _infer_structural_tension(
                target_dimension="",
                target_gap="",
                target_types=[],
                question_mode="causal",
                causal_hypothesis=hyp,
            )
        else:
            insight = _infer_structural_tension(
                target_dimension=packet.materials.dimension_candidate.dimension if packet.materials.dimension_candidate else "",
                target_gap=target_gap,
                target_types=target_types,
            )

    round_action = _clean_text(result.get("round_action", "")).lower()
    if round_action not in {"ask_user", "answer_now"}:
        round_action = "ask_user"

    question_candidate = _reframe_to_second_person(_clean_text(result.get("question", "")))
    if round_action == "answer_now":
        question = ""
    else:
        if (
            _looks_generic_question(question_candidate)
            or _looks_abstract_dimension_question(question_candidate)
            or (
                ask_mode == "causal"
                and _looks_overquoted_causal_question(question_candidate)
            )
        ):
            question_candidate = ""
        question = _ensure_question_mark(
            question_candidate
            or (
                _infer_causal_verification_question(
                    packet.materials.causal_candidate.source_content,
                    packet.materials.causal_candidate.target_content,
                )
                if ask_mode == "causal" and packet.materials.causal_candidate
                else _infer_scene_specific_question(
                    target_dimension=packet.materials.dimension_candidate.dimension if packet.materials.dimension_candidate else "",
                    target_gap=target_gap,
                    target_types=target_types,
                    task_type=packet.constraints.task_type,
                )
            )
        )

    return {
        "response": response,
        "insight": insight,
        "question": question,
        "question_mode": ask_mode,
        "round_action": round_action,
        "target_dimension": packet.materials.dimension_candidate.dimension if packet.materials.dimension_candidate else "",
        "target_gap": target_gap,
        "target_types": target_types,
        "verifying_hypothesis": packet.materials.causal_candidate.hypothesis_id if ask_mode == "causal" and packet.materials.causal_candidate else None,
    }


# ---------------------------------------------------------------------------
# Main question generation function
# ---------------------------------------------------------------------------

def generate_question(
    graph: IntentGraph,
    conversation_history: str,
    task_type: str,
    llm_fn: LLMFunction,
    round_count: int = 0,
    complexity: float = 0.5,
    waived_dimensions: set | None = None,
    causal_tracker: CausalTracker | None = None,
    question_mode: str = "dimension",
    target_dimension: str = "",
    causal_hypothesis: CausalHypothesis | None = None,
    redirect_context: dict | None = None,
    dimension_semantics: dict[str, str] | None = None,
    dimension_states: dict[str, dict[str, Any]] | None = None,
    dimension_bound_responses: dict[str, dict[str, Any]] | None = None,
    root_question: str = "",
    user_context: str = "",
    **_: Any,
) -> dict:
    """Backward-compatible wrapper around the RoundPacket question path."""

    if question_mode == "redirect_confirm" and redirect_context:
        orig_src = redirect_context.get("original_source", "A")
        orig_tgt = redirect_context.get("original_target", "B")
        user_alt = (
            redirect_context.get("proposed_cause_text", "")
            or redirect_context.get("user_response", "")
        )[:200]
        return {
            "response": _infer_minimal_progress(
                target_dimension="",
                target_gap="",
                target_types=[],
                question_mode="redirect_confirm",
                redirect_context=redirect_context,
            ),
            "insight": _infer_structural_tension(
                target_dimension="",
                target_gap="",
                target_types=[],
                question_mode="redirect_confirm",
                redirect_context=redirect_context,
            ),
            "question": _render_redirect_confirm_question(orig_src, orig_tgt, user_alt),
            "round_action": "ask_user",
            "question_mode": "causal",
            "target_dimension": "",
            "target_gap": "",
            "target_types": [],
            "verifying_hypothesis": redirect_context.get("hypothesis_id", ""),
        }

    bound_dimensions = set((dimension_bound_responses or {}).keys())
    effective_waived_dimensions = (waived_dimensions or set()) | bound_dimensions
    packet_root = root_question or ""

    if question_mode == "causal" and causal_hypothesis is not None:
        packet = RoundPacket(
            constraints=RoundConstraints(
                task_type=task_type,
                round_count=round_count,
                allowed_modes=["causal"],
            ),
            materials=RoundMaterials(
                root_question=packet_root,
                conversation_history=conversation_history,
                graph_summary=build_graph_summary(graph),
                causal_candidate=CausalCandidate(
                    hypothesis_id=causal_hypothesis.id,
                    source_node_id=causal_hypothesis.source_node_id,
                    target_node_id=causal_hypothesis.target_node_id,
                    source_content=causal_hypothesis.source_content,
                    target_content=causal_hypothesis.target_content,
                ),
                dimension_states=dict(dimension_states or {}),
                dimension_bound_responses=dict(dimension_bound_responses or {}),
                user_context=user_context,
            ),
        )
        return generate_round_output(packet, llm_fn)

    if not target_dimension or target_dimension in bound_dimensions:
        target_dimension = find_target_dimension(
            graph,
            complexity,
            effective_waived_dimensions,
            task_type=task_type,
        )

    dim_desc = (
        ((dimension_states or {}).get(target_dimension, {}) or {}).get("open_gap", "")
        or (dimension_semantics or {}).get(target_dimension)
        or DIMENSION_DESCRIPTIONS.get(target_dimension, "supplementary information relevant to the current task")
    )

    packet = RoundPacket(
        constraints=RoundConstraints(
            task_type=task_type,
            round_count=round_count,
            allowed_modes=["dimension"],
        ),
        materials=RoundMaterials(
            root_question=packet_root,
            conversation_history=conversation_history,
            graph_summary=build_graph_summary(graph),
            dimension_candidate=DimensionCandidate(
                dimension=target_dimension,
                gap=_clean_text(dim_desc),
            ),
            dimension_states=dict(dimension_states or {}),
            dimension_bound_responses=dict(dimension_bound_responses or {}),
            user_context=user_context,
        ),
    )
    return generate_round_output(packet, llm_fn)

