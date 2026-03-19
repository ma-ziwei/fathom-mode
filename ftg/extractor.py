"""
Information Extraction — extract structured nodes and edges from user text.

Pipeline: causal pre-filter → LLM extraction → validation → edge inference.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import Counter
from typing import TYPE_CHECKING, Callable

from ftg.dimensions import infer_edges
from ftg.graph import IntentGraph
from ftg.models import Dimension, DimensionCoverage, Edge, EdgeSource, Node, NodeType, RelationType
from ftg.utils import cosine_similarity, parse_llm_json

if TYPE_CHECKING:
    from ftg.fathom import LLMRequest

logger = logging.getLogger(__name__)

LLMFunction = Callable[["LLMRequest"], str]
EmbedFunction = Callable[[list[str]], list[list[float]]]


# ---------------------------------------------------------------------------
# Unified Understanding Prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are the information understanding engine in the Fathom-then-Generate system.
In a single pass, fully understand the user's text: extract information, classify dimensions, identify relationships, and detect bias signals.

For each piece of "new information", create a node:
  - id: a unique snake_case identifier, 2 to 4 words
  - content: distilled information content
  - raw_quote: exact quote from the user's original text
  - confidence: 0.0 to 1.0, how explicitly the user expressed this
  - node_type: fact|belief|value|intent|constraint|emotion|goal|assumption
  - dimension: one primary dimension, must be one of:
      who   = people, roles, agents, stakeholders, identities
      what  = core object, subject, event, category, content
      why   = purpose, motivation, reason, value driver, emotional intent
      when  = time, period, time window, deadline
      where = physical location, spatial context, geographic position
      how   = method, execution approach, style, constraints, risks, conditions
  - secondary_dimensions: other relevant dimensions, return [] if none

Dimension rules:
1. The QUESTION context hints which dimension this round's answer primarily addresses.
2. Purpose expressions like "in order to..." or "planning to use it for..." should be classified as why, not where.
3. Subject/object goes to what, method/approach goes to how.
4. where should only be used for real physical spaces or locations — do not overgeneralize.

If relationships exist between nodes (among new nodes, or between new and existing nodes), create edges:
  - source / target: node ids
  - relation_type: causal|dependency|contradiction|conditional|supports
  - source_type: "user_explicit" or "user_implied"

Causality rules:
- Only use causal + user_explicit when the user explicitly states a causal relationship.
- If it is merely correlation, support, or dependency, use supports or dependency.
- Never infer causation from correlation.

Cognitive bias labels to detect:
sunk_cost, loss_aversion, status_quo_bias, anchoring,
confirmation_bias, availability_heuristic, framing_effect,
endowment_effect

Hard requirements:
- Do not fabricate information; only extract what the user actually said.
- raw_quote must come from the user's original text.
- raw_quote must preserve the user's original language.
- All natural language fields must be in English:
  content, dimension_semantics, _reasoning
- Technical terms, proper nouns, product names, and game names may be kept as-is.

Task type classification (by collaboration mode, not content):
  thinking   = user needs help clarifying thoughts, making decisions, analyzing problems, or diagnosing issues
  creation   = user needs a deliverable produced — text, code, images, plans, or any artifact
  execution  = user needs actions performed in real systems — deploying, configuring, operating, sending
  learning   = user wants to build understanding of a domain, skill, or concept
  general    = not yet clear or does not fit the above

Examples of boundary cases:
  "Help me write an email" → creation (deliverable needed)
  "How should I phrase this to avoid offending them" → thinking (needs to reason first)
  "Send this email to the team" → execution (action in real system)
  "Teach me how to write professional emails" → learning (building skill)

Dimension coverage assessment:
After considering the entire knowledge graph, assess each of the 6 dimensions as:
  - "covered"
  - "covered_implicitly"
  - "not_relevant"
  - "missing"
Do not abuse not_relevant just to reduce follow-up questions.

Return strict JSON:
{
  "_reasoning": "Briefly explain your reasoning process in English",
  "task_type": "thinking|creation|execution|learning|general",
  "nodes": [
    {"id": "node_id", "content": "distilled information in English", "raw_quote": "user's original words",
     "confidence": 0.8, "node_type": "fact", "dimension": "what",
     "secondary_dimensions": ["when"]}
  ],
  "edges": [
    {"source": "node_a", "target": "node_b", "relation_type": "supports",
     "source_type": "user_implied"}
  ],
  "bias_updates": [
    {"node_id": "...", "bias_flags": ["anchoring"]}
  ],
  "dimension_assessment": {
    "who": "covered|covered_implicitly|not_relevant|missing",
    "what": "covered|covered_implicitly|not_relevant|missing",
    "why": "covered|covered_implicitly|not_relevant|missing",
    "when": "covered|covered_implicitly|not_relevant|missing",
    "where": "covered|covered_implicitly|not_relevant|missing",
    "how": "covered|covered_implicitly|not_relevant|missing"
  },
  "dimension_states": {
    "who": {"evidence_present": true, "coverage_level": "none|partial|sufficient|not_relevant", "supporting_node_ids": ["node_id"], "open_gap": "a concrete, scene-specific question about what is still unclear"},
    "what": {"evidence_present": true, "coverage_level": "none|partial|sufficient|not_relevant", "supporting_node_ids": ["node_id"], "open_gap": ""},
    "why": {"evidence_present": true, "coverage_level": "none|partial|sufficient|not_relevant", "supporting_node_ids": ["node_id"], "open_gap": ""},
    "when": {"evidence_present": true, "coverage_level": "none|partial|sufficient|not_relevant", "supporting_node_ids": ["node_id"], "open_gap": ""},
    "where": {"evidence_present": false, "coverage_level": "none|partial|sufficient|not_relevant", "supporting_node_ids": [], "open_gap": ""},
    "how": {"evidence_present": true, "coverage_level": "none|partial|sufficient|not_relevant", "supporting_node_ids": ["node_id"], "open_gap": ""}
  },
  "dimension_semantics": {
    "who": "One sentence explaining what 'who' means in this task",
    "what": "One sentence explaining what 'what' means in this task",
    "why": "One sentence explaining what 'why' means in this task",
    "when": "One sentence explaining what 'when' means in this task",
    "where": "One sentence explaining what 'where' means in this task",
    "how": "One sentence explaining what 'how' means in this task"
  }
}

dimension_semantics requirements:
- Must be written in the context of the current user task — use specific nouns, objects, and details from the user's question.
- Be concrete, not abstract. E.g., for a stock trading question, "who" should be "whether you're managing your own portfolio or advising a client", not "who is involved".

dimension_states requirements:
- These are fine-grained coverage states for the 6 dimensions, expressing "whether relevant evidence exists" and "overall coverage level".
- evidence_present indicates whether any relevant data points have surfaced for this dimension; even with evidence_present=true, coverage_level can still be partial.
- coverage_level must be one of:
  - none        = no available evidence currently
  - partial     = some relevant evidence exists, but the dimension is not fully explored
  - sufficient  = this dimension has enough information to support downstream understanding
  - not_relevant = this dimension is irrelevant to the current task
- supporting_node_ids may only reference node ids from this response's nodes or the existing graph.
- open_gap should be one concrete, scene-specific sentence about what is still unclear — e.g., "whether the user is deciding for themselves or advising someone else" rather than abstract descriptions like "who is involved". If coverage_level is sufficient or not_relevant, return an empty string.
- dimension_assessment should be consistent with dimension_states:
  - sufficient -> covered
  - partial -> covered_implicitly
  - none -> missing
  - not_relevant -> not_relevant

clarification_hints requirements:
- This is an advisory field within the same understanding task; it does not replace task_type, nodes, edges, or dimension_assessment.
- Only fill in clarification_hints when the user's input appears to "need a brief follow-up question before execution".
- Treat clarification_hints as part of the same understanding task as nodes/edges/task_type: you are identifying "what is the most critical semantic gap right now, and after asking, is the likely route direct, light, or deep".
- If the question can be answered directly, or you can only think of generic platitudes, return an empty object {}.
- clarification_target identifies "the most worthwhile gap to clarify with the user first", e.g.:
  - goal
  - criterion
  - scope
  - deliverable_spec
  - risk_constraint
  - continuation_anchor
- clarification_reason should briefly explain in one sentence why this gap is the most critical.
- draft_question must be a natural, on-topic follow-up question.
- draft_question should reuse specific nouns, objects, locations, products, tasks, or deliverables from the user's original question — do not reduce different questions to the same abstract template.
- Do not write generic platitudes like "would you prefer advice first or context first" unless that is genuinely the key distinction the user's question requires.
- These hints should be based primarily on the user's latest message, not on speculative graph state.
- route_hint_after_answer should only be filled when you are very confident; valid values are:
  - direct
  - deep
  - light
- If the user's message is expressing a preference like "just give me the result / just recommend / just give the conclusion", treat it as an answer to the clarification target, and give route_hint_after_answer=direct when very confident.
- If the user's answer reveals higher risk, higher complexity, or obvious multi-step task dependencies, give route_hint_after_answer=deep when very confident.
- route_hint_after_answer is advisory only — do not treat it as a mandatory routing command.

The JSON response may additionally include:
  "clarification_hints": {
    "clarification_target": "optional, goal|criterion|scope|deliverable_spec|risk_constraint|continuation_anchor",
    "clarification_reason": "optional, one sentence explaining why to ask about this",
    "subject": "optional, question subject",
    "subject_a": "optional, left-side subject for comparison",
    "subject_b": "optional, right-side subject for comparison",
    "contrast_a": "optional, first on-topic follow-up focus",
    "contrast_b": "optional, second on-topic follow-up focus",
    "draft_question": "optional, a natural follow-up question",
    "route_hint_after_answer": "optional, direct|deep|light"
  }

External context rules:
- If the user prompt contains an [External context] block, treat it only as background information.
- Do not create nodes directly from external context.
- Do not use external context as a raw_quote source.
- Nodes must come from the user's own expressions.
"""

INITIAL_ROUND_CORE_FIRST_APPENDIX = """

Initial round / empty graph priority rules:
- If the current knowledge graph is empty, your primary task is to preserve the user's main question, decision, or object.
- In nodes, you must prioritize keeping at least one intent or goal node that represents the main task.
- If information is insufficient, use a summary intent/goal node to anchor the main task — do not leave the first round's nodes empty.
- clarification_hints are advisory only; only consider filling them after securing the root anchor.
- If you are unsure about clarification_hints, return an empty object {} — do not omit the main task anchor for the sake of writing hints.
"""

INITIAL_ROUND_ROOT_ANCHOR_APPENDIX = """

Initial round / empty graph priority rules:
- If the current knowledge graph is empty, your primary task is to preserve the user's main question, decision, or object.
- In nodes, you must prioritize keeping at least one intent or goal node that represents the main task.
- If information is insufficient, use a summary intent/goal node to anchor the main task — do not leave the first round's nodes empty.
"""


# ---------------------------------------------------------------------------
# Speech Act Filter
# ---------------------------------------------------------------------------

SPEECH_ACT_TOKENS = frozenset({
    "ok", "okay", "yes", "yeah", "yep", "right", "sure", "correct",
    "got it", "understood", "fine", "alright", "absolutely", "definitely",
    "no", "nope", "not really", "wrong", "nah", "negative",
    "i see", "makes sense", "sounds good", "agreed", "exactly",
})

QUESTION_END_PATTERN = re.compile(r"[?!.…]+$")
QUESTION_SIGNAL_PATTERN = re.compile(
    r"\b(how|why|which|what|who|where|when|"
    r"should\s+i|do\s+i|can\s+i|is\s+it|are\s+there|"
    r"worth|recommend|compare|difference|better)\b",
    re.IGNORECASE,
)
DECISION_ROOT_PATTERNS = (
    re.compile(r"should\s+i\s+(?P<action>.+)$", re.IGNORECASE),
    re.compile(r"(?:do\s+i\s+need\s+to|do\s+i\s+have\s+to)\s+(?P<action>.+)$", re.IGNORECASE),
    re.compile(r"is\s+it\s+worth\s+(?:it\s+to\s+)?(?P<action>.+)$", re.IGNORECASE),
    re.compile(r"(?:shall|would)\s+(?:i|we)\s+(?P<action>.+)$", re.IGNORECASE),
)


def is_speech_act(content: str) -> bool:
    """Filter out pure confirmation/denial speech acts with no new info."""
    cleaned = content.strip().rstrip(".!?~,")
    return len(cleaned) <= 12 and cleaned.lower() in SPEECH_ACT_TOKENS


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_content_quality(node: Node) -> bool:
    """Returns True if node passes quality checks."""
    content = node.content.strip()
    if len(content) < 3:
        return False
    if not any(c.isalnum() for c in content):
        return False
    return True


def validate_raw_quote(node: Node, user_text: str) -> bool:
    """Returns True if raw_quote actually appears in the user's text."""
    if not node.raw_quote or len(node.raw_quote.strip()) < 2:
        logger.debug("Node '%s' has empty/short raw_quote — accepted without validation", node.id)
        return True
    normalized_text = " ".join(user_text.split()).lower()
    normalized_raw = " ".join(node.raw_quote.split()).lower()
    if normalized_raw in normalized_text:
        return True
    raw_words = set(normalized_raw.split())
    text_words = set(normalized_text.split())
    if raw_words and len(raw_words & text_words) / len(raw_words) >= 0.5:
        return True
    compact_text = _compact_cjk_text(user_text)
    compact_raw = _compact_cjk_text(node.raw_quote)
    if compact_raw and compact_text and _contains_cjk(compact_raw + compact_text):
        if compact_raw in compact_text:
            return True
        overlap = _char_overlap_ratio(compact_raw, compact_text)
        subseq = _ordered_subsequence_ratio(compact_raw, compact_text)
        if len(compact_raw) >= 4 and overlap >= 0.7 and subseq >= 0.7:
            return True
        if len(compact_raw) >= 6 and overlap >= 0.6 and subseq >= 0.8:
            return True
    return False


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _compact_cjk_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def _char_overlap_ratio(raw_text: str, full_text: str) -> float:
    if not raw_text:
        return 0.0
    counts = Counter(full_text)
    shared = 0
    for ch in raw_text:
        if counts[ch] > 0:
            shared += 1
            counts[ch] -= 1
    return shared / max(1, len(raw_text))


def _ordered_subsequence_ratio(raw_text: str, full_text: str) -> float:
    if not raw_text:
        return 0.0
    idx = 0
    for ch in full_text:
        if idx < len(raw_text) and ch == raw_text[idx]:
            idx += 1
    return idx / max(1, len(raw_text))


def _has_root_anchor(nodes: list[Node]) -> bool:
    return any(node.node_type in {NodeType.INTENT, NodeType.GOAL} for node in nodes)


def _strip_terminal_punctuation(text: str) -> str:
    return QUESTION_END_PATTERN.sub("", text.strip())


def _clean_anchor_fragment(text: str) -> str:
    cleaned = _strip_terminal_punctuation(text)
    cleaned = re.sub(r"^(just|now|currently)\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,;:.?!")


def _infer_root_anchor_node(user_text: str) -> Node | None:
    cleaned_full = _strip_terminal_punctuation(user_text)
    if len(cleaned_full) < 4:
        return None

    for pattern in DECISION_ROOT_PATTERNS:
        match = pattern.search(cleaned_full)
        if not match:
            continue
        action = _clean_anchor_fragment(match.group("action"))
        if not action:
            continue
        return Node(
            id=f"root_decision_{uuid.uuid4().hex[:6]}",
            content=f"User is considering whether to {action}",
            raw_quote=action,
            confidence=0.95,
            node_type=NodeType.GOAL,
            dimension=Dimension.WHAT,
            secondary_dimensions=["how"],
        )

    if "?" in user_text or QUESTION_SIGNAL_PATTERN.search(cleaned_full):
        return Node(
            id=f"root_question_{uuid.uuid4().hex[:6]}",
            content=f"User's question: {cleaned_full}",
            raw_quote=cleaned_full,
            confidence=0.9,
            node_type=NodeType.INTENT,
            dimension=Dimension.WHAT,
        )

    return None


def _infer_followup_goal_node(user_text: str, graph: IntentGraph) -> Node | None:
    cleaned_full = _strip_terminal_punctuation(user_text)
    if len(cleaned_full) < 4:
        return None
    if not graph.get_all_nodes():
        return None
    if not ("?" in user_text or QUESTION_SIGNAL_PATTERN.search(cleaned_full)):
        return None

    return Node(
        id=f"followup_goal_{uuid.uuid4().hex[:6]}",
        content=f"User currently wants to know: {cleaned_full}",
        raw_quote=cleaned_full,
        confidence=0.88,
        node_type=NodeType.GOAL,
        dimension=Dimension.WHY,
        secondary_dimensions=["what"],
    )


def _infer_followup_answer_node(user_text: str, graph: IntentGraph) -> Node | None:
    """Rescue a node from a user's answer when LLM extraction returns 0 nodes."""
    cleaned = _strip_terminal_punctuation(user_text)
    if len(cleaned) < 4:
        return None
    if not graph.get_all_nodes():
        return None
    return Node(
        id=f"followup_answer_{uuid.uuid4().hex[:6]}",
        content=cleaned,
        raw_quote=cleaned,
        confidence=0.80,
        node_type=NodeType.FACT,
        dimension=Dimension.WHAT,
        secondary_dimensions=[],
    )


def _pick_followup_anchor_node(graph: IntentGraph) -> Node | None:
    for node_type in (NodeType.INTENT, NodeType.GOAL):
        nodes = graph.get_nodes_by_type(node_type)
        if nodes:
            return nodes[0]
    nodes = graph.get_all_nodes()
    return nodes[0] if nodes else None


def _build_extraction_system_prompt(
    *,
    initial_round: bool,
    hard_causal_edges: list[dict],
    include_clarification_hints: bool = False,
) -> str:
    system_prompt = EXTRACTION_SYSTEM_PROMPT
    if not include_clarification_hints:
        system_prompt = _strip_clarification_sections(system_prompt)
    if initial_round:
        system_prompt += (
            INITIAL_ROUND_CORE_FIRST_APPENDIX
            if include_clarification_hints
            else INITIAL_ROUND_ROOT_ANCHOR_APPENDIX
        )
    if hard_causal_edges:
        causal_context = "\n".join(
            f"  - \"{he['cause']}\" {he.get('marker', '->')} \"{he['effect']}\" "
            f"[MUST preserve as CAUSAL + USER_EXPLICIT]"
            for he in hard_causal_edges
        )
        system_prompt += (
            f"\n\nDETERMINISTIC CAUSAL EDGES DETECTED (from user's language):\n"
            f"{causal_context}\n"
            f"You MUST include these as causal edges with source_type='user_explicit'."
        )
    return system_prompt


def _strip_clarification_sections(system_prompt: str) -> str:
    start_marker = "clarification_hints requirements:"
    end_marker = "External context rules:"
    start = system_prompt.find(start_marker)
    end = system_prompt.find(end_marker)
    if start != -1 and end != -1 and end > start:
        system_prompt = system_prompt[:start] + system_prompt[end:]
    return system_prompt


def deduplicate_nodes(
    new_nodes: list[Node],
    existing_nodes: list[Node],
    embed_fn: EmbedFunction | None = None,
    threshold: float = 0.92,
    hard_edges: list[dict] | None = None,
) -> list[Node]:
    """
    Semantic deduplication with Topological Lock.

    TOPOLOGICAL LOCK: Nodes referenced by deterministic causal hard_edges
    are FORBIDDEN from being merged, regardless of embedding similarity.
    If "lacked oversight" and "failed audit" have similarity > 0.92,
    naive dedup merges them. Then edge (lacked_oversight -> failed_audit)
    becomes a self-loop, deleted by DAG validation. The causal relationship
    is silently destroyed. The lock prevents this.
    """
    if not new_nodes:
        return new_nodes

    locked_contents: set[str] = set()
    if hard_edges:
        for he in hard_edges:
            cause = he.get("cause", "").lower().strip()
            effect = he.get("effect", "").lower().strip()
            if cause:
                locked_contents.add(cause)
            if effect:
                locked_contents.add(effect)

    existing_contents = {n.content.lower().strip() for n in existing_nodes}

    # Fast path: exact content dedup
    unique_by_text: list[Node] = []
    for node in new_nodes:
        key = node.content.lower().strip()
        is_locked = any(
            (lc in key or key in lc) for lc in locked_contents
        ) if locked_contents else False

        if is_locked:
            unique_by_text.append(node)
            existing_contents.add(key)
        elif key not in existing_contents:
            unique_by_text.append(node)
            existing_contents.add(key)

    # Embedding-based semantic dedup (works for existing-vs-new AND new-vs-new)
    if not embed_fn or not unique_by_text:
        return unique_by_text

    try:
        pool_vecs: list[list[float]] = []
        if existing_nodes:
            pool_vecs = embed_fn([n.content for n in existing_nodes])

        unique: list[Node] = []
        for node in unique_by_text:
            node_lower = node.content.lower().strip()
            is_locked = any(
                (lc in node_lower or node_lower in lc) for lc in locked_contents
            ) if locked_contents else False

            if is_locked:
                unique.append(node)
                pool_vecs.append(embed_fn([node.content])[0])
                continue

            node_vec = embed_fn([node.content])[0]
            is_dup = any(
                cosine_similarity(node_vec, pv) > threshold
                for pv in pool_vecs
            )
            if not is_dup:
                unique.append(node)
                pool_vecs.append(node_vec)
        return unique
    except Exception as exc:
        logger.warning("Embedding dedup failed (fallback=allow): %s", exc)
        return unique_by_text


def validate_edges(edges: list[Edge]) -> list[Edge]:
    """Deterministic edge validation: no self-loops, no dupes, CAUSAL invariant."""
    validated = []
    seen: set[tuple] = set()
    for edge in edges:
        if edge.source == edge.target:
            continue
        key = (edge.source, edge.target, edge.relation_type)
        if key in seen:
            continue
        if (edge.relation_type == RelationType.CAUSAL
                and edge.source_type != EdgeSource.USER_EXPLICIT):
            edge = Edge(
                source=edge.source,
                target=edge.target,
                relation_type=RelationType.SUPPORTS,
                source_type=edge.source_type,
            )
            key = (edge.source, edge.target, edge.relation_type)
        seen.add(key)
        validated.append(edge)
    return validated


def parse_clarification_hints(result: dict) -> dict[str, str]:
    """Parse optional advisory clarification hints without blocking core extraction."""
    raw = result.get("clarification_hints", {})
    if not isinstance(raw, dict):
        return {}
    allowed_keys = {
        "clarification_target",
        "clarification_reason",
        "subject",
        "subject_a",
        "subject_b",
        "contrast_a",
        "contrast_b",
        "draft_question",
        "route_hint_after_answer",
    }
    hints: dict[str, str] = {}
    for key in allowed_keys:
        value = raw.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                hints[key] = cleaned
    draft = hints.get("draft_question")
    if draft and (len(draft) < 8 or len(draft) > 120 or "\n" in draft):
        hints.pop("draft_question", None)
    target = hints.get("clarification_target")
    if target and target not in {
        "goal",
        "criterion",
        "scope",
        "deliverable_spec",
        "risk_constraint",
        "continuation_anchor",
    }:
        hints.pop("clarification_target", None)
    route_hint = hints.get("route_hint_after_answer")
    if route_hint and route_hint not in {"direct", "deep", "light"}:
        hints.pop("route_hint_after_answer", None)
    return hints


def _assessment_from_coverage(level: str) -> str:
    mapping = {
        DimensionCoverage.SUFFICIENT.value: "covered",
        DimensionCoverage.PARTIAL.value: "covered_implicitly",
        DimensionCoverage.NONE.value: "missing",
        DimensionCoverage.NOT_RELEVANT.value: "not_relevant",
    }
    return mapping.get(level, "missing")


def _coverage_from_assessment(status: str) -> str:
    mapping = {
        "covered": DimensionCoverage.SUFFICIENT.value,
        "covered_implicitly": DimensionCoverage.PARTIAL.value,
        "missing": DimensionCoverage.NONE.value,
        "not_relevant": DimensionCoverage.NOT_RELEVANT.value,
    }
    return mapping.get(status, DimensionCoverage.NONE.value)


def _empty_dimension_state() -> dict[str, object]:
    return {
        "evidence_present": False,
        "coverage_level": DimensionCoverage.NONE.value,
        "supporting_node_ids": [],
        "open_gap": "",
    }


def _parse_dimension_states(
    raw_states: object,
    *,
    valid_node_ids: set[str],
    raw_assessment: dict[str, str],
    dimension_semantics: dict[str, str],
    graph: IntentGraph,
    raw_nodes: list[Node],
) -> dict[str, dict[str, object]]:
    states: dict[str, dict[str, object]] = {dim.value: _empty_dimension_state() for dim in Dimension}
    dim_names = {d.value for d in Dimension}
    coverage_values = {c.value for c in DimensionCoverage}

    if isinstance(raw_states, dict):
        for dim in dim_names:
            payload = raw_states.get(dim, {})
            if not isinstance(payload, dict):
                continue
            evidence_present = bool(payload.get("evidence_present", False))
            coverage_level = payload.get("coverage_level", "")
            if coverage_level not in coverage_values:
                coverage_level = _coverage_from_assessment(raw_assessment.get(dim, "missing"))
            supporting_node_ids = payload.get("supporting_node_ids", [])
            if not isinstance(supporting_node_ids, list):
                supporting_node_ids = []
            supporting_node_ids = [
                node_id for node_id in supporting_node_ids
                if isinstance(node_id, str) and node_id in valid_node_ids
            ]
            open_gap = payload.get("open_gap", "")
            if not isinstance(open_gap, str):
                open_gap = ""
            states[dim] = {
                "evidence_present": evidence_present,
                "coverage_level": coverage_level,
                "supporting_node_ids": supporting_node_ids,
                "open_gap": open_gap.strip(),
            }
        return states

    combined_nodes = list(graph.get_all_nodes()) + list(raw_nodes)
    for dim in dim_names:
        support_ids: list[str] = []
        for node in combined_nodes:
            primary = node.dimension.value if node.dimension else ""
            secondaries = list(node.secondary_dimensions or [])
            if dim == primary or dim in secondaries:
                support_ids.append(node.id)
        support_ids = list(dict.fromkeys(node_id for node_id in support_ids if node_id in valid_node_ids))
        evidence_present = bool(support_ids)
        status = raw_assessment.get(dim, "missing")
        coverage_level = _coverage_from_assessment(status)
        if coverage_level == DimensionCoverage.NONE.value and evidence_present:
            coverage_level = DimensionCoverage.PARTIAL.value
        open_gap = ""
        if coverage_level in {DimensionCoverage.NONE.value, DimensionCoverage.PARTIAL.value}:
            open_gap = dimension_semantics.get(dim, "")
        states[dim] = {
            "evidence_present": evidence_present,
            "coverage_level": coverage_level,
            "supporting_node_ids": support_ids,
            "open_gap": open_gap.strip(),
        }
    return states


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract(
    user_text: str,
    graph: IntentGraph,
    llm_fn: LLMFunction,
    embed_fn: EmbedFunction | None = None,
    conversation_context: str = "",
    target_dimension_context: str = "",
    task_type: str = "",
    causal_markers_fn=None,
    match_markers_fn=None,
    user_context: str = "",
    include_clarification_hints: bool = False,
) -> tuple[list[Node], list[Edge], str | None, dict[str, str], dict[str, dict[str, object]], dict[str, str], dict[str, str]]:
    """
    Extract nodes and edges using the Unified Understanding architecture.

    Returns:
      (nodes, edges, task_type_or_none, dimension_assessment, dimension_states, dimension_semantics, clarification_hints)
    """
    from ftg.fathom import LLMRequest

    # Step 0: Causal pre-filter (regex markers)
    hard_causal_edges: list[dict] = []
    if causal_markers_fn:
        hard_causal_edges, _ = causal_markers_fn(user_text)

    # Step 1: LLM extraction
    system_prompt = _build_extraction_system_prompt(
        initial_round=graph.node_count() == 0,
        hard_causal_edges=hard_causal_edges,
        include_clarification_hints=include_clarification_hints,
    )

    user_prompt_parts = [f"User text: {user_text}"]
    if conversation_context:
        user_prompt_parts.insert(0, f"Conversation so far:\n{conversation_context}\n---")
    if target_dimension_context:
        user_prompt_parts.insert(0, f"[Question targeted dimension: {target_dimension_context}]")
    if task_type:
        user_prompt_parts.insert(0, f"[Current task type: {task_type}]")
    if user_context:
        user_prompt_parts.insert(0, f"[External context provided by caller]:\n{user_context}\n---")

    # Current graph context
    existing_nodes = graph.get_all_nodes()
    if existing_nodes:
        node_lines = [f"  {n.id}: [{n.dimension.value if n.dimension else '?'}] {n.content}"
                      for n in existing_nodes[:20]]
        user_prompt_parts.append(f"\nExisting knowledge graph nodes:\n" + "\n".join(node_lines))

    user_prompt = "\n".join(user_prompt_parts)

    try:
        raw_response = llm_fn(LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            temperature=0.2,
        ))
        result = parse_llm_json(raw_response)
        logger.debug("Extraction raw LLM response (first 500 chars): %s", raw_response[:500])
    except Exception as exc:
        logger.error(
            "Extraction LLM call FAILED — 0 nodes will be extracted. "
            "Cause: %s (%s)",
            exc, type(exc).__name__,
            exc_info=True,
        )
        result = {"nodes": [], "edges": [], "bias_updates": []}

    result.pop("_reasoning", None)

    # Parse task_type
    VALID_TASK_TYPES = {"thinking", "creation", "execution", "learning", "general"}
    llm_task_type = result.get("task_type", "")
    detected_task_type = llm_task_type if llm_task_type in VALID_TASK_TYPES else None

    # --- Advisory fields (parse failures MUST NOT block core flow) ---
    VALID_ASSESSMENTS = {"covered", "covered_implicitly", "not_relevant", "missing"}
    raw_assessment = result.get("dimension_assessment", {})
    dim_names = {d.value for d in Dimension}
    dimension_semantics: dict[str, str] = {}
    try:
        raw_semantics = result.get("dimension_semantics", {})
        if isinstance(raw_semantics, dict):
            for k, v in raw_semantics.items():
                if k in dim_names and isinstance(v, str) and v.strip():
                    dimension_semantics[k] = v.strip()
    except Exception as exc:
        logger.warning("Advisory parse (dimension_semantics) failed: %s", exc)

    clarification_hints: dict[str, str] = {}
    if include_clarification_hints:
        try:
            clarification_hints = parse_clarification_hints(result)
        except Exception as exc:
            logger.warning("Advisory parse (clarification_hints) failed: %s", exc)

    # Parse nodes
    raw_nodes: list[Node] = []
    for nd in result.get("nodes", []):
        try:
            node_id = nd.get("id", f"node_{uuid.uuid4().hex[:6]}")
            if graph.get_node(node_id):
                node_id = f"{node_id}_{uuid.uuid4().hex[:4]}"

            content = nd.get("content", "")
            if not content:
                continue

            llm_dimension = nd.get("dimension", "")
            try:
                dimension = Dimension(llm_dimension) if llm_dimension else Dimension.WHAT
            except ValueError:
                dimension = Dimension.WHAT

            valid_secondary = []
            for sd in nd.get("secondary_dimensions", []):
                try:
                    Dimension(sd)
                    if sd != dimension.value:
                        valid_secondary.append(sd)
                except ValueError:
                    continue

            raw_node_type = nd.get("node_type", "fact")
            try:
                node_type = NodeType(raw_node_type)
            except ValueError:
                logger.debug("Unknown node_type '%s', defaulting to 'fact'", raw_node_type)
                node_type = NodeType.FACT

            node = Node(
                id=node_id,
                content=content,
                raw_quote=nd.get("raw_quote", nd.get("raw_input", "")),
                confidence=float(nd.get("confidence", 0.5)),
                node_type=node_type,
                dimension=dimension,
                bias_flags=nd.get("bias_flags", []),
                secondary_dimensions=valid_secondary,
            )

            if is_speech_act(content):
                continue

            raw_nodes.append(node)
        except (ValueError, KeyError) as exc:
            logger.debug("Node parse failed: %s — raw: %s", exc, nd)
            continue

    # Parse edges
    llm_edges: list[Edge] = []
    valid_ids = {n.id for n in graph.get_all_nodes()} | {n.id for n in raw_nodes}
    for ed in result.get("edges", []):
        try:
            source = ed.get("source", "")
            target = ed.get("target", "")
            if source in valid_ids and target in valid_ids:
                st_str = ed.get("source_type", "user_implied")
                try:
                    src_type = EdgeSource(st_str)
                except ValueError:
                    src_type = EdgeSource.USER_IMPLIED
                edge = Edge(
                    source=source,
                    target=target,
                    relation_type=RelationType(ed.get("relation_type", "supports")),
                    source_type=src_type,
                )
                llm_edges.append(edge)
        except (ValueError, KeyError):
            continue

    # Apply bias updates (advisory — failure does not block core flow)
    try:
        for bu in result.get("bias_updates", []):
            nid = bu.get("node_id", "")
            flags = bu.get("bias_flags", [])
            if flags:
                for n in raw_nodes:
                    if n.id == nid:
                        n.bias_flags.extend(flags)
                        break
                else:
                    existing = graph.get_node(nid)
                    if existing:
                        existing.bias_flags.extend(flags)
    except Exception as exc:
        logger.warning("Advisory parse (bias_updates) failed: %s", exc)

    # Step 2: Validation pipeline
    logger.debug("Nodes after parsing: %d — %s", len(raw_nodes), [(n.id, n.content[:50]) for n in raw_nodes])
    pre_quality = len(raw_nodes)
    raw_nodes = [n for n in raw_nodes if validate_content_quality(n)]
    if len(raw_nodes) != pre_quality:
        logger.debug("validate_content_quality dropped %d nodes", pre_quality - len(raw_nodes))
    pre_quote = len(raw_nodes)
    raw_nodes = [n for n in raw_nodes if validate_raw_quote(n, user_text)]
    if len(raw_nodes) != pre_quote:
        logger.debug("validate_raw_quote dropped %d nodes", pre_quote - len(raw_nodes))
    if graph.node_count() == 0 and not _has_root_anchor(raw_nodes):
        rescued_anchor = _infer_root_anchor_node(user_text)
        if rescued_anchor is not None:
            raw_nodes.append(rescued_anchor)
            logger.debug(
                "Initial root anchor rescued: %s | %s",
                rescued_anchor.id,
                rescued_anchor.content,
            )
    if graph.node_count() > 0 and not raw_nodes:
        rescued_followup = _infer_followup_goal_node(user_text, graph)
        if rescued_followup is None:
            rescued_followup = _infer_followup_answer_node(user_text, graph)
        if rescued_followup is not None:
            raw_nodes.append(rescued_followup)
            anchor = _pick_followup_anchor_node(graph)
            if anchor is not None:
                llm_edges.append(
                    Edge(
                        source=rescued_followup.id,
                        target=anchor.id,
                        relation_type=RelationType.SUPPORTS,
                        source_type=EdgeSource.USER_IMPLIED,
                    )
                )
            logger.debug(
                "Follow-up rescued: %s | %s",
                rescued_followup.id,
                rescued_followup.content,
            )
    pre_dedup = len(raw_nodes)
    raw_nodes = deduplicate_nodes(
        raw_nodes, list(graph.get_all_nodes()),
        embed_fn=embed_fn,
        hard_edges=hard_causal_edges,
    )
    if len(raw_nodes) != pre_dedup:
        logger.debug("deduplicate_nodes dropped %d nodes", pre_dedup - len(raw_nodes))
    llm_edges = validate_edges(llm_edges)

    # Step 3: Edge inference + causal marker edges
    # All candidate edges are collected and sent to graph.add_edge() which
    # applies strongest-edge-wins canonicalization. No pair-level filtering
    # here — a weak LLM edge must not block a stronger deterministic edge.
    inferred_edges = infer_edges(raw_nodes, graph)
    all_edges = list(llm_edges) + list(inferred_edges)

    if causal_markers_fn and match_markers_fn:
        hard_markers, _soft_markers = causal_markers_fn(user_text)
        if hard_markers and raw_nodes:
            causal_edges = match_markers_fn(hard_markers, raw_nodes, graph)
            all_edges.extend(causal_edges)

    valid_node_ids = {node.id for node in graph.get_all_nodes()} | {node.id for node in raw_nodes}
    dimension_states = _parse_dimension_states(
        result.get("dimension_states"),
        valid_node_ids=valid_node_ids,
        raw_assessment=raw_assessment if isinstance(raw_assessment, dict) else {},
        dimension_semantics=dimension_semantics,
        graph=graph,
        raw_nodes=raw_nodes,
    )

    dimension_assessment: dict[str, str] = {}
    if isinstance(raw_assessment, dict):
        for dim_val in [d.value for d in Dimension]:
            state_level = dimension_states.get(dim_val, {}).get("coverage_level", "")
            if state_level:
                dimension_assessment[dim_val] = _assessment_from_coverage(str(state_level))
                continue
            val = raw_assessment.get(dim_val, "missing")
            dimension_assessment[dim_val] = val if val in VALID_ASSESSMENTS else "missing"
    else:
        for dim_val in [d.value for d in Dimension]:
            state_level = dimension_states.get(dim_val, {}).get("coverage_level", "")
            dimension_assessment[dim_val] = _assessment_from_coverage(str(state_level))

    return (
        raw_nodes,
        all_edges,
        detected_task_type,
        dimension_assessment,
        dimension_states,
        dimension_semantics,
        clarification_hints,
    )
