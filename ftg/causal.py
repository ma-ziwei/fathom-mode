"""
Causal verification — detects and validates causal relationships.

CAUSAL edges are never created autonomously. They must be either detected
from user's explicit language markers or confirmed via verification questions.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Callable

from ftg.graph import IntentGraph
from ftg.models import (
    CausalHypothesis, Dimension, Edge, EdgeSource, Node, NodeType, RelationType,
)
from ftg.utils import cosine_similarity

if TYPE_CHECKING:
    from ftg.fathom import LLMRequest

logger = logging.getLogger(__name__)

EmbedFunction = Callable[[list[str]], list[list[float]]]
LLMFunction = Callable[["LLMRequest"], str]


# ---------------------------------------------------------------------------
# Embedding Cache — avoids O(n^2) API calls for pairwise similarity
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Session-level cache for embedding vectors. Batch-embeds missing texts."""

    def __init__(self, embed_fn: EmbedFunction) -> None:
        self._fn = embed_fn
        self._cache: dict[str, list[float]] = {}

    def get_many(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            try:
                vecs = self._fn(missing)
                for text, vec in zip(missing, vecs):
                    self._cache[text] = vec
            except Exception as exc:
                logger.warning("Embedding batch failed: %s", exc)
                return []
        return [self._cache[t] for t in texts if t in self._cache]

    def get_one(self, text: str) -> list[float] | None:
        result = self.get_many([text])
        return result[0] if result else None

    def similarity(self, text_a: str, text_b: str) -> float:
        vecs = self.get_many([text_a, text_b])
        if len(vecs) < 2:
            return 0.0
        return cosine_similarity(vecs[0], vecs[1])


# ---------------------------------------------------------------------------
# Causal Marker Detection
# ---------------------------------------------------------------------------

FORWARD_MARKERS_EN = [
    "therefore", "causes", "leads to", "results in",
    "which means", "consequently",
]
BACKWARD_MARKERS_EN = [
    "because", "due to", "as a result of",
    "caused by", "owing to",
]
PURPOSE_MARKERS_EN = [
    "in order to", "so that", "for the purpose of",
    "intended for", "to be used as", "used for",
]

ALL_CAUSAL_MARKERS = (
    FORWARD_MARKERS_EN + BACKWARD_MARKERS_EN + PURPOSE_MARKERS_EN
)

CORRELATION_RISK_THRESHOLD = 0.55


def detect_causal_markers(text: str) -> tuple[list[dict], list[dict]]:
    """
    Extract causal relationships from text using language markers.

    Returns: (hard_edges, soft_edges)
    - hard_edges: high confidence — injected as constraints for LLM
    - soft_edges: low confidence — logged only
    """
    hard_edges: list[dict] = []
    soft_edges: list[dict] = []
    lower = text.lower()

    all_forward = FORWARD_MARKERS_EN
    all_backward = BACKWARD_MARKERS_EN
    all_purpose = PURPOSE_MARKERS_EN

    # Forward markers: cause MARKER effect
    for marker in all_forward:
        if marker in lower:
            idx = lower.index(marker)
            cause = text[:idx].strip()
            effect = text[idx + len(marker):].strip()
            if cause and effect:
                hard_edges.append({
                    "cause": _clean_fragment(cause),
                    "effect": _clean_fragment(effect),
                    "marker": marker.strip(),
                    "direction": "forward",
                })

    # Backward markers: effect MARKER cause
    if not hard_edges:
        for marker in all_backward:
            if marker in lower:
                idx = lower.index(marker)
                effect = text[:idx].strip()
                cause = text[idx + len(marker):].strip()
                if cause and effect:
                    hard_edges.append({
                        "cause": _clean_fragment(cause),
                        "effect": _clean_fragment(effect),
                        "marker": marker.strip(),
                        "direction": "backward",
                    })

    # Purpose markers — soft only, never promoted to hard causal edges
    for marker in all_purpose:
        if marker in lower:
            idx = lower.index(marker)
            context = text[:idx].strip()
            purpose = text[idx + len(marker):].strip()
            if purpose:
                soft_edges.append({
                    "cause": _clean_fragment(purpose),
                    "effect": _clean_fragment(context) if context else "(action)",
                    "marker": marker.strip(),
                    "direction": "purpose",
                })

    # Deduplicate
    seen: set[tuple] = set()
    unique_hard: list[dict] = []
    for r in hard_edges:
        key = (r["cause"].lower(), r["effect"].lower())
        if key not in seen:
            seen.add(key)
            unique_hard.append(r)

    return unique_hard, soft_edges


def _clean_fragment(text: str) -> str:
    text = text.strip()
    for prefix in [",", ".", "and ", "then "]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    for suffix in [",", "."]:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()
    return text


def match_markers_to_nodes(
    markers: list[dict],
    nodes: list[Node],
    graph: IntentGraph,
) -> list[Edge]:
    """Match detected causal markers to extracted nodes. Creates USER_EXPLICIT CAUSAL edges."""
    edges: list[Edge] = []
    all_nodes = list(nodes) + list(graph.get_all_nodes())

    for marker_info in markers:
        cause_text = marker_info["cause"].lower()
        effect_text = marker_info["effect"].lower()

        best_cause: Node | None = None
        best_effect: Node | None = None
        best_cause_score = 0.0
        best_effect_score = 0.0

        for node in all_nodes:
            content_lower = node.content.lower()
            raw_lower = node.raw_quote.lower() if node.raw_quote else ""

            cause_score = _overlap_score(cause_text, content_lower, raw_lower)
            effect_score = _overlap_score(effect_text, content_lower, raw_lower)

            if cause_score > best_cause_score:
                best_cause_score = cause_score
                best_cause = node
            if effect_score > best_effect_score:
                best_effect_score = effect_score
                best_effect = node

        if (best_cause and best_effect
                and best_cause.id != best_effect.id
                and best_cause_score > 0.3
                and best_effect_score > 0.3):
            edges.append(Edge(
                source=best_cause.id,
                target=best_effect.id,
                relation_type=RelationType.CAUSAL,
                source_type=EdgeSource.USER_EXPLICIT,
            ))

    return edges


def _overlap_score(query: str, content: str, raw: str) -> float:
    if not query:
        return 0.0
    if query in content or query in raw:
        return 1.0
    if content in query or raw in query:
        return 0.8
    query_words = set(query.split())
    content_words = set(content.split()) | set(raw.split())
    if not query_words:
        return 0.0
    return len(query_words & content_words) / max(len(query_words), 1)


def _has_causal_markers_between(node_a: Node, node_b: Node) -> bool:
    """Check if user's raw text contains causal markers."""
    combined = (node_a.raw_quote + " " + node_b.raw_quote).lower()
    return any(m in combined for m in ALL_CAUSAL_MARKERS)


# ---------------------------------------------------------------------------
# Causal Worthiness Filter
# ---------------------------------------------------------------------------

TRIVIAL_PAIRS: set[tuple[Dimension, Dimension]] = {
    (Dimension.HOW, Dimension.WHAT),
    (Dimension.HOW, Dimension.WHERE),
    (Dimension.WHEN, Dimension.WHAT),
    (Dimension.WHEN, Dimension.HOW),
    (Dimension.WHERE, Dimension.WHEN),
}

HIGH_VALUE_PAIRS: set[tuple[Dimension, Dimension]] = {
    (Dimension.WHY, Dimension.WHAT),
    (Dimension.WHY, Dimension.HOW),
    (Dimension.WHY, Dimension.WHO),
    (Dimension.WHY, Dimension.WHERE),
}

SUBJECTIVE_NODE_TYPES = {NodeType.BELIEF, NodeType.ASSUMPTION, NodeType.VALUE, NodeType.EMOTION}

LOW_CONFIDENCE_THRESHOLD = 0.7


def is_worth_verifying(source: Node, target: Node) -> bool:
    """
    Pre-filter: should this node pair go through causal verification?

    Checks (in order):
      1. Subjective node types (BELIEF/ASSUMPTION/VALUE/EMOTION) → always verify
      2. Explicit causal markers already present → skip (already captured)
      3. Low confidence on either node → verify (LLM unsure = worth asking)
      4. Trivial dimension pairs (HOW↔WHAT etc.) → skip
      5. High-value dimension pairs (WHY↔WHAT etc.) → verify
      6. Default → skip
    """
    if source.node_type in SUBJECTIVE_NODE_TYPES or target.node_type in SUBJECTIVE_NODE_TYPES:
        return True

    if _has_causal_markers_between(source, target):
        return False

    if source.confidence < LOW_CONFIDENCE_THRESHOLD or target.confidence < LOW_CONFIDENCE_THRESHOLD:
        return True

    if not source.dimension or not target.dimension:
        return False

    pair = (source.dimension, target.dimension)
    rev = (target.dimension, source.dimension)

    if pair in TRIVIAL_PAIRS or rev in TRIVIAL_PAIRS:
        return False

    if pair in HIGH_VALUE_PAIRS or rev in HIGH_VALUE_PAIRS:
        return True

    return False


# ---------------------------------------------------------------------------
# b) Causal Hypothesis Tracker
# ---------------------------------------------------------------------------


class CausalTracker:
    """
    Manages causal hypotheses — proposed causal relationships that
    live outside the graph until confirmed by the user.
    """

    def __init__(self) -> None:
        self._hypotheses: list[CausalHypothesis] = []
        self._verified_pairs: set[tuple[str, str]] = set()
        self._rounds_since_verification: int = 0

    @staticmethod
    def _canonical_pair(source_id: str, target_id: str) -> tuple[str, str]:
        return tuple(sorted((source_id, target_id)))

    @property
    def hypotheses(self) -> list[CausalHypothesis]:
        return list(self._hypotheses)

    @property
    def pending_count(self) -> int:
        return sum(1 for h in self._hypotheses if h.status == "pending")

    def generate_hypotheses(
        self,
        new_nodes: list[Node],
        graph: IntentGraph,
        current_round: int,
        embed_fn: EmbedFunction | None = None,
    ) -> list[CausalHypothesis]:
        """After each extraction round, identify candidate causal pairs."""
        new_hypotheses: list[CausalHypothesis] = []
        existing_nodes = graph.get_all_nodes()

        # Round-level embedding cache: batch-embed all relevant texts once,
        # shared across all _compute_ambiguity() calls in this round.
        cache: EmbeddingCache | None = None
        if embed_fn:
            cache = EmbeddingCache(embed_fn)
            all_texts = list({n.content for n in existing_nodes + new_nodes})
            if all_texts:
                cache.get_many(all_texts)

        explicit_pairs: set[tuple[str, str]] = set()
        for edge in graph.get_all_edges():
            if edge.source_type == EdgeSource.USER_EXPLICIT:
                explicit_pairs.add((edge.source, edge.target))

        hyp_pairs = {
            self._canonical_pair(h.source_node_id, h.target_node_id)
            for h in self._hypotheses
        }

        def _check_pair(src: Node, tgt: Node) -> None:
            if not is_worth_verifying(src, tgt):
                return
            if (src.id, tgt.id) in explicit_pairs:
                return
            pair_key = self._canonical_pair(src.id, tgt.id)
            if pair_key in hyp_pairs:
                return
            if pair_key in self._verified_pairs:
                return

            ambiguity = self._compute_ambiguity(src, tgt, graph, cache)
            if ambiguity < 0.2:
                return

            hypothesis = CausalHypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                source_node_id=src.id,
                target_node_id=tgt.id,
                source_content=src.content[:120],
                target_content=tgt.content[:120],
                ambiguity_score=ambiguity,
                status="pending",
                created_at_round=current_round,
            )
            new_hypotheses.append(hypothesis)
            hyp_pairs.add(pair_key)

        for new_node in new_nodes:
            for existing in existing_nodes:
                if new_node.id == existing.id:
                    continue
                _check_pair(new_node, existing)
                _check_pair(existing, new_node)

        for i, n1 in enumerate(new_nodes):
            for n2 in new_nodes[i + 1:]:
                _check_pair(n1, n2)
                _check_pair(n2, n1)

        self._hypotheses.extend(new_hypotheses)

        # Cap hypothesis list to prevent unbounded growth in long sessions
        _MAX_HYPOTHESES = 200
        if len(self._hypotheses) > _MAX_HYPOTHESES:
            # Prioritize: confirmed > denied > pending (newest first) > expired
            self._hypotheses.sort(
                key=lambda h: (
                    h.status != "confirmed",
                    h.status != "denied",
                    h.status != "pending",
                    -h.created_at_round,
                )
            )
            self._hypotheses = self._hypotheses[:_MAX_HYPOTHESES]

        self._rounds_since_verification += 1
        return new_hypotheses

    def _compute_ambiguity(
        self,
        source: Node,
        target: Node,
        graph: IntentGraph,
        embed_cache: EmbeddingCache | None = None,
    ) -> float:
        """
        Score how likely a node pair involves correlation masquerading as causation.
        Higher = more suspicious. Range [0, 1].

        Accepts an optional round-level EmbeddingCache to avoid per-pair
        cache creation.
        """
        score = 0.0

        if source.dimension == Dimension.WHY or target.dimension == Dimension.WHY:
            score += 0.3

        confusion_pairs = {
            (Dimension.WHY, Dimension.WHAT),
            (Dimension.WHY, Dimension.HOW),
            (Dimension.HOW, Dimension.WHAT),
        }
        if source.dimension and target.dimension:
            pair = (source.dimension, target.dimension)
            if pair in confusion_pairs or (pair[1], pair[0]) in confusion_pairs:
                score += 0.2

        raw_combined = (source.raw_quote + " " + target.raw_quote).lower()
        if not any(m in raw_combined for m in ALL_CAUSAL_MARKERS):
            score += 0.3

        if source.confidence > 0.8 and target.confidence > 0.8:
            score += 0.1

        if embed_cache:
            try:
                sim = embed_cache.similarity(source.content, target.content)
                if sim > 0.8:
                    score += 0.15
                elif sim > 0.6:
                    score += 0.05
            except Exception:
                pass

        return min(1.0, score)

    def should_verify_now(
        self,
        graph: IntentGraph | None = None,
        embed_fn: EmbedFunction | None = None,
        task_type: str = "",
    ) -> bool:
        """Determine if we should switch to causal verification mode."""
        if graph and task_type and embed_fn:
            high_risk = self.get_high_risk_pairs(graph, embed_fn, task_type)
            if high_risk:
                return True

        pending = [h for h in self._hypotheses if h.status == "pending"]
        if not pending:
            return False
        if self._rounds_since_verification < 2:
            return False
        if any(h.ambiguity_score > 0.7 for h in pending):
            return True
        if len(pending) >= 3:
            return True
        return False

    def get_next_hypothesis(self) -> CausalHypothesis | None:
        """Get the highest-priority pending hypothesis for verification."""
        pending = [
            h for h in self._hypotheses
            if h.status == "pending"
            and self._canonical_pair(h.source_node_id, h.target_node_id) not in self._verified_pairs
        ]
        if not pending:
            return None
        pending.sort(key=lambda h: h.ambiguity_score, reverse=True)
        self._rounds_since_verification = 0
        return pending[0]

    def resolve_hypothesis(self, hypothesis_id: str, status: str) -> None:
        """Mark a hypothesis as confirmed, denied, or expired."""
        for h in self._hypotheses:
            if h.id == hypothesis_id:
                h.status = status
                pair_key = self._canonical_pair(h.source_node_id, h.target_node_id)
                self._verified_pairs.add(pair_key)
                for other in self._hypotheses:
                    if (
                        other.id != h.id
                        and other.status == "pending"
                        and self._canonical_pair(other.source_node_id, other.target_node_id) == pair_key
                    ):
                        other.status = "expired"
                break

    def expire_stale(self, current_round: int, max_age: int = 8) -> None:
        """Expire hypotheses that have been pending too long."""
        for h in self._hypotheses:
            if h.status == "pending" and (current_round - h.created_at_round) > max_age:
                h.status = "expired"

    # ------------------------------------------------------------------
    # Correlation-Substitution Risk
    # ------------------------------------------------------------------

    def get_high_risk_pairs(
        self,
        graph: IntentGraph,
        embed_fn: EmbedFunction | None = None,
        task_type: str = "general",
    ) -> list[tuple[Node, Node, float]]:
        """
        Scan all node pairs for correlation-substitution risk.
        Uses EmbeddingCache to batch-embed all texts once (not per-pair).
        """
        if not embed_fn:
            return []

        nodes = graph.get_all_nodes()
        if len(nodes) < 2:
            return []

        cache = EmbeddingCache(embed_fn)
        all_texts = [n.content for n in nodes]
        all_vecs = cache.get_many(all_texts)
        if len(all_vecs) != len(nodes):
            return []

        vec_map = dict(zip(all_texts, all_vecs))
        high_risk: list[tuple[Node, Node, float]] = []

        verified: set[tuple[str, str]] = set()
        for hyp in self._hypotheses:
            if hyp.status in ("confirmed", "denied"):
                verified.add((hyp.source_node_id, hyp.target_node_id))
                verified.add((hyp.target_node_id, hyp.source_node_id))

        for i, node_a in enumerate(nodes):
            vec_a = vec_map.get(node_a.content)
            if vec_a is None:
                continue
            for node_b in nodes[i + 1:]:
                if not is_worth_verifying(node_a, node_b):
                    continue
                if (node_a.id, node_b.id) in verified:
                    continue
                vec_b = vec_map.get(node_b.content)
                if vec_b is None:
                    continue

                sim = cosine_similarity(vec_a, vec_b)
                if sim >= CORRELATION_RISK_THRESHOLD:
                    risk = (sim - CORRELATION_RISK_THRESHOLD) / (1.0 - CORRELATION_RISK_THRESHOLD)
                    if node_a.bias_flags or node_b.bias_flags:
                        risk = min(1.0, risk * 1.5)
                    high_risk.append((node_a, node_b, risk))

        return sorted(high_risk, key=lambda x: x[2], reverse=True)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "hypotheses": [h.model_dump() for h in self._hypotheses],
            "verified_pairs": [list(p) for p in self._verified_pairs],
            "rounds_since_verification": self._rounds_since_verification,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CausalTracker":
        tracker = cls()
        tracker._hypotheses = [CausalHypothesis(**h) for h in data.get("hypotheses", [])]
        tracker._verified_pairs = {
            cls._canonical_pair(*p) if len(p) == 2 else tuple(p)
            for p in data.get("verified_pairs", [])
        }
        tracker._rounds_since_verification = data.get("rounds_since_verification", 0)
        return tracker


# ---------------------------------------------------------------------------
# d) Extractive LLM-Judge (A+C Stage 2)
# ---------------------------------------------------------------------------

CAUSAL_JUDGE_PROMPT = """\
You are a strict causal auditor. Your ONLY job is to determine \
if the user explicitly stated a causal relationship between two concepts.

User's exact transcript (this is the ONLY source of truth):
\"\"\"{user_history}\"\"\"

Node A: "{node_a_text}"
Node B: "{node_b_text}"

TASK: Did the user EXPLICITLY state or directly imply that changing Node A \
would change Node B (or vice versa)?

CRITICAL RULES:
- You must NOT use your own knowledge or reasoning. ONLY the user's words matter.
- If the user did not explicitly link these concepts causally, answer false.

Output ONLY this JSON:
{{"is_causal": true/false, "direction": "A_causes_B"|"B_causes_A"|"bidirectional"|"none", "exact_proof_quote": "An EXACT substring from the transcript that proves the causal link. If none exists, this MUST be null."}}
"""


CAUSAL_FEEDBACK_PROMPT = """\
You are a strict interpreter of the user's response to a causal verification question.

The system asked the user whether:
  "{source_content}" CAUSES "{target_content}"

The question was: "{question}"
The user responded: "{user_text}"

Interpret the user's response as ONE of:
- "confirmed": User agrees the causal relationship exists
- "denied": User says they are NOT causally related
- "partial": User partially agrees or adds nuance
- "redirected": User mentions a DIFFERENT cause (not the one asked about)
- "ambiguous": Can't determine from the response

Output ONLY this JSON:
{{"verdict": "confirmed"|"denied"|"partial"|"redirected"|"ambiguous", "reasoning": "brief explanation", "alternative_cause": "if redirected, what did the user say instead (or null)"}}
"""


def deep_causal_verification(
    node_a_text: str,
    node_b_text: str,
    user_history: str,
    llm_fn: LLMFunction,
) -> dict:
    """
    Stage 2: Extractive LLM-Judge with anti-hallucination.
    The LLM must cite an exact quote from user text.
    If the quote doesn't exist in the transcript, we reject.
    """
    from ftg.fathom import LLMRequest

    prompt = CAUSAL_JUDGE_PROMPT.format(
        user_history=user_history,
        node_a_text=node_a_text,
        node_b_text=node_b_text,
    )
    try:
        from ftg.utils import parse_llm_json
        raw = llm_fn(LLMRequest(
            system_prompt=prompt,
            user_prompt=f'Node A: "{node_a_text}"\nNode B: "{node_b_text}"',
            json_mode=True,
            temperature=0.0,
        ))
        result = parse_llm_json(raw)
    except Exception as exc:
        logger.warning("Causal LLM-Judge failed: %s", exc)
        return {"is_causal": False, "reason": f"LLM call failed: {exc}"}

    # Anti-hallucination: verify proof quote exists in user transcript
    if result.get("is_causal") and result.get("exact_proof_quote"):
        quote = result["exact_proof_quote"]
        if quote and quote not in user_history:
            return {
                "is_causal": False,
                "reason": "Proof quote not found in transcript (hallucination rejected)",
            }

    return result


def process_causal_feedback(
    user_text: str,
    hypothesis: CausalHypothesis,
    question: str,
    llm_fn: LLMFunction,
) -> dict:
    """Process user's response to a counterfactual causal verification question."""
    from ftg.fathom import LLMRequest

    prompt = CAUSAL_FEEDBACK_PROMPT.format(
        source_content=hypothesis.source_content,
        target_content=hypothesis.target_content,
        question=question,
        user_text=user_text,
    )
    try:
        from ftg.utils import parse_llm_json
        raw = llm_fn(LLMRequest(
            system_prompt=prompt,
            user_prompt=f"User response: {user_text}",
            json_mode=True,
            temperature=0.2,
        ))
        result = parse_llm_json(raw)
    except Exception:
        result = {"verdict": "ambiguous"}

    return result
