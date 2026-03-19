"""
Prompt Compiler — transforms a fathomed Intent Graph into a structured prompt.
"""

from __future__ import annotations

from typing import Iterable

from ftg.dimensions import DIMENSION_LABELS
from ftg.graph import IntentGraph
from ftg.models import Dimension, EdgeSource, Node, NodeType, RelationType

MAX_EXPLICIT_CAUSAL_ITEMS = 6
MAX_DEPENDENCY_CHAIN_ITEMS = 6
MAX_CONSTRAINT_ITEMS = 8
MAX_ANCHORED_ITEMS_PER_QUOTE = 6
MAX_SYSTEM_INFERRED_ITEMS = 6

BIAS_CORRECTION_MAP = {
    "sunk_cost": "When evaluating future options, do not treat already-invested time, money, or effort as a reason to continue.",
    "loss_aversion": "Weigh both gains and losses equally — do not fixate solely on what might be lost.",
    "status_quo_bias": "Do not assume maintaining the status quo is safer — compare all options on equal footing.",
    "anchoring": "Do not over-rely on the first number, anecdote, or impression that appeared.",
    "confirmation_bias": "Actively consider information that challenges the user's current inclination.",
    "availability_heuristic": "Base probability judgments on facts, not on recent examples or vivid impressions.",
    "framing_effect": "Try restating the same issue from both positive and negative angles to avoid being skewed by framing.",
    "endowment_effect": "Do not overvalue something simply because it is already owned or nearly owned.",
}

BIAS_LABELS = {
    "sunk_cost": "Sunk Cost",
    "loss_aversion": "Loss Aversion",
    "status_quo_bias": "Status Quo Bias",
    "anchoring": "Anchoring",
    "confirmation_bias": "Confirmation Bias",
    "availability_heuristic": "Availability Heuristic",
    "framing_effect": "Framing Effect",
    "endowment_effect": "Endowment Effect",
}


# ---------------------------------------------------------------------------
# Task-aware compilation directives — injected at the top of compiled prompt
# to guide downstream LLM output format and focus.
# ---------------------------------------------------------------------------

TASK_COMPILATION_DIRECTIVES: dict[str, str] = {
    "thinking": "This is a thinking request. Present a structured analysis that addresses the user's stated concerns. Highlight trade-offs and hidden assumptions. Make a recommendation if the evidence supports one, but acknowledge uncertainty.",
    "creation": "This is a creation request. Produce the requested deliverable directly, respecting all stated constraints on purpose, audience, tone, and format. If key information is missing, note it but still produce a complete draft.",
    "execution": "This is an execution request. List the exact steps to perform. Flag any steps that could have significant or irreversible consequences. State what you will NOT do as clearly as what you will do. If prerequisites or permissions are unclear, ask before proceeding.",
    "learning": "This is a learning request. Provide a structured path tailored to the user's level and constraints. Be specific about resources and milestones. Match the depth and complexity to what the user can absorb.",
}

COMPILER_INSTRUCTION_HEADER = (
    "The following is a structured intent analysis of the user's request, "
    "verified through multi-round dialogue. Use it to guide your response:\n"
    "\n"
    "- [1] Task Anchor: The user's original request — your response must address this.\n"
    "- [2] User Expression: Direct quotes and extracted understanding, grouped by the user's own words. "
    "Treat raw_quote lines as ground truth.\n"
    "- [3] User-Explicit Causal: Cause-effect relationships the user personally confirmed. "
    "Do NOT substitute correlation for these stated causes.\n"
    "- [4] Constraints & Conflicts: Hard constraints and internal tensions the user expressed. "
    "Address conflicts explicitly rather than ignoring them.\n"
    "- [5] System-Inferred: Supplementary understanding not directly quoted. "
    "Use cautiously; prefer [2] when they conflict.\n"
    "- Bias signals (e.g., [Sunk Cost], [Anchoring]) indicate cognitive biases detected in the user's reasoning. "
    "Gently correct for these in your response without being condescending.\n"
    "- Low confidence notes mean the system is uncertain about that interpretation. "
    "Acknowledge uncertainty rather than asserting confidently."
)


def compile_intent_graph(
    graph: IntentGraph,
    original_request: str,
    task_type: str = "general",
    user_context: str = "",
    dimension_bound_responses: dict[str, dict] | None = None,
    dimension_states: dict[str, dict] | None = None,
    attachment_contexts: list[dict] | None = None,
    dimension_assessment: dict[str, str] | None = None,
    fathom_score: float | None = None,
    causal_hypotheses: list | None = None,
    waived_dimensions: set | None = None,
) -> str:
    """Compile the Intent Graph into a structured prompt for downstream LLM."""
    sections: list[str] = [COMPILER_INSTRUCTION_HEADER]

    directive = TASK_COMPILATION_DIRECTIVES.get(task_type)
    if directive:
        sections.append(f"\nTask context: {directive}")

    if user_context:
        sections.append("=== [0] External Context ===")
        sections.append(user_context)

    if attachment_contexts:
        attachment_lines = _render_attachment_contexts(attachment_contexts)
        if attachment_lines:
            sections.append("\n=== [0a] User-Provided Attachment Context ===" if sections else "=== [0a] User-Provided Attachment Context ===")
            sections.extend(attachment_lines)

    sections.append("\n=== [1] Task Anchor ===" if sections else "=== [1] Task Anchor ===")
    sections.append(f"Original user request: {original_request}")

    sections.append("\n=== [2] User Expression & System Understanding ===")
    sections.extend(
        _render_user_expression_section(
            graph,
            dimension_bound_responses=dimension_bound_responses or {},
            dimension_states=dimension_states or {},
        )
    )

    explicit_causal_lines = _render_explicit_causal(graph)
    if explicit_causal_lines:
        sections.append("\n=== [3] User-Explicit Causal Relationships ===")
        sections.extend(explicit_causal_lines)

    constraint_lines = _render_constraints_section(graph)
    if constraint_lines:
        sections.append("\n=== [4] Constraints & Conflicts ===")
        sections.extend(constraint_lines)

    inferred_lines = _render_system_inferences(graph)
    if inferred_lines:
        sections.append("\n=== [5] System-Inferred Supplements ===")
        sections.extend(inferred_lines)

    return "\n".join(sections)


def _render_attachment_contexts(attachment_contexts: list[dict]) -> list[str]:
    lines: list[str] = []
    for item in attachment_contexts:
        label = str(item.get("label") or "").strip()
        summary = str(item.get("summary") or "").strip()
        raw_ref = str(item.get("raw_ref") or "").strip()
        parts: list[str] = []
        if label:
            parts.append(label)
        if summary:
            parts.append(summary)
        if raw_ref:
            parts.append(f"ref={raw_ref}")
        if parts:
            lines.append("- " + " | ".join(parts))
    return lines


def _render_user_expression_section(
    graph: IntentGraph,
    *,
    dimension_bound_responses: dict[str, dict],
    dimension_states: dict[str, dict],
) -> list[str]:
    groups = _group_nodes_by_raw_quote(graph, dimension_bound_responses)
    if not groups:
        return ["No graph information that can be reliably anchored to the user's own words."]

    lines: list[str] = []
    anchor_idx = 0
    for group in groups:
        quote = group["quote"]
        nodes = _dedupe_nodes_by_content(group["nodes"])
        bound_dims = _dedupe_preserve_order(group["bound_dimensions"])

        if bound_dims:
            continue

        anchor_idx += 1
        lines.append(f"--- Anchor {anchor_idx} ---")
        lines.append(f"User: {quote}")

        if nodes:
            visible = nodes[:MAX_ANCHORED_ITEMS_PER_QUOTE]
            for node in visible:
                lines.extend(_render_anchored_node(graph, node))
            hidden = len(nodes) - len(visible)
            if hidden > 0:
                lines.append(f"- ({hidden} additional items from this anchor omitted)")
        else:
            lines.append("- System note: this is a valid dimension follow-up response; refer to the user's original words when answering.")

        lines.append("")

    if lines and not lines[-1].strip():
        lines.pop()
    return lines


def _render_inline_dimension_states(
    dimension_states: dict[str, dict],
    *,
    node_ids: list[str],
    bound_dimensions: list[str],
) -> list[str]:
    if not dimension_states:
        return []

    related_node_ids = {node_id for node_id in node_ids if node_id}
    related_dimensions = {dim for dim in bound_dimensions if dim}
    rendered: list[str] = []

    for dim in [d.value for d in Dimension]:
        payload = dimension_states.get(dim, {})
        if not isinstance(payload, dict):
            continue

        supporting_node_ids = payload.get("supporting_node_ids", [])
        if not isinstance(supporting_node_ids, list):
            supporting_node_ids = []
        supporting_node_ids = {
            node_id for node_id in supporting_node_ids if isinstance(node_id, str) and node_id.strip()
        }
        if dim not in related_dimensions and not (supporting_node_ids & related_node_ids):
            continue

        coverage_level = str(payload.get("coverage_level") or "none").strip() or "none"
        evidence_present = bool(payload.get("evidence_present", False))
        open_gap = str(payload.get("open_gap") or "").strip()

        state_text = f"{DIMENSION_LABELS.get(dim, dim)}={coverage_level}"
        if evidence_present:
            state_text += " (evidence present"
            if open_gap:
                state_text += f", still missing: {open_gap}"
            state_text += ")"
        elif open_gap:
            state_text += f" (still missing: {open_gap})"
        rendered.append(state_text)

    return rendered


def _group_nodes_by_raw_quote(
    graph: IntentGraph,
    dimension_bound_responses: dict[str, dict],
) -> list[dict]:
    groups: list[dict] = []
    index_by_quote: dict[str, int] = {}

    def ensure_group(quote: str) -> dict:
        if quote not in index_by_quote:
            index_by_quote[quote] = len(groups)
            groups.append({"quote": quote, "nodes": [], "bound_dimensions": []})
        return groups[index_by_quote[quote]]

    for node in graph.get_all_nodes():
        quote = (node.raw_quote or "").strip()
        if not quote:
            continue
        ensure_group(quote)["nodes"].append(node)

    for dimension, payload in (dimension_bound_responses or {}).items():
        quote = str(payload.get("raw_text") or "").strip()
        if not quote:
            continue
        ensure_group(quote)["bound_dimensions"].append(dimension)

    return groups


def _render_anchored_node(graph: IntentGraph, node: Node) -> list[str]:
    lines = [f"- Understanding: {node.content}"]

    causal_roles = _render_inline_causal_roles(graph, node)
    if causal_roles:
        lines.append(f"  Causal role: {'; '.join(causal_roles)}")

    bias_notes = _render_inline_bias_notes(node)
    if bias_notes:
        lines.append(f"  Bias signal: {'; '.join(bias_notes)}")

    if node.confidence < 0.5:
        lines.append(f"  Low confidence: node confidence is {node.confidence:.0%} — retain uncertainty when using.")

    return lines


def _format_dimension_line(node: Node) -> str:
    primary = node.dimension.value if node.dimension else ""
    secondary = [d for d in node.secondary_dimensions if d and d != primary]

    parts: list[str] = []
    if primary:
        parts.append(DIMENSION_LABELS.get(primary, primary))
    if secondary:
        parts.append("secondary: " + _format_dimension_names(secondary))
    return "; ".join(parts) if parts else "not yet classified"


def _format_dimension_names(dimensions: Iterable[str]) -> str:
    names = [DIMENSION_LABELS.get(dim, dim) for dim in dimensions if dim]
    return ", ".join(_dedupe_preserve_order(names))


def _render_inline_causal_roles(graph: IntentGraph, node: Node) -> list[str]:
    roles: list[str] = []
    for edge in graph.get_all_edges():
        if edge.relation_type != RelationType.CAUSAL:
            continue
        if edge.source == node.id:
            if edge.source_type == EdgeSource.USER_EXPLICIT:
                roles.append("user-stated cause")
            else:
                roles.append("system-inferred cause")
        if edge.target == node.id:
            if edge.source_type == EdgeSource.USER_EXPLICIT:
                roles.append("user-stated effect")
            else:
                roles.append("system-inferred effect")
    return _dedupe_preserve_order(roles)


def _render_inline_bias_notes(node: Node) -> list[str]:
    notes: list[str] = []
    for bias in node.bias_flags:
        label = BIAS_LABELS.get(bias, bias)
        correction = BIAS_CORRECTION_MAP.get(bias, f"Be mindful of {label} during analysis.")
        notes.append(f"[{label}] {correction}")
    return _dedupe_preserve_order(notes)


def _render_explicit_causal(graph: IntentGraph) -> list[str]:
    """Render only user-explicit causal relationships. Returns [] if none."""
    causal_edges = [
        e for e in graph.get_all_edges()
        if e.relation_type == RelationType.CAUSAL
        and e.source_type == EdgeSource.USER_EXPLICIT
    ]
    if not causal_edges:
        return []

    lines: list[str] = []
    for edge in causal_edges[:MAX_EXPLICIT_CAUSAL_ITEMS]:
        src = graph.get_node(edge.source)
        tgt = graph.get_node(edge.target)
        if src and tgt:
            lines.append(f"- {src.content} -> causes -> {tgt.content}")
    hidden = max(0, len(causal_edges) - MAX_EXPLICIT_CAUSAL_ITEMS)
    if hidden:
        lines.append(f"- ({hidden} additional user-explicit causal relationships omitted)")
    return lines


def _render_constraints_section(graph: IntentGraph) -> list[str]:
    constraints = _extract_constraints(graph)[:MAX_CONSTRAINT_ITEMS]
    if not constraints:
        return []
    return [f"- {item}" for item in constraints]


def _extract_constraints(graph: IntentGraph) -> list[str]:
    constraints: list[str] = []
    for node in graph.get_nodes_by_type(NodeType.CONSTRAINT):
        if node.content:
            constraints.append(node.content)

    for src, tgt in graph.has_contradictions():
        src_node = graph.get_node(src)
        tgt_node = graph.get_node(tgt)
        if not src_node or not tgt_node:
            continue
        if src_node.dimension != tgt_node.dimension:
            continue
        constraints.append(
            "Note: the following two items show tension or conflict: "
            f'"{_node_anchor_or_content(src_node)}" vs "{_node_anchor_or_content(tgt_node)}".'
        )

    return _dedupe_preserve_order([item for item in constraints if item])


def _node_anchor_or_content(node: Node) -> str:
    quote = (node.raw_quote or "").strip()
    return quote if quote else node.content[:80]


def _render_system_inferences(graph: IntentGraph) -> list[str]:
    lines: list[str] = []
    unanchored = [
        node for node in graph.get_all_nodes()
        if not (node.raw_quote or "").strip()
    ]
    nodes = _dedupe_nodes_by_content(unanchored)[:MAX_SYSTEM_INFERRED_ITEMS]
    if not nodes:
        return []

    for node in nodes:
        lines.append(f"- {node.content}")
        if node.bias_flags:
            lines.append(f"  Bias signal: {'; '.join(_render_inline_bias_notes(node))}")
        if node.confidence < 0.5:
            lines.append(f"  Low confidence: node confidence is {node.confidence:.0%}.")
    hidden = max(0, len(unanchored) - len(nodes))
    if hidden:
        lines.append(f"- ({hidden} additional system-inferred supplements omitted)")
    return lines


def _render_chain(graph: IntentGraph, chain: list[str]) -> str:
    labels: list[str] = []
    for nid in chain:
        node = graph.get_node(nid)
        labels.append(node.content[:60] if node else nid)
    return " -> ".join(labels)


def _render_unique_chains(
    graph: IntentGraph,
    chains: list[list[str]],
    *,
    limit: int,
) -> list[str]:
    rendered = _dedupe_preserve_order(
        [_render_chain(graph, chain) for chain in chains if chain]
    )
    return rendered[:limit]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _dedupe_nodes_by_content(nodes: list[Node]) -> list[Node]:
    seen: set[str] = set()
    result: list[Node] = []
    for node in nodes:
        key = node.content.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(node)
    return result


def _render_coverage_summary(
    dimension_assessment: dict[str, str],
    waived_dimensions: set,
    causal_hypotheses: list,
    fathom_score: float | None,
) -> list[str]:
    """Render a brief coverage summary for downstream LLM awareness."""
    if not dimension_assessment and fathom_score is None:
        return []

    lines: list[str] = []
    covered = sorted(d for d, s in dimension_assessment.items() if s in ("covered", "covered_implicitly"))
    gaps = sorted(d for d, s in dimension_assessment.items() if s == "missing")
    waived = sorted(str(d) for d in waived_dimensions) if waived_dimensions else []

    if covered:
        lines.append(f"Dimensions covered: {', '.join(covered)}")
    if gaps:
        lines.append(f"Dimensions with gaps: {', '.join(gaps)}")
    if waived:
        lines.append(f"Dimensions waived by user: {', '.join(waived)}")

    if causal_hypotheses:
        confirmed = sum(1 for h in causal_hypotheses if getattr(h, "status", "") == "confirmed")
        pending = sum(1 for h in causal_hypotheses if getattr(h, "status", "") == "pending")
        if confirmed or pending:
            lines.append(f"Causal verification: {confirmed} confirmed, {pending} pending")

    if fathom_score is not None:
        lines.append(f"Fathom confidence: {fathom_score:.2f}")

    return lines


