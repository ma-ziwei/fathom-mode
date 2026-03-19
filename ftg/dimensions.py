"""
5W1H Dimensional Coordinate System — definitions, rules, and target selection.
"""

from __future__ import annotations


from ftg.graph import IntentGraph
from ftg.models import Dimension, Edge, EdgeSource, Node, RelationType


# ---------------------------------------------------------------------------
# Dimension descriptions (for question generation context)
# ---------------------------------------------------------------------------

DIMENSION_LABELS: dict[str, str] = {
    "who": "Who",
    "what": "What",
    "why": "Why",
    "when": "When",
    "where": "Where",
    "how": "How",
}

DIMENSION_DESCRIPTIONS: dict[str, str] = {
    "who": "who is involved or affected",
    "what": "what the core problem or subject is",
    "why": "why this matters — the motivation or goal",
    "when": "whether there's a timeline or deadline",
    "where": "the location, context, or setting",
    "how": "how to approach this — method, constraints, or criteria",
}


# ---------------------------------------------------------------------------
# Cross-Dimension Rules
# ---------------------------------------------------------------------------
# CAUSAL edges ONLY come from user's explicit language or confirmed hypotheses.
# Algorithm-inferred edges are structural, never causal.

CROSS_DIM_RULES: dict[tuple[Dimension, Dimension], RelationType] = {
    (Dimension.WHY, Dimension.WHAT): RelationType.SUPPORTS,
    (Dimension.WHY, Dimension.HOW): RelationType.SUPPORTS,
    (Dimension.WHY, Dimension.WHERE): RelationType.SUPPORTS,
    (Dimension.HOW, Dimension.WHAT): RelationType.DEPENDENCY,
    (Dimension.WHO, Dimension.WHAT): RelationType.DEPENDENCY,
    (Dimension.WHEN, Dimension.WHAT): RelationType.CONDITIONAL,
    (Dimension.WHERE, Dimension.HOW): RelationType.DEPENDENCY,
    (Dimension.WHAT, Dimension.WHERE): RelationType.DEPENDENCY,
    (Dimension.WHAT, Dimension.WHEN): RelationType.DEPENDENCY,
    (Dimension.WHAT, Dimension.HOW): RelationType.DEPENDENCY,
    (Dimension.WHO, Dimension.HOW): RelationType.DEPENDENCY,
    (Dimension.WHO, Dimension.WHERE): RelationType.DEPENDENCY,
}


# ---------------------------------------------------------------------------
# Universal dimension priority weights: HOW > WHY > WHO > WHAT > WHEN > WHERE
# ---------------------------------------------------------------------------
DEFAULT_DIM_PRIORITY: dict[str, float] = {
    "how": 6.0, "why": 5.0, "who": 4.0,
    "what": 3.0, "when": 2.0, "where": 1.0,
}


def find_target_dimension(
    graph: IntentGraph,
    complexity: float = 0.5,
    waived_dimensions: set | None = None,
    task_type: str = "general",
) -> str:
    """
    Determine which dimension to ask about next.

    score = priority_weight / (node_count + 1)
    The dimension with the highest score is selected next.
    """
    waived = waived_dimensions or set()
    counts = graph.dimension_node_counts(include_secondary=True)
    all_dims = [d.value for d in Dimension]

    candidates = [d for d in all_dims if d not in waived]
    if not candidates:
        candidates = all_dims

    priorities = DEFAULT_DIM_PRIORITY

    best_score = -1.0
    target = "how"
    for dim in candidates:
        count = counts.get(dim, 0)
        weight = priorities.get(dim, 3.0)
        score = weight / (count + 1)
        if score > best_score:
            best_score = score
            target = dim

    return target


def infer_edges(new_nodes: list[Node], graph: IntentGraph) -> list[Edge]:
    """Infer edges based on dimension relationships."""
    edges: list[Edge] = []
    existing_nodes = graph.get_all_nodes()

    if not existing_nodes and len(new_nodes) < 2:
        return edges

    existing_pairs = {(e.source, e.target) for e in graph.get_all_edges()}

    def _add(src: str, tgt: str, rel: RelationType) -> None:
        if (src, tgt) not in existing_pairs:
            edges.append(Edge(
                source=src, target=tgt,
                relation_type=rel,
                source_type=EdgeSource.ALGORITHM_INFERRED,
            ))
            existing_pairs.add((src, tgt))

    for new_node in new_nodes:
        if not new_node.dimension:
            continue

        # Same-dimension: SUPPORTS
        same_dim = [
            n for n in existing_nodes
            if n.dimension == new_node.dimension and n.id != new_node.id
        ]
        if same_dim:
            _add(new_node.id, same_dim[-1].id, RelationType.SUPPORTS)

        # Cross-dimension rules
        for existing in existing_nodes:
            if existing.id == new_node.id or not existing.dimension:
                continue
            key = (new_node.dimension, existing.dimension)
            if key in CROSS_DIM_RULES:
                _add(new_node.id, existing.id, CROSS_DIM_RULES[key])
            rev_key = (existing.dimension, new_node.dimension)
            if rev_key in CROSS_DIM_RULES:
                _add(existing.id, new_node.id, CROSS_DIM_RULES[rev_key])

    # Among new nodes themselves (check both directions, like new-vs-existing)
    for i, n1 in enumerate(new_nodes):
        if not n1.dimension:
            continue
        for n2 in new_nodes[i + 1:]:
            if not n2.dimension:
                continue
            if n1.dimension == n2.dimension:
                _add(n1.id, n2.id, RelationType.SUPPORTS)
            else:
                key = (n1.dimension, n2.dimension)
                if key in CROSS_DIM_RULES:
                    _add(n1.id, n2.id, CROSS_DIM_RULES[key])
                rev_key = (n2.dimension, n1.dimension)
                if rev_key in CROSS_DIM_RULES:
                    _add(n2.id, n1.id, CROSS_DIM_RULES[rev_key])

    return edges
