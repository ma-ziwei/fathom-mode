"""
Fathom depth scoring and diagnostics.

``compute_fathom_breakdown()`` is the FtG depth model. It measures how much
high-value, increasingly deep, increasingly grounded information has surfaced.

``evaluate_fathom_gates()`` computes score-related diagnostics. It intentionally
does not decide when a session should stop — manual control is handled at the
FtG protocol layer via explicit commands such as ``fathom`` and ``stop``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ftg.graph import IntentGraph
from ftg.models import CausalHypothesis, Dimension, EdgeSource, Node, RelationType

if TYPE_CHECKING:
    from ftg.causal import CausalTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEPTH_CAP = 3
SURFACE_DIMENSIONS = {"what", "when", "where"}
DEPTH_DIMENSIONS = {"why", "who", "how"}

SURFACE_WEIGHT = 1.0
DEPTH_WEIGHT = 1.9
PARTIAL_CAUSAL_WEIGHT = 2.8
VERIFIED_CAUSAL_WEIGHT = 3.4

ENTROPY_REGULARIZER_WEIGHT = 0.15
DISPLAY_SCALE = 0.10
DEPTH_PENETRATION_SCALE = 0.30
BEDROCK_SCALE = 1.0


# ---------------------------------------------------------------------------
# Fathom Score breakdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _UtilityAtom:
    atom_id: str
    dimension: str
    base_utility: float


@dataclass(frozen=True)
class FathomScoreBreakdown:
    fathom_score: float
    surface_coverage: float
    depth_penetration: float
    bedrock_grounding: float
    utility_mass: float
    grounding_mass: float
    entropy_regularizer: float
    atom_count: int = 0
    grounded_pair_count: int = 0
    relevant_dimensions: list[str] = field(default_factory=list)
    creditable_dimensions: list[str] = field(default_factory=list)
    blocked_secondary_count: int = 0
    blocked_secondary_dimensions: list[str] = field(default_factory=list)



def compute_fathom_breakdown(
    graph: IntentGraph,
    relevant_dimensions: set[str] | None = None,
    creditable_dimensions: set[str] | None = None,
    dimension_bound_responses: dict[str, dict] | None = None,
    causal_hypotheses: list[CausalHypothesis] | None = None,
) -> FathomScoreBreakdown:
    """
    Compute the current FtG depth score.

    Official score sources:
    - graph semantic nodes
    - dimension-bound responses
    - causal grounding

    Attachment/file context is intentionally excluded from score. It is context,
    not understanding depth.
    """
    relevant = _normalize_relevant_dimensions(graph, relevant_dimensions)
    creditable = _normalize_creditable_dimensions(relevant, creditable_dimensions)
    bound = dimension_bound_responses or {}
    atoms, blocked_secondary_dimensions = _graph_atoms(
        graph,
        relevant,
        creditable,
    )
    atoms.extend(_bound_response_atoms(graph, relevant, creditable, bound))

    utility_mass, credited_mass = _credit_utility_atoms(atoms)
    surface_coverage = _coverage_entropy(credited_mass, relevant)
    entropy_regularizer = ENTROPY_REGULARIZER_WEIGHT * surface_coverage
    depth_penetration = (
        1.0 - math.exp(-DEPTH_PENETRATION_SCALE * utility_mass)
        if utility_mass > 0.0
        else 0.0
    )

    grounding_by_pair = _collect_grounding_pairs(
        graph,
        causal_hypotheses or [],
        relevant,
    )
    grounding_mass = sum(grounding_by_pair.values())
    bedrock_grounding = (
        1.0 - math.exp(-BEDROCK_SCALE * grounding_mass)
        if grounding_mass > 0.0
        else 0.0
    )

    latent_depth = utility_mass + grounding_mass + entropy_regularizer
    fathom_score = (
        1.0 - math.exp(-DISPLAY_SCALE * latent_depth)
        if latent_depth > 0.0
        else 0.0
    )

    return FathomScoreBreakdown(
        fathom_score=fathom_score,
        surface_coverage=surface_coverage,
        depth_penetration=depth_penetration,
        bedrock_grounding=bedrock_grounding,
        utility_mass=utility_mass,
        grounding_mass=grounding_mass,
        entropy_regularizer=entropy_regularizer,
        atom_count=len(atoms),
        grounded_pair_count=len(grounding_by_pair),
        relevant_dimensions=sorted(relevant),
        creditable_dimensions=sorted(creditable),
        blocked_secondary_count=len(blocked_secondary_dimensions),
        blocked_secondary_dimensions=sorted(set(blocked_secondary_dimensions)),
    )


def _normalize_relevant_dimensions(
    graph: IntentGraph,
    relevant_dimensions: set[str] | None,
) -> set[str]:
    if relevant_dimensions is not None:
        normalized: set[str] = set()
        for dim in relevant_dimensions:
            dim_value = Dimension.normalize(dim)
            if dim_value and dim_value in _all_scoring_dimensions():
                normalized.add(dim_value)
        if normalized:
            return normalized
    active = {
        dim.value for dim in graph.get_active_dimensions(include_secondary=True)
        if dim.value in _all_scoring_dimensions()
    }
    return active or {"what"}


def _normalize_creditable_dimensions(
    relevant_dimensions: set[str],
    creditable_dimensions: set[str] | None,
) -> set[str]:
    if creditable_dimensions is None:
        return set(relevant_dimensions)

    normalized: set[str] = set()
    for dim in creditable_dimensions:
        dim_value = Dimension.normalize(dim)
        if dim_value and dim_value in _all_scoring_dimensions() and dim_value in relevant_dimensions:
            normalized.add(dim_value)
    return normalized


def _graph_atoms(
    graph: IntentGraph,
    relevant_dimensions: set[str],
    creditable_dimensions: set[str],
) -> tuple[list[_UtilityAtom], list[str]]:
    atoms: list[_UtilityAtom] = []
    blocked_secondary_dimensions: list[str] = []
    for node in graph.get_all_nodes():
        primary_dim = node.dimension.value if node.dimension else ""
        if not primary_dim or primary_dim not in relevant_dimensions:
            blocked_secondary_dimensions.extend(
                secondary
                for secondary in node.secondary_dimensions
                if secondary in relevant_dimensions
            )
            continue
        if primary_dim not in creditable_dimensions:
            blocked_secondary_dimensions.extend(
                secondary
                for secondary in node.secondary_dimensions
                if secondary in relevant_dimensions
            )
            continue

        allowed_secondary: list[str] = []
        primary_rank = _dimension_depth_rank(primary_dim)
        for secondary in node.secondary_dimensions:
            if secondary not in relevant_dimensions or secondary == primary_dim:
                continue
            if secondary not in creditable_dimensions:
                blocked_secondary_dimensions.append(secondary)
                continue
            if _dimension_depth_rank(secondary) > primary_rank:
                blocked_secondary_dimensions.append(secondary)
                continue
            allowed_secondary.append(secondary)

        primary_utility = _dimension_layer_weight(primary_dim)
        atoms.append(
            _UtilityAtom(
                atom_id=f"node:{node.id}:{primary_dim}",
                dimension=primary_dim,
                base_utility=primary_utility,
            )
        )

        for secondary in allowed_secondary:
            secondary_utility = _dimension_layer_weight(secondary)
            atoms.append(
                _UtilityAtom(
                    atom_id=f"node:{node.id}:{secondary}",
                    dimension=secondary,
                    base_utility=secondary_utility,
                )
            )
    return atoms, blocked_secondary_dimensions


def _bound_response_atoms(
    graph: IntentGraph,
    relevant_dimensions: set[str],
    creditable_dimensions: set[str],
    dimension_bound_responses: dict[str, dict],
) -> list[_UtilityAtom]:
    atoms: list[_UtilityAtom] = []
    for dim, response in dimension_bound_responses.items():
        if dim not in relevant_dimensions or dim not in creditable_dimensions:
            continue
        promoted_ids = [
            node_id
            for node_id in response.get("node_ids", []) or []
            if graph.get_node(node_id)
        ]
        if promoted_ids:
            continue
        raw_text = str(response.get("raw_text", "") or "").strip()
        if not raw_text:
            continue
        base_utility = _dimension_layer_weight(dim)
        atoms.append(
            _UtilityAtom(
                atom_id=f"bound:{dim}:{response.get('round', 0)}:{raw_text[:32]}",
                dimension=dim,
                base_utility=base_utility,
            )
        )
    return atoms


def _all_scoring_dimensions() -> set[str]:
    return SURFACE_DIMENSIONS | DEPTH_DIMENSIONS


def _dimension_depth_rank(dimension: str) -> int:
    return 1 if dimension in DEPTH_DIMENSIONS else 0


def _dimension_layer_weight(dimension: str) -> float:
    return DEPTH_WEIGHT if dimension in DEPTH_DIMENSIONS else SURFACE_WEIGHT



def _credit_utility_atoms(atoms: list[_UtilityAtom]) -> tuple[float, dict[str, float]]:
    utility_mass = 0.0
    credited_mass: dict[str, float] = defaultdict(float)
    atoms_by_dimension: dict[str, list[_UtilityAtom]] = defaultdict(list)
    for atom in atoms:
        atoms_by_dimension[atom.dimension].append(atom)

    for group in atoms_by_dimension.values():
        ordered = sorted(group, key=lambda atom: atom.base_utility, reverse=True)
        for index, atom in enumerate(ordered):
            novelty = 1.0 / math.sqrt(1.0 + index)
            contribution = atom.base_utility * novelty
            utility_mass += contribution
            credited_mass[atom.dimension] += contribution

    return utility_mass, dict(credited_mass)


def _coverage_entropy(
    credited_mass: dict[str, float],
    relevant_dimensions: set[str],
) -> float:
    if not relevant_dimensions:
        return 0.0
    total_mass = sum(credited_mass.get(dim, 0.0) for dim in relevant_dimensions)
    if total_mass <= 0.0:
        return 0.0
    if len(relevant_dimensions) == 1:
        return 1.0

    entropy = 0.0
    for dim in relevant_dimensions:
        mass = credited_mass.get(dim, 0.0)
        if mass <= 0.0:
            continue
        probability = mass / total_mass
        entropy -= probability * math.log(probability)

    denom = math.log(len(relevant_dimensions))
    return entropy / denom if denom > 0.0 else 1.0


def _collect_grounding_pairs(
    graph: IntentGraph,
    causal_hypotheses: list[CausalHypothesis],
    relevant_dimensions: set[str],
) -> dict[tuple[str, str], float]:
    pair_weights: dict[tuple[str, str], float] = {}

    for edge in graph.get_all_edges():
        if edge.relation_type != RelationType.CAUSAL:
            continue
        if edge.source_type != EdgeSource.USER_EXPLICIT:
            continue
        if not _pair_touches_relevant_dimensions(graph, edge.source, edge.target, relevant_dimensions):
            continue
        key = tuple(sorted((edge.source, edge.target)))
        pair_weights[key] = max(pair_weights.get(key, 0.0), PARTIAL_CAUSAL_WEIGHT)

    for hypothesis in causal_hypotheses:
        if hypothesis.status not in {"confirmed", "denied"}:
            continue
        if not _pair_touches_relevant_dimensions(
            graph,
            hypothesis.source_node_id,
            hypothesis.target_node_id,
            relevant_dimensions,
        ):
            continue
        key = tuple(sorted((hypothesis.source_node_id, hypothesis.target_node_id)))
        pair_weights[key] = max(pair_weights.get(key, 0.0), VERIFIED_CAUSAL_WEIGHT)

    return pair_weights


def _pair_touches_relevant_dimensions(
    graph: IntentGraph,
    source_id: str,
    target_id: str,
    relevant_dimensions: set[str],
) -> bool:
    source = graph.get_node(source_id)
    target = graph.get_node(target_id)
    if source is None or target is None:
        return False
    source_dims = _node_dimension_set(source, relevant_dimensions)
    target_dims = _node_dimension_set(target, relevant_dimensions)
    return bool(source_dims | target_dims)


def _node_dimension_set(node: Node, relevant_dimensions: set[str]) -> set[str]:
    dims: set[str] = set()
    if node.dimension and node.dimension.value in relevant_dimensions:
        dims.add(node.dimension.value)
    for secondary in node.secondary_dimensions:
        if secondary in relevant_dimensions:
            dims.add(secondary)
    return dims


def estimate_complexity(graph: IntentGraph) -> float:
    """
    Estimate task complexity from graph properties. Returns [0, 1].

    Signals: dimension breadth, WHY/HOW depth, contradictions,
    multiple connected components.
    """
    if graph.node_count() == 0:
        return 0.5

    score = 0.0
    counts = graph.dimension_node_counts()
    active_dims = sum(1 for c in counts.values() if c > 0)

    score += min(0.3, active_dims * 0.05)
    if counts.get("who", 0) > 0:
        score += 0.1
    if counts.get("why", 0) >= 2:
        score += 0.1
    if counts.get("how", 0) >= 2:
        score += 0.1
    score += min(0.2, graph.node_count() * 0.02)
    if graph.has_contradictions():
        score += 0.1

    return min(1.0, score)


def information_gain(score_before: float, score_after: float) -> float:
    """How much Fathom Score increased after new information was added."""
    return max(0.0, score_after - score_before)


# ---------------------------------------------------------------------------
# Fathom gate diagnostics
# ---------------------------------------------------------------------------


def evaluate_fathom_gates(
    graph: IntentGraph,
    fathom_history: list[float],
    round_count: int,
    waived_dimensions: set | None = None,
    causal_tracker: "CausalTracker | None" = None,
    dimension_assessment: dict | None = None,
    dimension_bound_responses: dict[str, dict] | None = None,
) -> dict:
    """
    Compute FtG depth diagnostics.

    This function intentionally does not auto-stop sessions. Manual control is
    handled at the FtG protocol layer via explicit commands such as ``fathom``
    and ``stop``.
    """

    waived = set(waived_dimensions or [])
    all_dims = {d.value for d in Dimension}

    conn_score = graph.connectivity_score(mode="weighted")
    complexity = estimate_complexity(graph)

    relevant_dims: set[str] = set()
    if dimension_assessment:
        for d in all_dims:
            if d in waived:
                continue
            relevant_dims.add(d)
    else:
        covered = {d.value for d in graph.get_active_dimensions(include_secondary=True)}
        relevant_dims = (covered | (all_dims - waived)) - waived
    if not relevant_dims:
        relevant_dims = {"what"}

    creditable_dims = _build_creditable_dimensions(
        graph=graph,
        relevant_dimensions=relevant_dims,
        dimension_assessment=dimension_assessment or {},
        dimension_bound_responses=dimension_bound_responses or {},
        causal_tracker=causal_tracker,
    )

    breakdown = compute_fathom_breakdown(
        graph,
        relevant_dimensions=relevant_dims,
        creditable_dimensions=creditable_dims,
        dimension_bound_responses=dimension_bound_responses,
        causal_hypotheses=causal_tracker.hypotheses if causal_tracker else None,
    )
    raw_score = breakdown.fathom_score

    historical_max = max(fathom_history) if fathom_history else 0.0
    current_score = max(raw_score, historical_max)

    effective_history = list(fathom_history) + [current_score]
    if len(effective_history) >= 2:
        latest_gain = information_gain(effective_history[-2], effective_history[-1])
    else:
        latest_gain = current_score

    covered_dims = {d.value for d in graph.get_active_dimensions(include_secondary=True)}
    uncovered = sorted(all_dims - covered_dims - waived)

    return {
        "is_fathomed": False,
        "fathom_type": "not_fathomed",
        "fathom_score": current_score,
        "surface_coverage": breakdown.surface_coverage,
        "depth_penetration": breakdown.depth_penetration,
        "bedrock_grounding": breakdown.bedrock_grounding,
        "utility_mass": breakdown.utility_mass,
        "grounding_mass": breakdown.grounding_mass,
        "entropy_regularizer": breakdown.entropy_regularizer,
        "score_atom_count": breakdown.atom_count,
        "grounded_pair_count": breakdown.grounded_pair_count,
        "score_relevant_dimensions": breakdown.relevant_dimensions,
        "creditable_dimensions": breakdown.creditable_dimensions,
        "blocked_secondary_count": breakdown.blocked_secondary_count,
        "blocked_secondary_dimensions": breakdown.blocked_secondary_dimensions,
        "reason": "manual fathom required",
        "uncovered_dimensions": uncovered,
        "waived_dimensions": sorted(waived),
        "connectivity_score": conn_score,
        "complexity": complexity,
        "information_gain": latest_gain,
        "round_count": round_count,
    }


def _build_creditable_dimensions(
    *,
    graph: IntentGraph,
    relevant_dimensions: set[str],
    dimension_assessment: dict,
    dimension_bound_responses: dict[str, dict],
    causal_tracker: "CausalTracker | None",
) -> set[str]:
    creditable: set[str] = set()

    for dim in relevant_dimensions:
        status = dimension_assessment.get(dim, "missing")
        if status in {"covered", "covered_implicitly"}:
            creditable.add(dim)

    for dim, response in dimension_bound_responses.items():
        if dim not in relevant_dimensions:
            continue
        raw_text = str(response.get("raw_text", "") or "").strip()
        if raw_text:
            creditable.add(dim)

    def _add_node_dimensions(node_id: str) -> None:
        node = graph.get_node(node_id)
        if not node:
            return
        if node.dimension and node.dimension.value in relevant_dimensions:
            creditable.add(node.dimension.value)
        for secondary in node.secondary_dimensions:
            if secondary in relevant_dimensions:
                creditable.add(secondary)

    for edge in graph.get_all_edges():
        if edge.relation_type.value != "causal":
            continue
        if edge.source_type.value != "user_explicit":
            continue
        _add_node_dimensions(edge.source)
        _add_node_dimensions(edge.target)

    if causal_tracker:
        for hypothesis in causal_tracker.hypotheses:
            if hypothesis.status not in {"confirmed", "denied"}:
                continue
            _add_node_dimensions(hypothesis.source_node_id)
            _add_node_dimensions(hypothesis.target_node_id)

    return creditable
