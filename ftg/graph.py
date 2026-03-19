"""
Intent Graph — directed graph of extracted information nodes and relationships.
"""

from __future__ import annotations

import json
import logging

import networkx as nx

from ftg.models import Dimension, Edge, EdgeSource, Node, NodeType, RelationType

EDGE_WEIGHT: dict[EdgeSource, float] = {
    EdgeSource.USER_EXPLICIT: 1.0,
    EdgeSource.USER_IMPLIED: 0.8,
    EdgeSource.ALGORITHM_INFERRED: 0.3,
}

logger = logging.getLogger(__name__)

DAG_RELATIONS = frozenset({RelationType.CAUSAL, RelationType.DEPENDENCY})

RELATION_PRIORITY: dict[RelationType, int] = {
    RelationType.CAUSAL: 5,
    RelationType.DEPENDENCY: 4,
    RelationType.CONTRADICTION: 3,
    RelationType.CONDITIONAL: 2,
    RelationType.SUPPORTS: 1,
}

SOURCE_PRIORITY: dict[EdgeSource, int] = {
    EdgeSource.USER_EXPLICIT: 3,
    EdgeSource.USER_IMPLIED: 2,
    EdgeSource.ALGORITHM_INFERRED: 1,
}


def _edge_strength(edge: Edge) -> tuple[int, int]:
    return (
        RELATION_PRIORITY.get(edge.relation_type, 0),
        SOURCE_PRIORITY.get(edge.source_type, 0),
    )


class IntentGraph:
    """
    A directed graph that stores extracted information nodes and their
    relationships. Grows dynamically as the user provides more information.
    """

    def __init__(self) -> None:
        self._G: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, Node] = {}
        self._edge_store: dict[tuple[str, str], Edge] = {}
        self._dag: nx.DiGraph = nx.DiGraph()  # causal + dependency edges only

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add or update an information node."""
        self._nodes[node.id] = node
        self._G.add_node(
            node.id,
            content=node.content,
            confidence=node.confidence,
            node_type=node.node_type.value,
            dimension=node.dimension.value if node.dimension else "what",
            bias_flags=node.bias_flags,
            secondary_dimensions=node.secondary_dimensions,
        )

    def add_edge(self, edge: Edge) -> bool:
        """
        Add a directed edge. Returns False if rejected.

        Strongest-edge-wins: each (source, target) pair holds at most one edge.
        A new edge replaces an existing one only if it has higher relation
        priority or (same relation but) stronger source provenance.

        CAUSAL INVARIANT: CAUSAL edges MUST have source_type == USER_EXPLICIT.
        Any violation is auto-downgraded to SUPPORTS.

        DAG CONSTRAINT: Only enforced on the CAUSAL + DEPENDENCY subgraph,
        not the entire graph. A SUPPORTS cycle will NOT block a legal
        DEPENDENCY edge.
        """
        if edge.source not in self._nodes or edge.target not in self._nodes:
            return False

        if edge.source == edge.target:
            return False

        if (edge.relation_type == RelationType.CAUSAL
                and edge.source_type != EdgeSource.USER_EXPLICIT):
            logger.warning(
                "CAUSAL invariant enforced: %s->%s downgraded to SUPPORTS "
                "(source_type=%s)",
                edge.source, edge.target, edge.source_type.value,
            )
            edge = Edge(
                source=edge.source,
                target=edge.target,
                relation_type=RelationType.SUPPORTS,
                source_type=edge.source_type,
                weight=edge.weight,
            )

        pair = (edge.source, edge.target)
        existing = self._edge_store.get(pair)

        if existing is not None:
            if _edge_strength(edge) <= _edge_strength(existing):
                return False

        if edge.relation_type in DAG_RELATIONS:
            if self._would_create_dag_cycle(edge.source, edge.target):
                return False

        # If replacing an existing DAG edge, remove old from _dag first
        if existing is not None and existing.relation_type in DAG_RELATIONS:
            self._dag.remove_edge(edge.source, edge.target)

        self._edge_store[pair] = edge
        self._G.add_edge(
            edge.source, edge.target,
            relation_type=edge.relation_type.value,
            source_type=edge.source_type.value,
        )
        if edge.relation_type in DAG_RELATIONS:
            self._dag.add_edge(edge.source, edge.target)
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edge_store)

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def get_all_edges(self) -> list[Edge]:
        return list(self._edge_store.values())

    def has_edge_between(self, source_id: str, target_id: str) -> bool:
        return ((source_id, target_id) in self._edge_store
                or (target_id, source_id) in self._edge_store)

    def get_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_nodes_by_dimension(
        self,
        dim: Dimension | str,
        include_secondary: bool = False,
    ) -> list[Node]:
        dim_val = Dimension.normalize(dim)
        result = []
        for n in self._nodes.values():
            if n.dimension and n.dimension.value == dim_val:
                result.append(n)
            elif include_secondary and dim_val in n.secondary_dimensions:
                result.append(n)
        return result

    def get_active_dimensions(self, include_secondary: bool = False) -> list[Dimension]:
        active: set[Dimension] = set()
        for n in self._nodes.values():
            if n.dimension:
                active.add(n.dimension)
            if include_secondary:
                for sd in n.secondary_dimensions:
                    try:
                        active.add(Dimension(sd))
                    except ValueError:
                        pass
        return sorted(active, key=lambda d: d.value)

    def dimension_coverage(self, include_secondary: bool = False) -> float:
        if not self._nodes:
            return 0.0
        return len(self.get_active_dimensions(include_secondary)) / len(Dimension)

    def dimension_node_counts(self, include_secondary: bool = False) -> dict[str, int]:
        counts: dict[str, int] = {d.value: 0 for d in Dimension}
        for n in self._nodes.values():
            if n.dimension:
                counts[n.dimension.value] += 1
            if include_secondary:
                for sd in n.secondary_dimensions:
                    if sd in counts:
                        counts[sd] += 1
        return counts

    def connectivity_score(self, mode: str = "weighted") -> float:
        """
        Ratio of connected evidence to total nodes.

        Modes:
          "weighted" — edges weighted by source provenance (default for Gate 3)
          "evidence" — only USER_EXPLICIT + USER_IMPLIED edges counted
          "all"      — all edges equal weight (legacy behaviour)
        """
        n = self.node_count()
        if n == 0:
            return 0.0

        if mode == "all":
            components = list(nx.connected_components(self._G.to_undirected()))
            largest = max(len(c) for c in components)
            return largest / n

        evidence_sources = {EdgeSource.USER_EXPLICIT, EdgeSource.USER_IMPLIED}
        g = nx.Graph()
        g.add_nodes_from(self._nodes.keys())

        for edge in self._edge_store.values():
            if mode == "evidence" and edge.source_type not in evidence_sources:
                continue
            w = EDGE_WEIGHT.get(edge.source_type, 0.3)
            if g.has_edge(edge.source, edge.target):
                cur = g[edge.source][edge.target]["weight"]
                g[edge.source][edge.target]["weight"] = max(cur, w)
            else:
                g.add_edge(edge.source, edge.target, weight=w)

        if g.number_of_edges() == 0:
            return 0.0

        components = list(nx.connected_components(g))
        largest_cc = max(components, key=len)
        total_weight = sum(
            g[u][v]["weight"]
            for u, v in g.edges()
            if u in largest_cc and v in largest_cc
        )
        max_possible = n * (n - 1) / 2
        if max_possible == 0:
            return 1.0 if len(largest_cc) == n else 0.0

        raw = (len(largest_cc) / n) * min(1.0, total_weight / (n - 1)) if n > 1 else 1.0
        return min(1.0, raw)

    def has_contradictions(self) -> list[tuple[str, str]]:
        return [
            (e.source, e.target)
            for e in self._edge_store.values()
            if e.relation_type == RelationType.CONTRADICTION
        ]

    def get_bias_flagged_nodes(self) -> list[Node]:
        return [n for n in self._nodes.values() if n.bias_flags]

    def get_causal_chains(
        self, user_explicit_only: bool = False, max_paths: int = 50,
    ) -> list[list[str]]:
        """Return root-to-leaf paths through causal/dependency subgraph.

        Args:
            user_explicit_only: If True, only traverse CAUSAL edges with
                USER_EXPLICIT source. If False, traverse CAUSAL + DEPENDENCY.
            max_paths: Maximum number of paths to return (early termination).
        """
        if user_explicit_only:
            directed_edges = [
                (e.source, e.target)
                for e in self._edge_store.values()
                if e.relation_type == RelationType.CAUSAL
                and e.source_type == EdgeSource.USER_EXPLICIT
            ]
        else:
            directed_edges = [
                (e.source, e.target)
                for e in self._edge_store.values()
                if e.relation_type in (RelationType.CAUSAL, RelationType.DEPENDENCY)
            ]
        if not directed_edges:
            return []
        sub_g = nx.DiGraph(directed_edges)
        paths: list[list[str]] = []
        roots = [n for n in sub_g.nodes() if sub_g.in_degree(n) == 0]
        leaves = [n for n in sub_g.nodes() if sub_g.out_degree(n) == 0]
        for root in roots:
            for leaf in leaves:
                for path in nx.all_simple_paths(sub_g, root, leaf):
                    paths.append(path)
                    if len(paths) >= max_paths:
                        return paths
        return paths

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "edges": [e.model_dump() for e in self._edge_store.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> IntentGraph:
        graph = cls()
        for nd in data.get("nodes", []):
            graph.add_node(Node(**nd))
        for ed in data.get("edges", []):
            graph.add_edge(Edge(**ed))
        return graph

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _would_create_dag_cycle(self, source: str, target: str) -> bool:
        """Check for cycles in the CAUSAL+DEPENDENCY subgraph only."""
        if target not in self._dag:
            return False  # target has no outgoing DAG edges, no cycle possible
        try:
            return nx.has_path(self._dag, target, source)
        except nx.NodeNotFound:
            return False
