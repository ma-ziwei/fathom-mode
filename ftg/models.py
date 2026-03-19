"""
Data models for the FtG framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Dimension(str, Enum):
    """The 6W Dimensional Coordinate System (5W1H)."""
    WHO = "who"
    WHAT = "what"
    WHY = "why"
    WHEN = "when"
    WHERE = "where"
    HOW = "how"

    @staticmethod
    def normalize(dim: "Dimension | str | None") -> str | None:
        """Convert Dimension enum or string to normalized string value."""
        if dim is None:
            return None
        return dim.value if isinstance(dim, Dimension) else str(dim).lower()


class NodeType(str, Enum):
    """Semantic sub-type of an information node."""
    FACT = "fact"
    BELIEF = "belief"
    VALUE = "value"
    INTENT = "intent"
    CONSTRAINT = "constraint"
    EMOTION = "emotion"
    ASSUMPTION = "assumption"
    GOAL = "goal"


class RelationType(str, Enum):
    """Semantic type of an edge between nodes."""
    CAUSAL = "causal"
    DEPENDENCY = "dependency"
    CONTRADICTION = "contradiction"
    CONDITIONAL = "conditional"
    SUPPORTS = "supports"


class NodeOrigin(str, Enum):
    """
    Provenance category of a node — how the information entered the graph.

    Supplements (not replaces) raw_quote. raw_quote gives the exact text;
    NodeOrigin gives the broad category for filtering and auditing.
    """
    USER_INPUT = "user_input"
    EXTERNAL_CONTEXT = "external_context"
    SYSTEM_INFERRED = "system_inferred"


class EdgeSource(str, Enum):
    """
    Provenance of an edge — how the relationship was established.

    USER_EXPLICIT: User explicitly stated a causal relationship
        (e.g., "because A, therefore B").
    USER_IMPLIED: LLM inferred the relationship from user's language,
        but the user did not explicitly state causality.
    ALGORITHM_INFERRED: Algorithm inferred the edge from dimension rules.
    """
    USER_EXPLICIT = "user_explicit"
    USER_IMPLIED = "user_implied"
    ALGORITHM_INFERRED = "algorithm"


class HypothesisStatus(str, Enum):
    """Status of a causal hypothesis in the verification pipeline."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DENIED = "denied"
    EXPIRED = "expired"
    PENDING_REDIRECT = "pending_redirect"


class DimensionCoverage(str, Enum):
    """Coverage granularity for a dimension across the current graph state."""
    NONE = "none"
    PARTIAL = "partial"
    SUFFICIENT = "sufficient"
    NOT_RELEVANT = "not_relevant"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class Node(BaseModel):
    """A single information node in the Intent Graph."""
    id: str
    content: str
    raw_quote: str = ""  # exact text from user that produced this node
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    node_type: NodeType = NodeType.FACT
    dimension: Optional[Dimension] = None
    origin: NodeOrigin = NodeOrigin.USER_INPUT
    bias_flags: list[str] = Field(default_factory=list)
    secondary_dimensions: list[str] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.id)


class Edge(BaseModel):
    """A directed relationship between two nodes."""
    source: str
    target: str
    relation_type: RelationType = RelationType.SUPPORTS
    source_type: EdgeSource = EdgeSource.USER_IMPLIED
    weight: float = Field(default=1.0, ge=0.0)


class CausalHypothesis(BaseModel):
    """
    A proposed causal relationship awaiting user verification.

    Stored outside the Intent Graph — hypotheses are NOT edges.
    They only become edges when confirmed by the user (USER_EXPLICIT).
    """
    id: str
    source_node_id: str
    target_node_id: str
    source_content: str = ""
    target_content: str = ""
    ambiguity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    status: HypothesisStatus = HypothesisStatus.PENDING
    verification_method: str = ""
    user_response: str = ""
    created_at_round: int = 0


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class FathomedIntent(BaseModel):
    """The output of a complete fathom session."""
    compiled_prompt: str
    fathom_score: float
    fathom_type: str  # "clean" | "override" | "stagnation" | "max_rounds"
    task_type: str
    rounds: int
    nodes: list[Node]
    edges: list[Edge]
    causal_hypotheses: list[CausalHypothesis] = Field(default_factory=list)
    bias_flags: list[str] = Field(default_factory=list)
    dimensions: dict[str, str] = Field(default_factory=dict)
    dimension_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    dialogue_history: list[dict] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def to_mermaid(self) -> str:
        """Render intent graph as Mermaid diagram."""
        lines = ["graph TD"]
        node_map = {n.id: n for n in self.nodes}
        for edge in self.edges:
            src = node_map.get(edge.source)
            tgt = node_map.get(edge.target)
            if src and tgt:
                src_label = src.content[:40].replace('"', "'")
                tgt_label = tgt.content[:40].replace('"', "'")
                rel = f"{edge.relation_type.value}: {edge.source_type.value}"
                lines.append(
                    f'    {edge.source}["{src_label}"] '
                    f'-->|"{rel}"| '
                    f'{edge.target}["{tgt_label}"]'
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Relay protocol response
# ---------------------------------------------------------------------------

@dataclass
class RelayResponse:
    """
    Unified response from FathomSession.relay().

    action:
        "ask_user"  — show display, wait for user reply
        "confirm"   — show display, wait for user to confirm or correct
        "execute"   — show display, then execute compiled_prompt
        "error"     — show display (error message)
    display:
        Pre-formatted text ready to show (English). Includes score, insight,
        question — the caller only needs to translate if needed.
    """
    action: str
    display: str
    compiled_prompt: str | None = None
    fathom_score: int = 0
    task_type: str | None = None


# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------


@dataclass
class FathomConfig:
    """
    Configuration for a single FathomSession.

    Only feature flags that still affect FtG behavior should live here.
    """

    enable_bias_detection: bool = True
    max_llm_tokens: int = 4096

    def to_dict(self) -> dict:
        return {
            "enable_bias_detection": self.enable_bias_detection,
            "max_llm_tokens": self.max_llm_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FathomConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
