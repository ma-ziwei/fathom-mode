"""
Fathom-then-Generate (FtG) — Reversible Intent Alignment Protocol.

Optimizes AI input by systematically eliciting complete, bias-corrected
information from humans before any LLM generation.

Quickstart:

    from ftg import Fathom

    fathom = Fathom.from_openai(api_key="sk-...")
    session = fathom.start("Should I change jobs?", dialogue_fn=my_callback)
    response = session.relay()

Provider backends are in ftg.contrib:

    from ftg.contrib.openai import OpenAIBackend
    from ftg.contrib.anthropic import AnthropicBackend
    from ftg.contrib.gemini import GeminiBackend
    from ftg.contrib.deepseek import DeepSeekBackend
    from ftg.contrib.openclaw import OpenClawBackend
"""

from ftg.backend import CompositeBackend, FtGBackend, FunctionBackend
from ftg.fathom import (
    Fathom,
    FathomSession,
    LLMRequest,
)
from ftg.models import (
    FathomConfig,
    FathomedIntent,
    RelayResponse,
    Dimension,
    Node,
    Edge,
    CausalHypothesis,
    HypothesisStatus,
    NodeType,
    NodeOrigin,
    RelationType,
    EdgeSource,
)
from ftg.graph import IntentGraph

__version__ = "0.1.0"

__all__ = [
    # Core API
    "Fathom",
    "FathomSession",
    "FathomConfig",
    "FathomedIntent",
    "RelayResponse",
    "LLMRequest",
    "FtGBackend",
    "CompositeBackend",
    "FunctionBackend",
    # Data models
    "IntentGraph",
    "Dimension",
    "Node",
    "Edge",
    "CausalHypothesis",
    "HypothesisStatus",
    "NodeType",
    "NodeOrigin",
    "RelationType",
    "EdgeSource",
    # Version
    "__version__",
]
