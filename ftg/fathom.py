"""
Fathom — the main entry point of Fathom-then-Generate.

Fathom is a stateless factory that creates FathomSessions.
FathomSession drives the fathom loop for a single user request.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from ftg.causal import (
    CausalTracker,
    detect_causal_markers,
    match_markers_to_nodes,
    process_causal_feedback,
)
from ftg.backend import FtGBackend, FunctionBackend, is_legacy_backend, resolve_backend
from ftg.compiler import compile_intent_graph, BIAS_LABELS
from ftg.dimensions import find_target_dimension, DIMENSION_LABELS
from ftg.extractor import extract
from ftg.scoring import estimate_complexity
from ftg.graph import IntentGraph
from ftg.models import (
    CausalHypothesis,
    Dimension,
    Edge,
    EdgeSource,
    FathomConfig,
    FathomedIntent,
    RelationType,
    RelayResponse,
)
from ftg.questioner import (
    CausalCandidate,
    DimensionCandidate,
    RoundConstraints,
    RoundMaterials,
    RoundPacket,
    build_graph_summary,
    generate_question,
    generate_round_output,
)
from ftg.scoring import evaluate_fathom_gates

logger = logging.getLogger(__name__)


# DIMENSION_LABELS imported from ftg.dimensions
# BIAS_LABELS imported from ftg.compiler


# ---------------------------------------------------------------------------
# LLM callback interface
# ---------------------------------------------------------------------------

@dataclass
class LLMRequest:
    """What FtG sends to the LLM callback."""
    system_prompt: str
    user_prompt: str
    json_mode: bool = False
    temperature: float = 0.3


LLMFunction = Callable[[LLMRequest], str]
EmbedFunction = Callable[[list[str]], list[list[float]]]
DialogueFunction = Callable[[str, Optional[str]], str]


# ---------------------------------------------------------------------------
# Fathom — stateless factory
# ---------------------------------------------------------------------------


class Fathom:
    """
    Main entry point. Stateless factory — creates FathomSessions.

    Args:
        llm_fn: Function that takes an LLMRequest and returns a string response.
        embed_fn: Optional. For semantic deduplication and causal screening.
    """

    def __init__(
        self,
        llm_fn: LLMFunction | None = None,
        embed_fn: EmbedFunction | None = None,
        question_llm_fn: LLMFunction | None = None,
        execute_llm_fn: LLMFunction | None = None,
        backend: FtGBackend | None = None,
    ):
        if backend is not None and any(fn is not None for fn in (llm_fn, question_llm_fn, execute_llm_fn)):
            raise ValueError("Pass either backend or llm_fn/question_llm_fn/execute_llm_fn, not both.")
        if backend is None:
            if llm_fn is None:
                raise TypeError("Fathom requires either backend or llm_fn.")
            backend = FunctionBackend(
                llm_fn=llm_fn,
                question_llm_fn=question_llm_fn,
                execute_llm_fn=execute_llm_fn,
            )
        # Normalize: new FtGBackend (call/send_to_session) → internal 3-method shape
        self._backend = resolve_backend(backend)
        self._raw_backend = backend
        self.llm_fn = llm_fn
        self.embed_fn = embed_fn
        self.question_llm_fn = question_llm_fn or llm_fn

    # --- Class method factories (recommended for new code) ---

    @classmethod
    def from_openai(
        cls,
        api_key: str,
        model: str = "gpt-4o",
        embed_model: str = "text-embedding-3-small",
        **kwargs,
    ) -> "Fathom":
        """Create a Fathom instance backed by OpenAI. One line, zero config."""
        from ftg.contrib.openai import OpenAIBackend, make_openai_embed
        backend = OpenAIBackend(api_key=api_key, model=model, **kwargs)
        embed_fn = make_openai_embed(api_key=api_key, model=embed_model)
        return cls(backend=backend, embed_fn=embed_fn)

    @classmethod
    def from_anthropic(
        cls,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        **kwargs,
    ) -> "Fathom":
        """Create a Fathom instance backed by Anthropic Claude."""
        from ftg.contrib.anthropic import AnthropicBackend
        backend = AnthropicBackend(api_key=api_key, model=model, **kwargs)
        return cls(backend=backend)

    @classmethod
    def from_gemini(
        cls,
        api_key: str,
        model: str = "gemini-2.0-flash",
        **kwargs,
    ) -> "Fathom":
        """Create a Fathom instance backed by Google Gemini."""
        from ftg.contrib.gemini import GeminiBackend
        backend = GeminiBackend(api_key=api_key, model=model, **kwargs)
        return cls(backend=backend)

    @classmethod
    def from_deepseek(
        cls,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        **kwargs,
    ) -> "Fathom":
        """Create a Fathom instance backed by DeepSeek."""
        from ftg.contrib.deepseek import DeepSeekBackend
        backend = DeepSeekBackend(api_key=api_key, model=model, base_url=base_url, **kwargs)
        return cls(backend=backend)

    @classmethod
    def from_openclaw(cls, **kwargs) -> "Fathom":
        """Create a Fathom instance backed by OpenClaw."""
        from ftg.contrib.openclaw import OpenClawBackend
        backend = OpenClawBackend(**kwargs)
        return cls(backend=backend)

    # --- Internal ---

    def _bind_session_llms(self) -> tuple[str, LLMFunction, LLMFunction, LLMFunction]:
        session_id = uuid.uuid4().hex[:12]
        extract_llm_fn = self._backend.make_extract_llm_fn(session_id)
        question_llm_fn = self._backend.make_question_llm_fn(session_id)
        execute_llm_fn = self._backend.make_execute_llm_fn(session_id)
        return session_id, extract_llm_fn, question_llm_fn, execute_llm_fn

    def start(
        self,
        user_input: str,
        dialogue_fn: DialogueFunction,
        config: FathomConfig | None = None,
        user_context: str = "",
        understanding_context: str | None = None,
        execution_context: str | None = None,
    ) -> "FathomSession":
        backend_session_id, llm_fn, question_llm_fn, execute_llm_fn = self._bind_session_llms()
        return FathomSession(
            user_input=user_input,
            dialogue_fn=dialogue_fn,
            llm_fn=llm_fn,
            question_llm_fn=question_llm_fn,
            execute_llm_fn=execute_llm_fn,
            embed_fn=self.embed_fn,
            config=config or FathomConfig(),
            user_context=user_context,
            understanding_context=understanding_context,
            execution_context=execution_context,
            backend_session_id=backend_session_id,
        )


# ---------------------------------------------------------------------------
# FathomSession — drives the fathom loop
# ---------------------------------------------------------------------------

class FathomSession:
    """
    Drives the fathom loop for a single user request.

    Usage (relay — recommended for agent integration):
        session = fathom.start("Should I change jobs?", dialogue_fn=lambda q,i=None: "")
        response = session.relay()            # first question
        while response.action != "execute":
            user_reply = get_user_input()
            response = session.relay(user_reply)
        # response.compiled_prompt is ready

    Usage (blocking):
        session = fathom.start("Should I change jobs?", dialogue_fn=my_callback)
        result = session.run()
        print(result.compiled_prompt)

    Usage (step-by-step):
        session = fathom.start("Should I change jobs?", dialogue_fn=my_callback)
        while not session.is_fathomed:
            q, insight = session.step()
            answer = get_user_answer(q, insight)
            session.answer(answer)
        result = session.compile()
    """

    AFFIRMATIVE = frozenset({
        "yes", "y", "ok", "okay", "correct", "confirmed", "execute",
        "good", "fine", "right", "yep", "yeah", "looks good",
        "go ahead", "proceed", "do it", "let's go", "start",
        "sure", "absolutely", "agreed", "approve", "lgtm",
    })

    def __init__(
        self,
        user_input: str,
        dialogue_fn: DialogueFunction,
        llm_fn: LLMFunction,
        question_llm_fn: LLMFunction | None = None,
        execute_llm_fn: LLMFunction | None = None,
        embed_fn: EmbedFunction | None = None,
        config: FathomConfig | None = None,
        user_context: str = "",
        understanding_context: str | None = None,
        execution_context: str | None = None,
        backend_session_id: str | None = None,
    ):
        self._user_input = user_input
        self._dialogue_fn = dialogue_fn
        self._llm_fn = llm_fn
        self._question_llm_fn = question_llm_fn or llm_fn
        self._execute_llm_fn = execute_llm_fn or self._question_llm_fn
        self._embed_fn = embed_fn
        self._config = config or FathomConfig()
        self._backend_session_id = backend_session_id or uuid.uuid4().hex[:12]
        self._understanding_context = (
            user_context if understanding_context is None else understanding_context
        )
        self._execution_context = (
            self._understanding_context if execution_context is None else execution_context
        )

        self._graph = IntentGraph()
        self._round = 0
        self._phase = "questioning"
        self._task_type = "general"
        self._is_fathomed = False
        self._fathom_score = 0.0
        self._fathom_type = "not_fathomed"
        self._clarification_hints: dict[str, str] = {}
        self._conversation_history: list[dict] = []
        self._fathom_history: list[float] = []
        self._causal_tracker = CausalTracker()
        self._waived_dimensions: set[str] = set()
        self._dimension_assessment: dict[str, str] = {}
        self._dimension_states: dict[str, dict[str, object]] = {}
        self._dimension_semantics: dict[str, str] = {}
        self._complexity = 0.5
        self._consecutive_dim_rounds = 0
        self._last_question_mode = "dimension"
        self._verifying_hypothesis: str | None = None
        self._redirect_queue: list[dict] = []
        self._active_redirect: dict | None = None
        self._pending_question: str | None = None
        self._pending_insight: str | None = None
        self._not_relevant_streak: dict[str, int] = {}
        self._pending_dimension: str | None = None
        self._dimension_bound_responses: dict[str, dict] = {}
        self._attachment_contexts: list[dict] = []
        self._attachment_score_bonus: float = 0.0
        self._latest_new_node_ids: list[str] = []
        self._pending_round_action: str = "ask_user"

    @property
    def is_fathomed(self) -> bool:
        return self._is_fathomed

    @property
    def fathom_score(self) -> float:
        return self._fathom_score

    @property
    def graph(self) -> IntentGraph:
        return self._graph

    @property
    def attachment_contexts(self) -> list[dict]:
        return list(self._attachment_contexts)

    def add_attachment_context(
        self,
        *,
        label: str,
        summary: str = "",
        raw_ref: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Register file/artifact context without parsing it into the semantic graph."""
        self._attachment_contexts.append(
            {
                "label": str(label or "").strip(),
                "summary": str(summary or "").strip(),
                "raw_ref": str(raw_ref or "").strip(),
                "metadata": dict(metadata or {}),
            }
        )

    # ------------------------------------------------------------------
    # Core state-machine step (all public methods delegate here)
    # ------------------------------------------------------------------

    def _advance(
        self,
        user_text: str,
        *,
        initial: bool = False,
        ask_question: bool = True,
    ) -> tuple[str, str | None] | None:
        """
        Single internal state-machine step.

        Phase A — process user input:
          If initial: extract from first user input (no history recording).
          Otherwise: record conversation history, handle causal feedback,
          extract new information.
          Always: score, check saturation.

        Phase B — generate next question (if ask_question=True):
          If fathomed: return None.
          Otherwise: generate question, return (question, insight_or_None).

        When ask_question=False, only Phase A runs (used by answer()).
        """
        # --- Phase A: ingest ---
        pending_dimension = self._pending_dimension if not initial else None
        if not initial:
            self._conversation_history.append({
                "round": self._round,
                "question": self._pending_question or "",
                "insight": self._pending_insight or "",
                "answer": user_text,
            })
            if self._last_question_mode == "causal" and self._verifying_hypothesis:
                self._process_causal_feedback(user_text)

        extracted_nodes = self._extract_and_update(user_text)
        if pending_dimension and user_text.strip():
            self._bind_dimension_response(
                pending_dimension,
                user_text,
                [node.id for node in extracted_nodes],
            )
            self._pending_dimension = None
        self._update_saturation(
            latest_user_response="" if initial else user_text,
        )

        if not ask_question:
            return None

        # --- Phase B: generate question ---
        return self._generate_and_store_question()

    def _generate_and_store_question(self) -> tuple[str, str | None] | None:
        """Phase B helper: generate next question and store metadata. Shared by _advance() and step()."""
        if self._is_fathomed:
            return None

        self._round += 1
        q_data = self._generate_next_question()
        question = self._combine_response(q_data)
        insight = q_data.get("insight", "")
        self._pending_question = question
        self._pending_insight = insight
        self._last_question_mode = q_data.get("question_mode", "dimension")
        self._verifying_hypothesis = q_data.get("verifying_hypothesis")
        self._pending_dimension = q_data.get("target_dimension") if self._last_question_mode == "dimension" else None
        self._pending_round_action = q_data.get("round_action", "ask_user")

        return question, insight if insight else None

    # ------------------------------------------------------------------
    # Public interaction APIs (thin wrappers around _advance)
    # ------------------------------------------------------------------

    def run(self) -> FathomedIntent:
        """
        Run the full fathom loop until saturation (blocking).

        Two-layer loop: questioning until fathomed, then confirmation.
        Corrections in confirmation re-enter questioning via _advance(),
        mirroring relay() semantics exactly.
        """
        # Layer 1: questioning loop
        result = self._advance(self._user_input, initial=True)
        while result is not None:
            question, insight = result
            user_answer = self._dialogue_fn(question, insight)
            result = self._advance(user_answer)

        # Layer 2: confirmation loop (supports multiple corrections)
        while True:
            confirmation_summary = self._generate_confirmation_summary()
            user_feedback = self._dialogue_fn(
                f"Here is my current understanding of your request:\n\n"
                f"{confirmation_summary}\n\n"
                f"If this is correct, reply \"execute\"; if something is off, tell me what needs to be changed.",
                "Final confirmation before execution.",
            )
            if self._is_confirmation(user_feedback):
                break
            # Correction: re-enter questioning via _advance()
            result = self._advance(user_feedback)
            while result is not None:
                question, insight = result
                user_answer = self._dialogue_fn(question, insight)
                result = self._advance(user_answer)

        return self.compile()

    def step(self) -> tuple[str, str | None] | None:
        """
        Run one round. Returns (question, insight) or None if fathomed.
        After calling step(), provide the answer via answer().
        """
        if self._is_fathomed:
            return None

        if self._round == 0:
            return self._advance(self._user_input, initial=True)

        return self._generate_and_store_question()

    def answer(self, user_text: str) -> None:
        """Provide user's answer for the current round."""
        self._advance(user_text, ask_question=False)

    def compile(self) -> FathomedIntent:
        """Compile the Intent Graph into a FathomedIntent."""
        compiled_prompt = compile_intent_graph(
            self._graph, self._user_input, self._task_type,
            user_context=self._execution_context,
            dimension_bound_responses=self._dimension_bound_responses,
            dimension_states=self._dimension_states,
            attachment_contexts=self._attachment_contexts,
            dimension_assessment=self._dimension_assessment,
            fathom_score=self._fathom_score,
            causal_hypotheses=self._causal_tracker.hypotheses,
            waived_dimensions=self._waived_dimensions,
        )

        all_bias_flags: list[str] = []
        for n in self._graph.get_all_nodes():
            all_bias_flags.extend(n.bias_flags)

        return FathomedIntent(
            compiled_prompt=compiled_prompt,
            fathom_score=self._fathom_score,
            fathom_type=self._fathom_type,
            task_type=self._task_type,
            rounds=self._round,
            nodes=self._graph.get_all_nodes(),
            edges=self._graph.get_all_edges(),
            causal_hypotheses=self._causal_tracker.hypotheses,
            bias_flags=list(set(all_bias_flags)),
            dimensions=self._dimension_assessment,
            dimension_states=self._dimension_states,
            dialogue_history=self._conversation_history,
        )

    def _compile_now_relay(self) -> RelayResponse:
        compiled = self.compile()
        self._phase = "compiled_review"
        score = round(self._fathom_score * 100)
        return RelayResponse(
            action="review",
            display=(
                f"[Fathom Score {score}%]\n\n"
                f"{compiled.compiled_prompt}\n\n"
                f'If correct, reply "execute"; otherwise tell me what needs to change.'
            ),
            compiled_prompt=compiled.compiled_prompt,
            fathom_score=score,
            task_type=compiled.task_type,
        )

    def _stop_now_relay(self) -> RelayResponse:
        self._phase = "stopped"
        self._pending_question = None
        self._pending_insight = None
        self._pending_dimension = None
        self._pending_round_action = "stop"
        return RelayResponse(
            action="stop",
            display="Stopped current FtG session.",
            fathom_score=round(self._fathom_score * 100),
            task_type=self._task_type,
        )

    # ------------------------------------------------------------------
    # Relay protocol — single-call interaction API
    # ------------------------------------------------------------------

    def relay(self, user_message: str = "") -> RelayResponse:
        """
        Single-call interaction protocol for agent integration.

        Routes user_message based on session phase, returns a structured
        RelayResponse. Any integrator calls this in a loop — no state
        machine knowledge required.

        First call (after start): pass empty string or omit argument.
        Subsequent calls: pass the user's reply text.
        """
        command = (user_message or "").strip().lower()
        if command == "stop":
            return self._stop_now_relay()

        if self._phase == "stopped":
            return RelayResponse(action="error", display="Current FtG session is stopped.")

        if command == "fathom":
            return self._compile_now_relay()

        if self._phase == "compiled_review":
            if self._is_confirmation(user_message):
                compiled = self.compile()
                self._is_fathomed = True
                self._fathom_type = "manual"
                self._phase = "compiled"
                return RelayResponse(
                    action="execute",
                    display="Confirmed execution of current compiled result.",
                    compiled_prompt=compiled.compiled_prompt,
                    fathom_score=round(self._fathom_score * 100),
                    task_type=compiled.task_type,
                )
            self._phase = "questioning"
            result = self._advance(user_message)
            if result is None:
                return self._make_confirm_relay()
            return self._make_round_relay(*result, self._pending_round_action)

        if self._phase == "compiled":
            return RelayResponse(action="error", display="This session has already been compiled.")

        if self._phase == "confirming":
            if self._is_confirmation(user_message):
                compiled = self.compile()
                self._phase = "compiled"
                return RelayResponse(
                    action="execute",
                    display="Understanding and compilation complete, ready for execution.",
                    compiled_prompt=compiled.compiled_prompt,
                    fathom_score=round(self._fathom_score * 100),
                    task_type=compiled.task_type,
                )
            # User provided a correction — process via _advance and continue
            self._phase = "questioning"
            result = self._advance(user_message)
            if result is None:
                return self._make_confirm_relay()
            return self._make_round_relay(*result, self._pending_round_action)

        # --- Questioning phase ---
        if self._round == 0 and not self._conversation_history:
            result = self._advance(self._user_input, initial=True)
        else:
            result = self._advance(user_message)

        if result is None:
            return self._make_confirm_relay()
        return self._make_round_relay(*result, self._pending_round_action)

    @staticmethod
    def _is_confirmation(text: str) -> bool:
        return text.strip().lower() in FathomSession.AFFIRMATIVE

    def _make_ask_relay(self, question: str, insight: str | None) -> RelayResponse:
        return self._make_round_relay(question, insight, "ask_user")

    def _make_round_relay(self, question: str, insight: str | None, action: str) -> RelayResponse:
        score = round(self._fathom_score * 100)
        display = f"[Fathom Score {score}%]\n\n{question}"
        return RelayResponse(
            action=action,
            display=display,
            fathom_score=score,
            task_type=self._task_type,
        )

    def _make_confirm_relay(self) -> RelayResponse:
        self._phase = "confirming"
        score = round(self._fathom_score * 100)
        summary = self._generate_confirmation_summary()
        display = (
            f"[Fathom Score {score}%]\n\n"
            f"Understanding complete.\n\n"
            f"Please confirm the following understanding:\n\n{summary}\n\n"
            f"If correct, reply \"execute\"; if something needs to be changed, tell me what's off."
        )
        return RelayResponse(
            action="confirm",
            display=display,
            fathom_score=score,
            task_type=self._task_type,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_and_update(self, text: str) -> list:
        """Extract info from text and update graph."""
        if not text or not text.strip():
            logger.warning("Empty user input in round %d, skipping extraction", self._round)
            return []
        extract_result = extract(
            user_text=text,
            graph=self._graph,
            llm_fn=self._llm_fn,
            embed_fn=self._embed_fn,
            conversation_context=self._format_history(),
            task_type=self._task_type,
            causal_markers_fn=detect_causal_markers,
            match_markers_fn=match_markers_to_nodes,
            user_context=self._understanding_context,
            include_clarification_hints=False,
        )
        (
            nodes,
            edges,
            llm_task_type,
            dim_assessment,
            dim_states,
            dim_semantics,
            clarification_hints,
        ) = extract_result

        logger.debug("_extract_and_update received: %d nodes, %d edges, task_type=%s", len(nodes), len(edges), llm_task_type)
        if not nodes and len(text.strip()) > 20:
            logger.warning(
                "EXTRACTION RETURNED 0 NODES for %d-char user input (round %d). "
                "Possible LLM failure or API key issue. Input preview: %.100s",
                len(text), self._round, text,
            )
        for node in nodes:
            self._graph.add_node(node)
        for edge in edges:
            if not self._graph.add_edge(edge):
                logger.warning("Edge rejected: %s -> %s", edge.source, edge.target)

        if llm_task_type:
            self._task_type = llm_task_type

        # Merge dimension_assessment: only upgrade, never downgrade.
        # A failed extraction returns all-"missing" — must not wipe prior "covered".
        _ASSESSMENT_RANK = {"missing": 0, "not_relevant": 1, "covered_implicitly": 2, "covered": 3}
        if dim_assessment:
            for dim_val, new_status in dim_assessment.items():
                old_status = self._dimension_assessment.get(dim_val, "missing")
                if _ASSESSMENT_RANK.get(new_status, 0) >= _ASSESSMENT_RANK.get(old_status, 0):
                    self._dimension_assessment[dim_val] = new_status

        # Merge dimension_states: only update if new state has substance
        for dim_val, new_state in dim_states.items():
            if new_state.get("coverage_level") or new_state.get("evidence_present"):
                self._dimension_states[dim_val] = new_state
        if dim_semantics and not self._dimension_semantics:
            self._dimension_semantics = dim_semantics
        self._clarification_hints = clarification_hints
        self._complexity = estimate_complexity(self._graph)

        # Generate causal hypotheses
        self._causal_tracker.generate_hypotheses(
            new_nodes=nodes,
            graph=self._graph,
            current_round=self._round,
            embed_fn=self._embed_fn,
        )
        self._causal_tracker.expire_stale(self._round)

        # Delayed waive: require 2 consecutive not_relevant before waiving
        if self._dimension_assessment:
            for dim_val, status in self._dimension_assessment.items():
                if status == "not_relevant":
                    self._not_relevant_streak[dim_val] = self._not_relevant_streak.get(dim_val, 0) + 1
                    if self._not_relevant_streak[dim_val] >= 2:
                        self._waived_dimensions.add(dim_val)
                else:
                    self._not_relevant_streak[dim_val] = 0
        self._latest_new_node_ids = [node.id for node in nodes]
        return nodes

    def _bind_dimension_response(
        self,
        dimension: str,
        raw_text: str,
        node_ids: list[str],
    ) -> None:
        self._dimension_bound_responses[dimension] = {
            "raw_text": raw_text,
            "round": self._round,
            "node_ids": list(node_ids),
        }

    def _update_saturation(self, latest_user_response: str = "") -> None:
        state = evaluate_fathom_gates(
            graph=self._graph,
            fathom_history=self._fathom_history,
            round_count=self._round,
            waived_dimensions=self._waived_dimensions,
            causal_tracker=self._causal_tracker,
            dimension_assessment=self._dimension_assessment,
            dimension_bound_responses=self._dimension_bound_responses,
        )
        self._fathom_history.append(state["fathom_score"])
        self._is_fathomed = state["is_fathomed"]
        self._fathom_score = state["fathom_score"]
        self._fathom_type = state["fathom_type"]

    def _build_round_packet(self) -> RoundPacket:
        blocked_dimensions = self._waived_dimensions | set(self._dimension_bound_responses.keys())
        all_dimensions = {dim.value for dim in Dimension}
        dimension_candidate = None
        if blocked_dimensions != all_dimensions:
            target_dim = find_target_dimension(
                self._graph,
                self._complexity,
                blocked_dimensions,
                task_type=self._task_type,
            )
            dimension_candidate = DimensionCandidate(
                dimension=target_dim,
                reason="This dimension is most likely to change the answer direction, and the user hasn't provided a bound response for it yet.",
                gap=(self._dimension_states.get(target_dim, {}).get("open_gap", "") if self._dimension_states else "")
                or (self._dimension_semantics or {}).get(target_dim)
                or DIMENSION_LABELS.get(target_dim, target_dim),
                related_quotes=[
                    self._user_input.strip(),
                    *((self._conversation_history[-1:] or [{}])[0].get("answer", ""),),
                ],
            )
            dimension_candidate.related_quotes = [
                quote for quote in dimension_candidate.related_quotes if isinstance(quote, str) and quote.strip()
            ]

        causal_candidate = None
        if self._causal_tracker.should_verify_now(
            graph=self._graph,
            embed_fn=self._embed_fn,
            task_type=self._task_type,
        ):
            hypothesis = self._causal_tracker.get_next_hypothesis()
            if hypothesis:
                causal_candidate = CausalCandidate(
                    hypothesis_id=hypothesis.id,
                    source_node_id=hypothesis.source_node_id,
                    target_node_id=hypothesis.target_node_id,
                    source_content=hypothesis.source_content,
                    target_content=hypothesis.target_content,
                    reason="If this relationship holds, it would directly change the direction of analysis.",
                )

        if causal_candidate is not None and dimension_candidate is None:
            allowed_modes = ["causal"]
        elif causal_candidate is not None:
            allowed_modes = ["dimension", "causal"]
        else:
            allowed_modes = ["dimension"]

        return RoundPacket(
            constraints=RoundConstraints(
                task_type=self._task_type,
                round_count=self._round,
                allowed_modes=allowed_modes,
            ),
            materials=RoundMaterials(
                root_question=self._user_input,
                latest_user_response=(self._conversation_history[-1]["answer"] if self._conversation_history else ""),
                conversation_history=self._format_history(),
                graph_summary=build_graph_summary(self._graph),
                dimension_candidate=dimension_candidate,
                causal_candidate=causal_candidate,
                dimension_states=dict(self._dimension_states),
                dimension_bound_responses=dict(self._dimension_bound_responses),
                user_context=self._understanding_context,
            ),
        )

    def _generate_next_question(self) -> dict:
        """Decide question mode and generate the question."""
        if self._redirect_queue:
            mode = "redirect_confirm"
        else:
            mode = ""

        if mode == "redirect_confirm" and self._redirect_queue:
            redirect_item = self._redirect_queue.pop(0)
            self._active_redirect = redirect_item
            return generate_question(
                graph=self._graph,
                conversation_history=self._format_history(),
                task_type=self._task_type,
                llm_fn=self._question_llm_fn,
                round_count=self._round,
                complexity=self._complexity,
                waived_dimensions=self._waived_dimensions,
                question_mode="redirect_confirm",
                redirect_context=redirect_item,
                user_context=self._understanding_context,
            )

        packet = self._build_round_packet()
        q_data = generate_round_output(packet, self._question_llm_fn)
        if q_data.get("question_mode") == "dimension":
            self._consecutive_dim_rounds += 1
        else:
            self._consecutive_dim_rounds = 0
        return q_data

    def _process_causal_feedback(self, user_text: str) -> None:
        """Process user's response to a causal verification question."""
        hypothesis = None
        for h in self._causal_tracker.hypotheses:
            if h.id == self._verifying_hypothesis:
                hypothesis = h
                break
        if not hypothesis:
            return

        # Check if this is a redirect confirmation (not original verification)
        redirect_ctx = self._active_redirect
        if redirect_ctx is not None:
            self._active_redirect = None
            if self._is_confirmation(user_text):
                self._ground_redirect(redirect_ctx)
            return

        result = process_causal_feedback(
            user_text=user_text,
            hypothesis=hypothesis,
            question=self._pending_question or "",
            llm_fn=self._llm_fn,
        )

        verdict = result.get("verdict", "ambiguous")

        if verdict == "confirmed":
            self._causal_tracker.resolve_hypothesis(hypothesis.id, "confirmed")
            self._graph.add_edge(Edge(
                source=hypothesis.source_node_id,
                target=hypothesis.target_node_id,
                relation_type=RelationType.CAUSAL,
                source_type=EdgeSource.USER_EXPLICIT,
            ))
        elif verdict == "denied":
            self._causal_tracker.resolve_hypothesis(hypothesis.id, "denied")
        elif verdict == "redirected":
            self._causal_tracker.resolve_hypothesis(hypothesis.id, "denied")
            alt = result.get("alternative_cause", "")
            if alt:
                self._redirect_queue.append({
                    # questioner.py (redirect_confirm) reads these:
                    "hypothesis_id": hypothesis.id,
                    "original_source": hypothesis.source_content,
                    "original_target": hypothesis.target_content,
                    "user_response": user_text,
                    # _ground_redirect() reads these:
                    "original_hypothesis_id": hypothesis.id,
                    "original_target_node_id": hypothesis.target_node_id,
                    "proposed_cause_text": alt,
                    "user_quote": user_text,
                })
        else:
            pass  # ambiguous/partial — leave pending

    def _ground_redirect(self, redirect_ctx: dict) -> None:
        """Ground a redirect's alternative cause to a node and add the causal edge."""
        from ftg.causal import _overlap_score
        from ftg.models import Node, NodeOrigin, NodeType

        proposed_text = redirect_ctx["proposed_cause_text"]
        target_node_id = redirect_ctx["original_target_node_id"]

        best_node: Node | None = None
        best_score = 0.0
        for node in self._graph.get_all_nodes():
            score = _overlap_score(
                proposed_text.lower(), node.content.lower(),
                node.raw_quote.lower() if node.raw_quote else "",
            )
            if score > best_score:
                best_score = score
                best_node = node

        if best_node and best_score >= 0.3:
            cause_node_id = best_node.id
        else:
            cause_node_id = f"redirect_{uuid.uuid4().hex[:6]}"
            new_node = Node(
                id=cause_node_id,
                content=proposed_text[:200],
                raw_quote=redirect_ctx.get("user_quote", ""),
                confidence=0.6,
                node_type=NodeType.ASSUMPTION,
                dimension=Dimension.WHY,
                origin=NodeOrigin.SYSTEM_INFERRED,
            )
            self._graph.add_node(new_node)

        if target_node_id and self._graph.get_node(target_node_id):
            self._graph.add_edge(Edge(
                source=cause_node_id,
                target=target_node_id,
                relation_type=RelationType.CAUSAL,
                source_type=EdgeSource.USER_EXPLICIT,
            ))

    def _generate_confirmation_summary(self) -> str:
        """Generate a 5W1H structured summary for user confirmation."""
        lines: list[str] = []
        for dim in Dimension:
            nodes = self._graph.get_nodes_by_dimension(dim, include_secondary=True)
            if nodes:
                lines.append(f"[{DIMENSION_LABELS.get(dim.value, dim.value)}]")
                for n in nodes:
                    provenance = "you explicitly stated" if n.raw_quote else "inferred from context"
                    lines.append(f"  - {n.content} ({provenance})")

        causal_edges = [
            e for e in self._graph.get_all_edges()
            if e.relation_type == RelationType.CAUSAL
        ]
        if causal_edges:
            lines.append("\n[Your Explicit Causal Judgments]")
            for e in causal_edges:
                src = self._graph.get_node(e.source)
                tgt = self._graph.get_node(e.target)
                if src and tgt:
                    lines.append(f"  - {src.content} -> causes -> {tgt.content}")

        biased = self._graph.get_bias_flagged_nodes()
        if biased:
            lines.append("\n[Bias Signals to Watch For]")
            for n in biased:
                for flag in n.bias_flags:
                    lines.append(f"  - {BIAS_LABELS.get(flag, flag)}: {n.content}")

        return "\n".join(lines)

    def _format_history(self) -> str:
        """Format conversation history for prompt context."""
        if not self._conversation_history:
            return ""
        lines: list[str] = []
        for turn in self._conversation_history[-5:]:
            lines.append(f"Q: {turn['question']}")
            lines.append(f"A: {turn['answer']}")
        return "\n".join(lines)

    @staticmethod
    def _combine_response(q_data: dict) -> str:
        """Format one FtG round as progress + tension + next question."""
        response = (q_data.get("response") or "").strip()
        insight = (q_data.get("insight") or "").strip()
        question = (q_data.get("question") or "").strip()

        parts: list[str] = []
        if response and insight:
            # Ensure sentence boundary between response and insight
            if response[-1] not in ".!?":
                response += "."
            parts.append(f"{response} {insight}")
        elif response:
            parts.append(response)
        elif insight:
            parts.append(insight)
        if question:
            parts.append(question)
        parts.append('Say "fathom" to compile, or "stop" to exit Fathom Mode.')
        combined = "\n\n".join(parts)
        # Strip residual XML tags that may leak from LLM output
        combined = re.sub(r"</?[a-zA-Z_][a-zA-Z0-9_]*>?|</", "", combined).strip()
        return combined

    # ------------------------------------------------------------------
    # Serialization (for CLI / stateless deployments)
    # ------------------------------------------------------------------

    STATE_SCHEMA_VERSION = 1

    def to_state(self) -> dict:
        """Serialize all session state except function callbacks."""
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "user_input": self._user_input,
            "user_context": self._understanding_context,
            "understanding_context": self._understanding_context,
            "execution_context": self._execution_context,
            "clarification_hints": dict(self._clarification_hints),
            "graph": self._graph.to_dict(),
            "round": self._round,
            "phase": self._phase,
            "task_type": self._task_type,
            "is_fathomed": self._is_fathomed,
            "fathom_score": self._fathom_score,
            "fathom_type": self._fathom_type,
            "conversation_history": self._conversation_history,
            "fathom_history": self._fathom_history,
            "causal_tracker": self._causal_tracker.to_dict(),
            "waived_dimensions": list(self._waived_dimensions),
            "dimension_assessment": self._dimension_assessment,
            "dimension_states": self._dimension_states,
            "dimension_semantics": self._dimension_semantics,
            "complexity": self._complexity,
            "consecutive_dim_rounds": self._consecutive_dim_rounds,
            "last_question_mode": self._last_question_mode,
            "verifying_hypothesis": self._verifying_hypothesis,
            "redirect_queue": self._redirect_queue,
            "active_redirect": self._active_redirect,
            "pending_question": self._pending_question,
            "pending_insight": self._pending_insight,
            "not_relevant_streak": self._not_relevant_streak,
            "pending_dimension": self._pending_dimension,
            "dimension_bound_responses": self._dimension_bound_responses,
            "attachment_contexts": self._attachment_contexts,
            "attachment_score_bonus": self._attachment_score_bonus,
            "latest_new_node_ids": self._latest_new_node_ids,
            "pending_round_action": self._pending_round_action,
            "backend_session_id": self._backend_session_id,
            "config": self._config.to_dict(),
        }

    @classmethod
    def from_state(
        cls,
        state: dict,
        llm_fn: "LLMFunction | None" = None,
        question_llm_fn: "LLMFunction | None" = None,
        execute_llm_fn: "LLMFunction | None" = None,
        embed_fn: "EmbedFunction | None" = None,
        dialogue_fn: "DialogueFunction | None" = None,
        backend: "FtGBackend | None" = None,
    ) -> "FathomSession":
        """Reconstruct a session from serialized state + fresh callbacks."""
        version = state.get("schema_version", 0)
        if version < 1:
            state = cls._migrate_v0_to_v1(state)

        config_data = state.get("config") or state.get("saturation_config") or {}
        config = FathomConfig.from_dict(config_data) if config_data else FathomConfig()

        backend_session_id = state.get("backend_session_id") or uuid.uuid4().hex[:12]
        if backend is not None:
            resolved = resolve_backend(backend)
            llm_fn = resolved.make_extract_llm_fn(backend_session_id)
            question_llm_fn = resolved.make_question_llm_fn(backend_session_id)
            execute_llm_fn = resolved.make_execute_llm_fn(backend_session_id)
        if llm_fn is None:
            raise TypeError("FathomSession.from_state requires llm_fn or backend.")

        session = cls(
            user_input=state["user_input"],
            dialogue_fn=dialogue_fn or (lambda q, insight=None: ""),
            llm_fn=llm_fn,
            question_llm_fn=question_llm_fn or llm_fn,
            execute_llm_fn=execute_llm_fn or question_llm_fn or llm_fn,
            embed_fn=embed_fn,
            config=config,
            user_context=state.get("user_context", ""),
            understanding_context=state.get("understanding_context", state.get("user_context", "")),
            execution_context=state.get("execution_context", state.get("user_context", "")),
            backend_session_id=backend_session_id,
        )
        session._graph = IntentGraph.from_dict(state["graph"])
        session._round = state["round"]
        session._phase = state.get("phase", "questioning")
        session._task_type = state["task_type"]
        session._is_fathomed = state["is_fathomed"]
        session._fathom_score = state["fathom_score"]
        session._fathom_type = state["fathom_type"]
        session._conversation_history = state["conversation_history"]
        session._fathom_history = state["fathom_history"]
        session._clarification_hints = dict(state.get("clarification_hints", {}))
        session._causal_tracker = CausalTracker.from_dict(state["causal_tracker"])
        session._waived_dimensions = set(state["waived_dimensions"])
        session._dimension_assessment = state["dimension_assessment"]
        session._dimension_states = dict(state.get("dimension_states", {}))
        session._dimension_semantics = state.get("dimension_semantics", {})
        session._complexity = state["complexity"]
        session._consecutive_dim_rounds = state["consecutive_dim_rounds"]
        session._last_question_mode = state["last_question_mode"]
        session._verifying_hypothesis = state["verifying_hypothesis"]
        session._redirect_queue = state["redirect_queue"]
        session._active_redirect = state.get("active_redirect")
        session._pending_question = state["pending_question"]
        session._pending_insight = state["pending_insight"]
        session._not_relevant_streak = state.get("not_relevant_streak", {})
        session._pending_dimension = state.get("pending_dimension")
        session._dimension_bound_responses = dict(state.get("dimension_bound_responses", {}))
        session._attachment_contexts = list(state.get("attachment_contexts", []))
        session._attachment_score_bonus = float(state.get("attachment_score_bonus", 0.0) or 0.0)
        session._latest_new_node_ids = list(state.get("latest_new_node_ids", []))
        session._pending_round_action = state.get("pending_round_action", "ask_user")
        session._backend_session_id = backend_session_id
        return session

    @staticmethod
    def _migrate_v0_to_v1(state: dict) -> dict:
        """Migrate v0 (pre-versioned) state to v1 schema."""
        state = dict(state)
        state["schema_version"] = 1
        if "config" not in state and "saturation_config" not in state:
            state["config"] = FathomConfig().to_dict()
        if "dimension_semantics" not in state:
            state["dimension_semantics"] = {}
        if "dimension_states" not in state:
            state["dimension_states"] = {}
        if "redirect_queue" not in state:
            state["redirect_queue"] = []
        if "pending_insight" not in state:
            state["pending_insight"] = None
        if "user_context" not in state:
            state["user_context"] = ""
        if "understanding_context" not in state:
            state["understanding_context"] = state.get("user_context", "")
        if "execution_context" not in state:
            state["execution_context"] = state.get("user_context", "")
        if "clarification_hints" not in state:
            state["clarification_hints"] = {}
        if "active_redirect" not in state:
            state["active_redirect"] = None
        if "not_relevant_streak" not in state:
            state["not_relevant_streak"] = {}
        if "pending_dimension" not in state:
            state["pending_dimension"] = None
        if "dimension_bound_responses" not in state:
            state["dimension_bound_responses"] = {}
        if "attachment_contexts" not in state:
            state["attachment_contexts"] = []
        if "attachment_score_bonus" not in state:
            state["attachment_score_bonus"] = 0.0
        if "latest_new_node_ids" not in state:
            state["latest_new_node_ids"] = []
        if "pending_round_action" not in state:
            state["pending_round_action"] = "ask_user"
        if "backend_session_id" not in state:
            state["backend_session_id"] = uuid.uuid4().hex[:12]
        return state
