# Fathom Mode

**Your AI answers before it understands. Fathom Mode fixes that.**

[![PyPI version](https://img.shields.io/pypi/v/fathom-mode)](https://pypi.org/project/fathom-mode/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Fathom Mode is an intent compiler for AI. It sits between your user and your LLM, asking targeted questions to understand what the user *actually* wants — then compiles that understanding into a structured prompt that any model can execute precisely.

---

## See the difference

> **Without Fathom Mode** — ask "Should I start a YouTube channel?"
> → 1,500-word generic guide. Covers everything. Answers nothing specific. ~2,000 output tokens.
>
> **With Fathom Mode** — same question, 3 rounds of targeted questions
> → 400-word personalized strategy with one clear recommendation. ~550 output tokens.
>
> **Same model. Same question. 39% fewer total tokens. Dramatically better answer.**

Why? When your input is vague, LLMs hedge — they generate long, generic responses covering every possibility. Fathom Mode eliminates the ambiguity upfront, so the model doesn't have to guess. Better input → shorter, more precise output → lower cost.

---

## Quick start

```bash
pip install "fathom-mode[openai] @ git+https://github.com/ma-ziwei/fathom-mode.git"
```

```python
from ftg import Fathom

fathom = Fathom.from_openai(api_key="sk-...")

session = fathom.start("Should I start a YouTube channel?")
result = session.run()

print(result.compiled_prompt)  # Structured prompt for any downstream LLM
print(result.fathom_score)     # How deeply intent was understood (0 → 1)
```

That's it. Fathom Mode handles the rest — extracting intent, building a structured graph, asking the right questions, and compiling everything into a prompt your model can act on.

---

## How it works

Each round, Fathom Mode does two things:

1. **Gives the user a high-quality response** based on what it knows so far
2. **Asks one targeted question** to deepen understanding

The user stays in control. They can keep answering to go deeper, or say `fathom` at any time to compile and execute.

Under the hood, each round runs a **sandwich architecture** — two LLM calls bracketing a fully deterministic pipeline:

```
User message
    ↓
[Extract LLM]        ← 1 LLM call (stateless — no memory between calls)
    ↓
Intent Graph          ← deterministic: graph update, edge inference,
Fathom Score             scoring, dimension selection, causal tracking
Causal Verification      — zero LLM calls for any of this
    ↓
[Question LLM]        ← 1 LLM call (stateful — shares the conversation)
    ↓
Response + Question → User
```

Everything between the two LLM calls is deterministic. Same graph → same compiled prompt, always. No randomness, no model variance, fully auditable.

---

## Why not just prompt engineer?

Prompt engineering optimizes what the **developer** writes.
Fathom Mode optimizes what the **user** means.

The best system prompt in the world can't fix a user who said
"help me with that thing" instead of "help me write a resignation
letter that's firm but not bridge-burning, for a boss I still respect."

Fathom Mode gets you from the first sentence to the second — in about 3 rounds.

---

## Supports any LLM

```python
from ftg import Fathom

# Built-in providers
fathom = Fathom.from_openai(api_key="...")       # GPT-4o, GPT-4, etc.
fathom = Fathom.from_anthropic(api_key="...")     # Claude
fathom = Fathom.from_gemini(api_key="...")        # Gemini
fathom = Fathom.from_deepseek(api_key="...")      # DeepSeek

# Or bring your own
from ftg import LLMRequest

def my_llm(req: LLMRequest) -> str:
    return call_my_api(req.system_prompt, req.user_prompt)

fathom = Fathom(llm_fn=my_llm)
```

Zero provider lock-in. One callback function is all you need.

---

## What makes this different

|  | Prompt Engineering | Fathom Mode |
|---|---|---|
| **Optimizes** | Developer's system prompt | User's actual intent |
| **When** | Before deployment | At interaction time |
| **Auditable** | No — intent is a black box | Yes — every round shows understanding |
| **Causal reasoning** | LLM guesses causation | User's causation protected by code |
| **Output cost** | Verbose (LLM hedges uncertainty) | Precise (uncertainty resolved upfront) |
| **Integration** | Part of your prompt | Protocol layer — works with any model or framework |

---

## When NOT to use Fathom Mode

- **Simple factual queries** — "What's the capital of France?" doesn't need intent alignment
- **Fully specified tasks** — if the user already gave you everything, skip the questions
- **Zero-latency chat** — each round adds 2 LLM calls; Fathom Mode trades speed for precision
- **You need a full agent framework** — Fathom Mode is a protocol layer, not an orchestrator. It pairs *with* frameworks like LangChain or OpenClaw, not *instead of* them.

Fathom Mode is built for **complex, ambiguous, high-stakes tasks** where getting the intent right matters more than answering fast.

---

## Integration

### OpenClaw integration

**macOS / Linux:**
```bash
pip3 install "fathom-mode[openclaw] @ git+https://github.com/ma-ziwei/fathom-mode.git"
python3 -m ftg install-openclaw
```

**Windows:**
```powershell
py -m pip install "fathom-mode[openclaw] @ git+https://github.com/ma-ziwei/fathom-mode.git"
py -m ftg install-openclaw
```

Then in any OpenClaw chat, say **"fathom mode"** to activate.

### Relay protocol (for custom agents)

A single entry point that handles all state transitions:

```python
session = fathom.start("Plan my career transition")

response = session.relay(user_message)

response.action          # "ask_user" | "review" | "execute" | "stop"
response.display         # Text to show the user
response.compiled_prompt # Available when action == "execute"
response.fathom_score    # Current depth of understanding
```

User commands during a session:
- Keep talking → deeper understanding, better responses each round
- `fathom` → compile current understanding into structured prompt
- `stop` → end session

### CLI

```bash
# Start a session
python3 -m ftg start "Should I change jobs?"

# Continue with relay (for agent integration)
python3 -m ftg relay "Should I change jobs?"
python3 -m ftg relay --session <id> "I've been at my company for 5 years..."

# Session management
python3 -m ftg list
python3 -m ftg status <session_id>
```

---

## Under the hood

For developers who want to understand the theory:

**Intent Graph** — Directed graph (NetworkX) mapping extracted information across 6 dimensions (WHO, WHAT, WHY, WHEN, WHERE, HOW). Every node carries provenance — was this stated by the user, implied, or inferred by the system?

**Fathom Score** — Three-layer depth model. Surface (what/when/where) → Depth (why/who/how) → Bedrock (causally verified). Measures how *deeply* intent is understood, not just how much information was collected. Based on Shannon entropy with mathematically derived parameters.

**Causal Fathom Protocol** — The hardest constraint in the system: if the LLM tries to create a causal relationship the user didn't explicitly state, it gets automatically downgraded. This is a code-level `if/else`, not a prompt instruction. The user's causal reasoning is preserved, never replaced by LLM correlation.

**Deterministic Compilation** — The compiled prompt is pure graph traversal + templates. Zero LLM calls. The same Intent Graph always produces the same output. Fully reproducible, fully auditable.

**White-Box Audit** — Fathom Mode deliberately uses an LLM for extraction (not rules), because auditing an LLM's understanding is the only way to predict how a downstream LLM will interpret the same input. Every round externalizes this understanding — the user sees what the system thinks, and can correct it before it matters.

📄 [Paper (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6229858) &nbsp;·&nbsp; 📝 [Blog Series](https://open.substack.com/pub/ziweima/p/intent-decay-the-failure-mode-nobody)

---

## Project structure

```
ftg/
├── fathom.py        # Session state machine + Fathom factory
├── extractor.py     # LLM extraction + deterministic validation
├── questioner.py    # Question generation + fallback chain
├── compiler.py      # Deterministic prompt compilation
├── scoring.py       # Three-layer Fathom Score
├── graph.py         # Intent Graph (NetworkX)
├── causal.py        # Causal verification protocol
├── dimensions.py    # 6W coordinate system
├── models.py        # All data structures (Pydantic)
├── backend.py       # Provider abstraction
├── utils.py         # Helpers
├── cli.py           # CLI entry point
└── contrib/         # Provider backends
    ├── openai.py    ├── anthropic.py
    ├── gemini.py    ├── deepseek.py
    └── openclaw.py
```

---

## Contributing

This is a `0.1.0` research preview. The core protocol is stable, but the API surface may evolve.

Bug reports, test cases, and integration examples are especially welcome.

---

## Citation

If you use Fathom Mode in research, please cite:

```bibtex
@article{ma2026fathom,
  title={Fathom-then-Generate: A Reversible Intent Alignment Protocol},
  author={Ma, Ziwei},
  journal={SSRN},
  year={2026}
}
```

---

## License

MIT
