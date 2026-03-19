#!/usr/bin/env python3
"""
FtG CLI — Fathom-then-Generate command-line interface.

Manages multi-turn fathom sessions with state persisted to disk.
Designed for integration with OpenClaw skills or any bash-based automation.

Usage:
    fathom start "Should I buy a car or invest in stocks?"
    fathom answer <session_id> "I need a car for commuting..."
    fathom confirm <session_id> "yes"
    fathom compile <session_id>
    fathom status <session_id>
    fathom list

Environment:
    FTG_LLM_PROVIDER — auto|gemini|openai|anthropic|deepseek
    GEMINI_API_KEY   — Google Gemini API key
    GEMINI_MODEL     — Gemini model name (default: gemini-2.0-flash)
    DEEPSEEK_API_KEY — DeepSeek API key
    DEEPSEEK_MODEL   — DeepSeek model name (default: deepseek-chat)
    OPENAI_API_KEY   — OpenAI-compatible API key
    OPENAI_MODEL     — OpenAI model name (default: gpt-4o)
    OPENAI_BASE_URL  — Optional OpenAI-compatible base URL
    ANTHROPIC_API_KEY — Anthropic API key
    ANTHROPIC_MODEL   — Anthropic model name (default: claude-sonnet-4-20250514)
"""

from __future__ import annotations
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path


from ftg import Fathom, FathomSession, LLMRequest
from ftg.backend import CompositeBackend, FunctionBackend
from ftg.contrib.openclaw import OpenClawBackend
SESSIONS_DIR = Path.home() / ".ftg-sessions"
SESSION_TTL = 30 * 60  # 30 minutes
OPENCLAW_WSL_DISTRO = os.environ.get("FTG_OPENCLAW_WSL_DISTRO", "Ubuntu")


# ------------------------------------------------------------------
# LLM backends
# ------------------------------------------------------------------

_EXTRACT_SIGNAL = "information understanding engine"


def _load_json_if_exists(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _openclaw_candidate_paths(*parts: str) -> list[Path]:
    candidates = [Path.home() / ".openclaw"]
    if sys.platform == "win32":
        candidates.append(
            Path(rf"\\wsl.localhost\{OPENCLAW_WSL_DISTRO}\home\{os.environ.get('FTG_OPENCLAW_WSL_USER', 'ubuntu')}\.openclaw")
        )
    paths: list[Path] = []
    for base in candidates:
        candidate = base.joinpath(*parts)
        if candidate.exists():
            paths.append(candidate)
    return paths


def _read_openclaw_auth_profile(provider: str) -> dict | None:
    for path in _openclaw_candidate_paths("agents", "main", "agent", "auth-profiles.json"):
        data = _load_json_if_exists(path)
        if not isinstance(data, dict):
            continue
        profiles = data.get("profiles", {})
        if not isinstance(profiles, dict):
            continue
        for profile in profiles.values():
            if (
                isinstance(profile, dict)
                and profile.get("provider") == provider
                and profile.get("type") == "api_key"
                and profile.get("key")
            ):
                return profile
    return None


def _read_openclaw_primary_model(provider: str) -> str | None:
    for path in _openclaw_candidate_paths("openclaw.json"):
        data = _load_json_if_exists(path)
        if not isinstance(data, dict):
            continue
        primary = (
            data.get("agents", {})
            .get("defaults", {})
            .get("model", {})
            .get("primary")
        )
        if isinstance(primary, str) and primary.startswith(f"{provider}/"):
            return primary.split("/", 1)[1]
    return None


def _resolve_api_key(env_name: str, provider: str) -> str:
    api_key = os.environ.get(env_name, "")
    if api_key:
        return api_key

    profile = _read_openclaw_auth_profile(provider)
    if profile:
        return str(profile["key"])

    print(json.dumps({"error": f"{env_name} not set"}))
    sys.exit(1)

def _configured_provider() -> str:
    provider = os.environ.get("FTG_LLM_PROVIDER", "").strip().lower()
    if provider and provider != "auto":
        return provider

    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if _read_openclaw_auth_profile("anthropic"):
        return "anthropic"
    if _read_openclaw_auth_profile("openai"):
        return "openai"

    print(json.dumps({
        "error": (
            "No LLM provider configured. Set FTG_LLM_PROVIDER plus a matching API key, "
            "or export one of GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "or DEEPSEEK_API_KEY. FtG can also reuse OpenClaw auth profiles for "
            "Anthropic/OpenAI when present."
        )
    }))
    sys.exit(1)


def _make_gemini_llm_fn():
    from ftg.contrib.gemini import GeminiBackend

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("FTG ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        print(json.dumps({"error": "GEMINI_API_KEY not set"}))
        sys.exit(1)

    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    backend = GeminiBackend(api_key=api_key, model=model)
    return backend.call


def _make_deepseek_llm_fn():
    from ftg.contrib.deepseek import DeepSeekBackend

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None

    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    backend = DeepSeekBackend(api_key=api_key, model=model)
    return backend.call


def _make_openai_llm_fn():
    import openai

    api_key = _resolve_api_key("OPENAI_API_KEY", "openai")

    model = os.environ.get("OPENAI_MODEL") or _read_openclaw_primary_model("openai") or "gpt-4o"
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def llm_fn(req: LLMRequest) -> str:
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.user_prompt},
            ],
            "temperature": req.temperature,
        }
        if req.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    return llm_fn


def _make_anthropic_llm_fn():
    import anthropic

    api_key = _resolve_api_key("ANTHROPIC_API_KEY", "anthropic")

    model = (
        os.environ.get("ANTHROPIC_MODEL")
        or _read_openclaw_primary_model("anthropic")
        or "claude-sonnet-4-20250514"
    )
    client = anthropic.Anthropic(api_key=api_key)

    def llm_fn(req: LLMRequest) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=req.system_prompt,
            messages=[{"role": "user", "content": req.user_prompt}],
            temperature=req.temperature,
        )
        return response.content[0].text

    return llm_fn


def _make_routing_llm_fn():
    """Create the configured LLM callback, preserving Gemini+DeepSeek routing."""
    provider = _configured_provider()

    if provider == "openai":
        return _make_openai_llm_fn()

    if provider == "anthropic":
        return _make_anthropic_llm_fn()

    if provider == "deepseek":
        deepseek = _make_deepseek_llm_fn()
        if not deepseek:
            print(json.dumps({"error": "DEEPSEEK_API_KEY not set"}))
            sys.exit(1)
        return deepseek

    if provider != "gemini":
        print(json.dumps({"error": f"Unsupported FTG_LLM_PROVIDER: {provider}"}))
        sys.exit(1)

    gemini = _make_gemini_llm_fn()
    deepseek = _make_deepseek_llm_fn()
    if not deepseek:
        return gemini

    def llm_fn(req: LLMRequest) -> str:
        if _EXTRACT_SIGNAL in req.system_prompt.lower():
            try:
                return deepseek(req)
            except Exception:
                return gemini(req)
        return gemini(req)

    return llm_fn


# ------------------------------------------------------------------
# Session persistence
# ------------------------------------------------------------------

def _clean_session_id(session_id: str) -> str:
    return session_id.strip('"').strip("'")


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{_clean_session_id(session_id)}.json"


def _save_session(session_id: str, session: FathomSession, meta: dict):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "session_state": session.to_state(),
        "meta": meta,
    }
    _session_path(session_id).write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_session(session_id: str) -> tuple[FathomSession, dict]:
    path = _session_path(session_id)
    if not path.exists():
        print(json.dumps({"error": f"Session {session_id} not found"}))
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["meta"]

    if time.time() - meta.get("last_active", 0) > SESSION_TTL:
        path.unlink(missing_ok=True)
        print(json.dumps({"error": f"Session {session_id} expired"}))
        sys.exit(1)

    backend = _make_ftg_backend()
    try:
        session = FathomSession.from_state(
            data["session_state"],
            backend=backend,
        )
    except TypeError:
        session = FathomSession.from_state(
            data["session_state"],
            llm_fn=_make_routing_llm_fn(),
        )
    return session, meta


def _touch_meta(meta: dict) -> dict:
    meta["last_active"] = time.time()
    return meta


def _phase_for_relay_action(action: str | None, current_phase: str | None = None) -> str:
    mapping = {
        "ask_user": "questioning",
        "answer_now": "questioning",
        "confirm": "confirming",
        "review": "compiled_review",
        "execute": "compiled",
        "stop": "stopped",
    }
    if (action or "").strip() == "error" and current_phase:
        return current_phase
    return mapping.get((action or "").strip(), current_phase or "questioning")


def _make_ftg_backend():
    base_llm = _make_routing_llm_fn()
    mode = os.environ.get("FTG_BACKEND", "").strip().lower()
    persistent_session_key = os.environ.get("FTG_OPENCLAW_PERSISTENT_SESSION_KEY", "").strip()

    persistent_factory = None
    if persistent_session_key:
        persistent_factory = lambda _session_id, key=persistent_session_key: key

    if mode == "provider":
        return FunctionBackend(base_llm)

    question_backend = OpenClawBackend(
        session_prefix="ftg",
        fallback_llm=base_llm,
        persistent_session_key_factory=persistent_factory,
    )
    if mode == "openclaw":
        return question_backend

    return CompositeBackend(
        default_backend=FunctionBackend(base_llm),
        question_backend=question_backend,
    )


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

def cmd_start(question: str):
    session_id = uuid.uuid4().hex[:12]
    backend = _make_ftg_backend()
    try:
        fathom = Fathom(backend=backend)
    except TypeError:
        fathom = Fathom(llm_fn=_make_routing_llm_fn())

    session = fathom.start(user_input=question, dialogue_fn=lambda q, i=None: "")

    meta = {
        "question": question,
        "phase": "questioning",
        "created_at": time.time(),
        "last_active": time.time(),
    }

    result = session.step()
    if result is None:
        meta["phase"] = "confirming"
        summary = session._generate_confirmation_summary()
        _save_session(session_id, session, meta)
        print(json.dumps({
            "session_id": session_id,
            "is_fathomed": True,
            "needs_confirmation": True,
            "confirmation_summary": summary,
            "fathom_score": round(session.fathom_score * 100),
        }, ensure_ascii=False))
        return

    q_text, insight = result
    _save_session(session_id, session, meta)
    print(json.dumps({
        "session_id": session_id,
        "question": q_text,
        "insight": insight,
        "fathom_score": round(session.fathom_score * 100),
        "is_fathomed": False,
        "needs_confirmation": False,
        "round": session._round,
    }, ensure_ascii=False))


def cmd_answer(session_id: str, text: str):
    session, meta = _load_session(session_id)
    if meta["phase"] != "questioning":
        print(json.dumps({"error": f"Session is in '{meta['phase']}' phase, not questioning"}))
        sys.exit(1)

    session.answer(text)
    _touch_meta(meta)

    if session.is_fathomed:
        meta["phase"] = "confirming"
        summary = session._generate_confirmation_summary()
        _save_session(session_id, session, meta)
        print(json.dumps({
            "is_fathomed": True,
            "needs_confirmation": True,
            "confirmation_summary": summary,
            "fathom_score": round(session.fathom_score * 100),
            "round": session._round,
        }, ensure_ascii=False))
        return

    result = session.step()
    if result is None:
        meta["phase"] = "confirming"
        summary = session._generate_confirmation_summary()
        _save_session(session_id, session, meta)
        print(json.dumps({
            "is_fathomed": True,
            "needs_confirmation": True,
            "confirmation_summary": summary,
            "fathom_score": round(session.fathom_score * 100),
            "round": session._round,
        }, ensure_ascii=False))
        return

    q_text, insight = result
    _save_session(session_id, session, meta)
    print(json.dumps({
        "question": q_text,
        "insight": insight,
        "fathom_score": round(session.fathom_score * 100),
        "is_fathomed": False,
        "needs_confirmation": False,
        "round": session._round,
    }, ensure_ascii=False))


def cmd_confirm(session_id: str, feedback: str):
    session, meta = _load_session(session_id)
    if meta["phase"] != "confirming":
        print(json.dumps({"error": f"Session is in '{meta['phase']}' phase, not confirming"}))
        sys.exit(1)

    affirmative = {"yes", "y", "ok", "okay", "correct", "confirmed",
                   "good", "fine", "right", "yep", "yeah", "looks good",
                   "go ahead", "proceed", "do it", "sure", "absolutely"}
    if feedback.strip().lower() not in affirmative:
        session._extract_and_update(feedback)

    _do_compile(session_id, session, meta)


def cmd_compile(session_id: str):
    session, meta = _load_session(session_id)
    _do_compile(session_id, session, meta)


def cmd_fathom(session_id: str):
    session, meta = _load_session(session_id)
    response = session.relay("fathom")
    meta["phase"] = _phase_for_relay_action(getattr(response, "action", None), meta.get("phase"))
    _touch_meta(meta)
    _save_session(session_id, session, meta)

    output = {
        "action": response.action,
        "display": response.display,
        "session_id": session_id,
        "fathom_score": response.fathom_score,
    }
    if response.compiled_prompt:
        output["compiled_prompt"] = response.compiled_prompt
    if response.task_type:
        output["task_type"] = response.task_type
    print(json.dumps(output, ensure_ascii=False))


def cmd_stop(session_id: str):
    session, meta = _load_session(session_id)
    response = session.relay("stop")
    meta["phase"] = _phase_for_relay_action(getattr(response, "action", None), meta.get("phase"))
    _touch_meta(meta)
    _save_session(session_id, session, meta)

    output = {
        "action": response.action,
        "display": response.display,
        "session_id": session_id,
        "fathom_score": response.fathom_score,
    }
    if response.task_type:
        output["task_type"] = response.task_type
    print(json.dumps(output, ensure_ascii=False))


def _do_compile(session_id: str, session: FathomSession, meta: dict):
    result = session.compile()
    meta["phase"] = "compiled"
    _save_session(session_id, session, meta)
    print(json.dumps({
        "compiled_prompt": result.compiled_prompt,
        "fathom_score": round(result.fathom_score * 100),
        "fathom_type": result.fathom_type,
        "task_type": result.task_type,
        "rounds": result.rounds,
        "bias_flags": result.bias_flags,
        "dimensions": result.dimensions,
    }, ensure_ascii=False))


def cmd_status(session_id: str):
    path = _session_path(session_id)
    if not path.exists():
        print(json.dumps({"error": f"Session {session_id} not found"}))
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["meta"]
    state = data["session_state"]
    print(json.dumps({
        "session_id": session_id,
        "question": meta["question"],
        "phase": meta.get("phase", "questioning"),
        "round": state["round"],
        "fathom_score": round(state["fathom_score"] * 100),
        "task_type": state["task_type"],
        "is_fathomed": state["is_fathomed"],
    }, ensure_ascii=False))


def cmd_list():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []
    now = time.time()
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            meta = data["meta"]
            if now - meta.get("last_active", 0) > SESSION_TTL:
                f.unlink(missing_ok=True)
                continue
            sessions.append({
                "session_id": f.stem,
                "question": meta["question"][:60],
                "phase": meta.get("phase", "questioning"),
                "round": data["session_state"]["round"],
            })
        except Exception:
            continue
    print(json.dumps(sessions, ensure_ascii=False))


# ------------------------------------------------------------------
# Relay command (recommended for agent integration)
# ------------------------------------------------------------------

def cmd_relay(session_id: str | None, user_message: str):
    backend = _make_ftg_backend()
    relay_input = user_message

    if session_id and _session_path(session_id).exists():
        session, meta = _load_session(session_id)
    else:
        session_id = uuid.uuid4().hex[:12]
        try:
            fathom = Fathom(backend=backend)
        except TypeError:
            fathom = Fathom(llm_fn=_make_routing_llm_fn())
        session = fathom.start(
            user_input=user_message,
            dialogue_fn=lambda q, i=None: "",
        )
        meta = {
            "question": user_message,
            "phase": "questioning",
            "created_at": time.time(),
            "last_active": time.time(),
        }
        # The initial user message is already stored in session._user_input.
        # Passing it again to relay() would double-feed the first turn.
        relay_input = ""

    response = session.relay(relay_input)
    meta["phase"] = _phase_for_relay_action(getattr(response, "action", None), meta.get("phase"))
    _touch_meta(meta)
    _save_session(session_id, session, meta)

    output = {
        "action": response.action,
        "display": response.display,
        "session_id": session_id,
        "fathom_score": response.fathom_score,
    }
    if response.compiled_prompt:
        output["compiled_prompt"] = response.compiled_prompt
    if response.task_type:
        output["task_type"] = response.task_type
    print(json.dumps(output, ensure_ascii=False))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def cmd_install_openclaw():
    """Install Fathom Mode skill for OpenClaw."""
    import shutil
    skill_src = Path(__file__).parent / "data" / "SKILL.md"
    if not skill_src.exists():
        print("Error: SKILL.md not found in package. Reinstall fathom-mode.")
        sys.exit(1)
    target_dir = Path.home() / ".openclaw" / "skills" / "fathom_mode"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(skill_src, target_dir / "SKILL.md")
    print("Fathom Mode skill installed for OpenClaw.")
    print("")
    print("Next steps:")
    print("  1. Start a new conversation in your OpenClaw client")
    print("  2. Say: fathom mode")


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: cli.py <command> [args...]"}))
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "start":
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Usage: cli.py start <question>"}))
            sys.exit(1)
        cmd_start(sys.argv[2])

    elif cmd == "answer":
        if len(sys.argv) < 4:
            print(json.dumps({"error": "Usage: cli.py answer <session_id> <text>"}))
            sys.exit(1)
        cmd_answer(_clean_session_id(sys.argv[2]), sys.argv[3])

    elif cmd == "confirm":
        if len(sys.argv) < 4:
            print(json.dumps({"error": "Usage: cli.py confirm <session_id> <feedback>"}))
            sys.exit(1)
        cmd_confirm(_clean_session_id(sys.argv[2]), sys.argv[3])

    elif cmd == "compile":
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Usage: cli.py compile <session_id>"}))
            sys.exit(1)
        cmd_compile(_clean_session_id(sys.argv[2]))

    elif cmd == "fathom":
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Usage: cli.py fathom <session_id>"}))
            sys.exit(1)
        cmd_fathom(_clean_session_id(sys.argv[2]))

    elif cmd == "stop":
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Usage: cli.py stop <session_id>"}))
            sys.exit(1)
        cmd_stop(_clean_session_id(sys.argv[2]))

    elif cmd == "status":
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Usage: cli.py status <session_id>"}))
            sys.exit(1)
        cmd_status(_clean_session_id(sys.argv[2]))

    elif cmd == "relay":
        session_id = None
        args = sys.argv[2:]
        if "--session" in args:
            idx = args.index("--session")
            if idx + 1 < len(args):
                session_id = _clean_session_id(args[idx + 1])
                args = args[:idx] + args[idx + 2:]
            else:
                print(json.dumps({"action": "error", "display": "--session requires a value"}))
                sys.exit(1)
        if args:
            message = " ".join(args)
        elif not sys.stdin.isatty():
            message = sys.stdin.read().strip()
        else:
            print(json.dumps({"action": "error", "display": "No message provided"}))
            sys.exit(1)
        cmd_relay(session_id, message)

    elif cmd == "list":
        cmd_list()

    elif cmd == "install-openclaw":
        cmd_install_openclaw()

    else:
        print(json.dumps({"error": f"Unknown command: {cmd}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
