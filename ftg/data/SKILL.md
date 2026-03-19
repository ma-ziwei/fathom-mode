---
name: fathom_mode
description: "Activate Fathom Mode — deeply understand the user's intent before generating any response. Use when the user says 'fathom mode'. Fathom Mode asks targeted questions to build a complete Intent Graph, then compiles a high-quality structured prompt for downstream generation."
homepage: https://github.com/ma-ziwei/fathom-mode
metadata:
  {
    "openclaw":
      {
        "emoji": "🔮",
        "install":
          [
            {
              "id": "uv",
              "kind": "uv",
              "package": "fathom-mode[openclaw]",
              "label": "Install Fathom Mode",
            },
          ],
      },
  }
---

# Fathom Mode

**IMPORTANT: You MUST use the shell tool to run `python3 -m ftg relay` commands. Do NOT simulate, roleplay, or generate responses yourself. All responses come from the `python3 -m ftg relay` CLI — you are only a relay between the user and the fathom process.**

## When to activate

Activate when the user says "fathom mode".

## Step-by-step protocol

### 1. User triggers "fathom mode"

Respond: "Fathom Mode activated. What would you like to explore?"

Wait for the user's next message (their actual request).

### 2. Start a fathom session

Run this shell command with the user's request:

```bash
python3 -m ftg relay "USER'S REQUEST HERE"
```

The command returns JSON. Read the `action` field and follow the rules below.

### 3. Handle the JSON response

| `action` value | What you MUST do |
|---|---|
| `ask_user` | Show the `display` field to the user. Wait for their answer. |
| `review` | Show the `display` field. Ask if they want to continue, edit, or execute. |
| `execute` | The `compiled_prompt` field contains the final structured prompt. Use it for downstream generation. |
| `stop` | Tell the user the session has ended. |

### 4. Continue the session

When the user replies, run:

```bash
python3 -m ftg relay --session SESSION_ID "USER'S REPLY"
```

Use the `session_id` from the previous JSON response. Repeat step 3.

### 5. Special user commands

| User says | What to run |
|---|---|
| "fathom" | `python3 -m ftg relay --session SESSION_ID "fathom"` |
| "execute" | `python3 -m ftg relay --session SESSION_ID "execute"` |
| "stop" | `python3 -m ftg relay --session SESSION_ID "stop"` |

## Rules

1. **NEVER generate fathom questions yourself.** Always run `python3 -m ftg relay` and use its output.
2. **ALWAYS show the `display` field** from the JSON response to the user as-is.
3. **Track the `session_id`** across the conversation. Every continuation must include `--session`.
4. When `action` is `execute`, the `compiled_prompt` is a high-quality structured prompt — use it as input for your next generation task.
5. The `fathom_score` field (0-100%) indicates understanding depth. You may mention it to the user.

## Environment variables

Fathom Mode auto-detects LLM providers from OpenClaw auth profiles. No extra configuration needed.

If needed, these env vars are also supported: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`.
