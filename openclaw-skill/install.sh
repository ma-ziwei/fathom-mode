#!/bin/bash
# Install the Fathom Mode skill for OpenClaw.
#
# What this script does:
#   1. Installs fathom-mode Python package (via uv or pip3)
#   2. Copies SKILL.md to ~/.openclaw/skills/fathom_mode/
#   3. Verifies fathom-mode works
#
# Usage:
#   ./openclaw-skill/install.sh          # normal install
#   ./openclaw-skill/install.sh --dev    # editable install (for development)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILL_FILE="$SCRIPT_DIR/SKILL.md"
DEV_MODE=false

if [ "$1" = "--dev" ]; then
    DEV_MODE=true
fi

echo "Installing Fathom Mode for OpenClaw..."
echo ""

# --- Step 1: Install the Python package ---

if $DEV_MODE; then
    echo "[1/3] Installing fathom-mode (development mode)..."
    if command -v pip3 &>/dev/null; then
        pip3 install -e "$PROJECT_DIR[gemini,openai,anthropic,openclaw]"
    elif command -v pip &>/dev/null; then
        pip install -e "$PROJECT_DIR[gemini,openai,anthropic,openclaw]"
    else
        echo "Error: pip not found. Install Python first."
        exit 1
    fi
else
    echo "[1/3] Installing fathom-mode..."
    if command -v pip3 &>/dev/null; then
        pip3 install "fathom-mode[openclaw] @ git+https://github.com/ma-ziwei/fathom-mode.git"
    elif command -v pip &>/dev/null; then
        pip install "fathom-mode[openclaw] @ git+https://github.com/ma-ziwei/fathom-mode.git"
    else
        echo "Error: pip not found. Install Python first."
        exit 1
    fi
fi

# --- Step 2: Install SKILL.md to ~/.openclaw/skills/ ---

echo "[2/3] Installing OpenClaw skill..."

TARGET_DIR="$HOME/.openclaw/skills/fathom_mode"
mkdir -p "$TARGET_DIR"
cp "$SKILL_FILE" "$TARGET_DIR/SKILL.md"

# --- Step 3: Verify ---

echo "[3/3] Verifying..."
if python3 -m ftg list >/dev/null 2>&1; then
    echo "  fathom-mode works."
else
    echo "Warning: fathom-mode may not be installed correctly."
    echo "  Try running: python3 -m ftg list"
fi

echo ""
echo "Done! Fathom Mode skill installed."
echo ""
echo "Next steps:"
echo "  1. Start a new conversation in your OpenClaw client"
echo "  2. Say: fathom mode"
