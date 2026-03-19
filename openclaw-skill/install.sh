#!/bin/bash
# Install the Fathom Mode skill for OpenClaw.
#
# What this script does:
#   1. Installs fathom-mode Python package (via uv or pip3)
#   2. Ensures the `fathom` command is on a system-wide PATH
#   3. Copies SKILL.md to ~/.openclaw/skills/fathom_mode/
#   4. Verifies the fathom command works
#
# Usage:
#   ./openclaw-skill/install.sh          # normal install (from PyPI)
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
    echo "[1/4] Installing fathom-mode (development mode)..."
    if command -v pip3 &>/dev/null; then
        pip3 install -e "$PROJECT_DIR[gemini,openai,anthropic,openclaw]"
    elif command -v pip &>/dev/null; then
        pip install -e "$PROJECT_DIR[gemini,openai,anthropic,openclaw]"
    else
        echo "Error: pip3 not found. Install Python first."
        exit 1
    fi
else
    echo "[1/4] Installing fathom-mode..."
    if command -v uv &>/dev/null; then
        uv tool install "fathom-mode[openclaw]" 2>/dev/null || uv tool upgrade "fathom-mode[openclaw]" 2>/dev/null || true
    elif command -v pip3 &>/dev/null; then
        pip3 install --upgrade "fathom-mode[openclaw]"
    else
        echo "Error: Neither uv nor pip3 found. Install Python first."
        exit 1
    fi
fi

# --- Step 2: Ensure `fathom` is on a system-wide PATH ---

echo "[2/4] Checking fathom command..."

# Find the fathom binary wherever it was installed
FATHOM_BIN=""
if command -v fathom &>/dev/null; then
    FATHOM_BIN="$(command -v fathom)"
else
    # Common locations where pip/uv install scripts on macOS
    for candidate in \
        "$HOME/.local/bin/fathom" \
        "$HOME/Library/Python"/*/bin/fathom \
        /opt/homebrew/bin/fathom \
        /usr/local/bin/fathom; do
        if [ -f "$candidate" ]; then
            FATHOM_BIN="$candidate"
            break
        fi
    done
fi

if [ -z "$FATHOM_BIN" ]; then
    echo "Error: Could not find the fathom binary after installation."
    echo "Try: pip3 install fathom-mode"
    exit 1
fi

# Check if fathom is already in a standard PATH directory that the gateway can see
STANDARD_PATHS="/opt/homebrew/bin /usr/local/bin /usr/bin"
FATHOM_DIR="$(dirname "$FATHOM_BIN")"
IN_STANDARD_PATH=false

for sp in $STANDARD_PATHS; do
    if [ "$FATHOM_DIR" = "$sp" ]; then
        IN_STANDARD_PATH=true
        break
    fi
done

if ! $IN_STANDARD_PATH; then
    # fathom exists but in a non-standard location — create symlink
    SYMLINK_TARGET=""
    if [ -d "/opt/homebrew/bin" ]; then
        SYMLINK_TARGET="/opt/homebrew/bin/fathom"
    elif [ -d "/usr/local/bin" ]; then
        SYMLINK_TARGET="/usr/local/bin/fathom"
    fi

    if [ -n "$SYMLINK_TARGET" ]; then
        echo "  fathom found at: $FATHOM_BIN (not in standard PATH)"
        echo "  Creating symlink: $SYMLINK_TARGET -> $FATHOM_BIN"
        ln -sf "$FATHOM_BIN" "$SYMLINK_TARGET"
        echo "  Done."
    else
        echo "Warning: fathom is installed at $FATHOM_BIN but not in a standard PATH."
        echo "OpenClaw may not find it. Add its directory to your PATH:"
        echo "  export PATH=\"$FATHOM_DIR:\$PATH\""
    fi
else
    echo "  fathom found at: $FATHOM_BIN"
fi

# --- Step 3: Install SKILL.md to ~/.openclaw/skills/ ---

echo "[3/4] Installing OpenClaw skill..."

TARGET_DIR="$HOME/.openclaw/skills/fathom_mode"
mkdir -p "$TARGET_DIR"
cp "$SKILL_FILE" "$TARGET_DIR/SKILL.md"

# --- Step 4: Verify ---

echo "[4/4] Verifying..."
if python3 -m ftg relay --help >/dev/null 2>&1; then
    echo "  fathom-mode works."
else
    echo "Warning: fathom-mode may not be installed correctly."
    echo "  Try running: python3 -m ftg relay --help"
fi

echo ""
echo "Done! Fathom Mode skill installed."
echo ""
echo "Next steps:"
echo "  1. Start a new conversation in your OpenClaw client"
echo "  2. Say: fathom mode"
