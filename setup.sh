#!/usr/bin/env bash
#
# OpenReel - Guided Setup
# Run: bash setup.sh
#
# This script walks you through installing prerequisites and configuring
# OpenReel so it's ready to process your first video.
#
set -eo pipefail

# -- Colors -------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { printf "${BLUE}[info]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[ok]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[warn]${NC}  %s\n" "$*"; }
fail()  { printf "${RED}[error]${NC} %s\n" "$*"; }

# -- Locate repo root ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "       OpenReel - Guided Setup            "
echo "=========================================="
echo ""

# ── Step 1: Check Python ───────────────────────────────────────────────────
info "Checking Python..."
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
        ok "Python $PY_VERSION"
    else
        fail "Python $PY_VERSION found, but 3.11+ is required."
        echo ""
        echo "  Install Python 3.12:"
        echo "    macOS:        brew install python@3.12"
        echo "    Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
        echo "    Windows:      winget install Python.Python.3.12"
        echo ""
        exit 1
    fi
else
    fail "Python not found."
    echo ""
    echo "  Install Python 3.12:"
    echo "    macOS:        brew install python@3.12"
    echo "    Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
    echo "    Windows:      winget install Python.Python.3.12"
    echo ""
    exit 1
fi

# ── Step 2: Check ffmpeg ──────────────────────────────────────────────────
info "Checking ffmpeg..."
if command -v ffmpeg &>/dev/null; then
    FF_VERSION=$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')
    ok "ffmpeg $FF_VERSION"
else
    fail "ffmpeg not found."
    echo ""
    echo "  Install ffmpeg:"
    echo "    macOS:        brew install ffmpeg"
    echo "    Ubuntu/Debian: sudo apt install ffmpeg"
    echo "    Windows:      winget install ffmpeg"
    echo ""
    exit 1
fi

info "Checking ffprobe..."
if command -v ffprobe &>/dev/null; then
    ok "ffprobe found"
else
    fail "ffprobe not found (usually ships with ffmpeg)."
    exit 1
fi

# ── Step 3: Create virtual environment ────────────────────────────────────
info "Setting up Python virtual environment..."
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    ok "Virtual environment already exists at .venv"
else
    python3 -m venv "$VENV_DIR"
    ok "Created virtual environment at .venv"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Activated virtual environment"

# ── Step 4: Install OpenReel ──────────────────────────────────────────────
echo ""
printf "Install speaker diarization? (identifies who is talking)\n"
printf "  This is a large download (~2 GB) and requires a HuggingFace token.\n"
printf "  You can always install it later with: pip install -e \".[diarize]\"\n"
echo ""
printf "Install diarization? [y/N]: "
read -r INSTALL_DIARIZE

if [ "$INSTALL_DIARIZE" = "y" ] || [ "$INSTALL_DIARIZE" = "Y" ]; then
    info "Installing OpenReel (with diarization)..."
    pip install -e "$SCRIPT_DIR[diarize]" --quiet 2>&1 | tail -1
    ok "OpenReel installed (with diarization)"
else
    info "Skipping diarization"
    info "Installing OpenReel..."
    pip install -e "$SCRIPT_DIR" --quiet 2>&1 | tail -1
    ok "OpenReel installed (for auto-captions later: pip install -e \".[captions]\")"
fi

# Verify CLI
if command -v openreel &>/dev/null; then
    ok "CLI available: $(which openreel)"
else
    fail "openreel CLI not found after install. Make sure the venv is active."
    exit 1
fi

# ── Step 5: Configure API key ────────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/.env"

if [ -f "$ENV_FILE" ] && grep -q "GEMINI_API_KEY=" "$ENV_FILE" && ! grep -q "GEMINI_API_KEY=your-api-key-here" "$ENV_FILE"; then
    ok "Gemini API key already configured in .env"
else
    cp "$SCRIPT_DIR/.env.example" "$ENV_FILE" 2>/dev/null || true

    echo ""
    printf "${BOLD}You need a Google Gemini API key (free).${NC}\n"
    echo ""
    echo "  1. Go to: https://aistudio.google.com/apikey"
    echo "  2. Click 'Create API Key'"
    echo "  3. Copy the key (starts with 'AIza...')"
    echo ""
    printf "Paste your Gemini API key (or press Enter to skip): "
    read -r API_KEY

    if [ -n "$API_KEY" ]; then
        if [ -f "$ENV_FILE" ]; then
            # Replace the placeholder line
            sed -i.bak "s|GEMINI_API_KEY=.*|GEMINI_API_KEY=$API_KEY|" "$ENV_FILE"
            rm -f "$ENV_FILE.bak"
        else
            echo "GEMINI_API_KEY=$API_KEY" > "$ENV_FILE"
        fi
        ok "API key saved to .env"
    else
        warn "Skipped — you'll need to add GEMINI_API_KEY to .env before running jobs."
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
printf "${GREEN}${BOLD}Setup complete!${NC}\n"
echo ""
echo "  Start the web UI:"
echo "    bash start.sh"
echo ""
echo "  Or use an AI agent (Claude Code, Codex, etc.) to process videos"
echo "  directly -- just open this folder and ask it to run OpenReel"
echo "  on your video file."
echo ""
