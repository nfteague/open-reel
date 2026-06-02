#!/usr/bin/env bash
#
# OpenReel - One-command launcher
# Run: bash start.sh
#
# Sets up everything (if needed) and starts the server.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
ENV_FILE="$SCRIPT_DIR/.env"
NEEDS_SETUP=false

# -- Check prerequisites ------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    fail "Python not found."
    echo ""
    echo "  Install Python 3.12:"
    echo "    macOS:         brew install python@3.12"
    echo "    Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
    echo "    Windows:       winget install Python.Python.3.12"
    echo ""
    exit 1
fi

PY_MINOR=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f2)
if [ "$PY_MINOR" -lt 11 ]; then
    fail "Python 3.11+ is required (found 3.$PY_MINOR)."
    exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
    fail "ffmpeg not found."
    echo ""
    echo "  Install ffmpeg:"
    echo "    macOS:         brew install ffmpeg"
    echo "    Ubuntu/Debian: sudo apt install ffmpeg"
    echo "    Windows:       winget install ffmpeg"
    echo ""
    exit 1
fi

# -- Set up venv if missing ----------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    NEEDS_SETUP=true
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    ok "Created .venv"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# -- Install if missing --------------------------------------------------------
if ! command -v openreel &>/dev/null; then
    NEEDS_SETUP=true
    info "Installing OpenReel..."
    pip install -e "$SCRIPT_DIR[captions]" --quiet 2>&1 | tail -1
    ok "Installed"
fi

# -- Configure API key if missing ---------------------------------------------
has_key=false
if [ -f "$ENV_FILE" ] && grep -q "GEMINI_API_KEY=" "$ENV_FILE" && ! grep -q "GEMINI_API_KEY=your-api-key-here" "$ENV_FILE"; then
    has_key=true
fi

if [ "$has_key" = false ]; then
    NEEDS_SETUP=true
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
            sed -i.bak "s|GEMINI_API_KEY=.*|GEMINI_API_KEY=$API_KEY|" "$ENV_FILE"
            rm -f "$ENV_FILE.bak"
        else
            echo "GEMINI_API_KEY=$API_KEY" > "$ENV_FILE"
        fi
        ok "API key saved to .env"
    else
        warn "No API key set -- jobs will fail until you add GEMINI_API_KEY to .env"
    fi
fi

# -- Open browser when server is ready -----------------------------------------
open_browser() {
    local url="$1"
    if command -v open &>/dev/null; then
        open "$url"                    # macOS
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$url"               # Linux
    elif command -v wslview &>/dev/null; then
        wslview "$url"                 # WSL
    fi
}

wait_and_open() {
    for _ in $(seq 1 30); do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            open_browser "http://localhost:8000"
            return
        fi
        sleep 0.5
    done
    warn "Server didn't respond in time -- open http://localhost:8000 manually"
}

# -- Launch server -------------------------------------------------------------
if [ "$NEEDS_SETUP" = true ]; then
    echo ""
    ok "Setup complete"
fi

echo ""
info "Starting OpenReel server on http://localhost:8000"
info "Press Ctrl+C to stop"
echo ""

wait_and_open &
exec openreel serve "$@"
