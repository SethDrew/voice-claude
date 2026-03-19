#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROUTER_INSTALL="$HOME/.local/share/voice-router"
LISTEN_INSTALL="$HOME/.local/share/listen"
BIN_DIR="$HOME/.local/bin"

RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[0;33m'
RST='\033[0m'

info()  { echo -e "${GRN}[✓]${RST} $*"; }
warn()  { echo -e "${YEL}[!]${RST} $*"; }
error() { echo -e "${RED}[✗]${RST} $*"; exit 1; }

echo "=== Voice Router Setup ==="
echo ""

# --- Prerequisites ---
info "Checking prerequisites..."

# Python 3.10+
python3 --version >/dev/null 2>&1 || error "Python 3 not found"
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
[[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 10 ]] || error "Python 3.10+ required (found $PY_VER)"
info "Python $PY_VER"

# Homebrew
command -v brew >/dev/null 2>&1 || error "Homebrew not found — install from https://brew.sh"
info "Homebrew"

# portaudio
if ! brew list portaudio >/dev/null 2>&1; then
    warn "Installing portaudio..."
    brew install portaudio
fi
info "portaudio"

# iTerm2
if [[ -d "/Applications/iTerm.app" ]]; then
    info "iTerm2"
else
    warn "iTerm2 not found — install from https://iterm2.com"
fi

# Claude Code
if command -v claude >/dev/null 2>&1; then
    info "Claude Code"
else
    warn "Claude Code not found on PATH"
fi

echo ""

# --- Voice Router venv ---
info "Setting up voice-router environment..."
mkdir -p "$ROUTER_INSTALL"

if [[ ! -d "$ROUTER_INSTALL/venv" ]]; then
    python3 -m venv "$ROUTER_INSTALL/venv"
    info "Created voice-router venv"
else
    info "Voice-router venv exists"
fi
source "$ROUTER_INSTALL/venv/bin/activate"
pip install -q -r "$SCRIPT_DIR/requirements.txt"
deactivate
info "Voice-router dependencies installed"

# --- Listen venv ---
info "Setting up listen environment..."
mkdir -p "$LISTEN_INSTALL"

if [[ ! -d "$LISTEN_INSTALL/venv" ]]; then
    python3 -m venv "$LISTEN_INSTALL/venv"
    info "Created listen venv"
else
    info "Listen venv exists"
fi
source "$LISTEN_INSTALL/venv/bin/activate"
pip install -q -r "$SCRIPT_DIR/requirements-listen.txt"
deactivate
info "Listen dependencies installed"

# --- Copy source files ---
info "Installing source files..."
cp "$SCRIPT_DIR"/src/*.py "$ROUTER_INSTALL/"
info "Voice-router sources → $ROUTER_INSTALL/"

if [[ ! -f "$LISTEN_INSTALL/listen.py" ]]; then
    cp "$SCRIPT_DIR"/listen/*.py "$LISTEN_INSTALL/"
    info "Listen sources → $LISTEN_INSTALL/"
else
    info "Listen sources already present (skipped)"
fi

# --- Copy models ---
mkdir -p "$ROUTER_INSTALL/models"
if ls "$SCRIPT_DIR"/models/*.onnx >/dev/null 2>&1; then
    cp "$SCRIPT_DIR"/models/*.onnx "$ROUTER_INSTALL/models/"
    info "Wake word models installed"
else
    info "No wake word models found (add .onnx files to models/ later)"
fi

# --- Symlink launchers ---
info "Symlinking launchers..."
mkdir -p "$BIN_DIR"
for script in "$SCRIPT_DIR"/bin/*; do
    name=$(basename "$script")
    chmod +x "$script"
    ln -sf "$script" "$BIN_DIR/$name"
    info "  $name → $BIN_DIR/$name"
done

# --- Hammerspoon ---
HS_DIR="$HOME/.hammerspoon"
HS_INIT="$HS_DIR/init.lua"
HS_MARKER="-- voice-router hotkey"
if [[ -f "$HS_INIT" ]] && grep -q "$HS_MARKER" "$HS_INIT"; then
    info "Hammerspoon config already present"
else
    mkdir -p "$HS_DIR"
    echo "" >> "$HS_INIT"
    cat "$SCRIPT_DIR/config/hammerspoon/init.lua" >> "$HS_INIT"
    info "Hammerspoon config appended to $HS_INIT"
fi

# --- zshrc ---
ZSHRC="$HOME/.zshrc"

if ! grep -q 'DISABLE_AUTO_TITLE' "$ZSHRC" 2>/dev/null; then
    echo 'DISABLE_AUTO_TITLE="true"' >> "$ZSHRC"
    info "Added DISABLE_AUTO_TITLE to .zshrc"
fi

if ! grep -q '$HOME/.local/bin' "$ZSHRC" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$ZSHRC"
    info "Added ~/.local/bin to PATH in .zshrc"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Manual steps:"
echo "  1. Enable iTerm2 Python API:"
echo "     iTerm2 → Preferences → General → Magic → Enable Python API"
echo "  2. Grant microphone access to Terminal/iTerm2"
echo "  3. Install Hammerspoon (for hotkey support):"
echo "     brew install --cask hammerspoon"
echo "  4. Reload zsh: source ~/.zshrc"
echo ""
echo "Quick start:"
echo "  cc firmware       # Launch a named Claude Code session"
echo "  cc frontend       # Another named session"
echo "  voice-route --list              # See sessions"
echo "  voice-route --text 'tell firmware: check GPIO'  # Route text"
echo "  voice-route --hotkey            # Record + transcribe + route"
