#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.local/share/voice-claude"
BIN_DIR="$HOME/.local/bin"

RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[0;33m'
RST='\033[0m'

info()  { echo -e "${GRN}[✓]${RST} $*"; }
warn()  { echo -e "${YEL}[!]${RST} $*"; }
error() { echo -e "${RED}[✗]${RST} $*"; exit 1; }

echo "=== Voice Claude Setup ==="
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

# ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
    warn "Installing ffmpeg (required by Whisper for audio decoding)..."
    brew install ffmpeg
fi
info "ffmpeg"

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

# --- Single unified venv ---
info "Setting up voice-claude environment..."
mkdir -p "$INSTALL_DIR"

if [[ ! -d "$INSTALL_DIR/venv" ]]; then
    python3 -m venv "$INSTALL_DIR/venv"
    info "Created venv"
else
    info "Venv exists"
fi
source "$INSTALL_DIR/venv/bin/activate"

# Apple Silicon detection for MLX
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    info "Apple Silicon detected — MLX Whisper will be installed"
fi

pip install -q -r "$SCRIPT_DIR/requirements.txt"
info "Core dependencies installed"

# Optional: mlx-lm for LLM-based voice routing
if [[ "$ARCH" == "arm64" ]]; then
    echo ""
    read -p "Install mlx-lm for LLM-based voice routing? (recommended, ~200MB) [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -q "mlx-lm>=0.20.0" || warn "mlx-lm install failed (optional)"
        info "mlx-lm installed"
    else
        info "Skipped mlx-lm (can install later: pip install mlx-lm)"
    fi
fi

deactivate
info "All dependencies installed"

# --- Copy source files ---
echo ""
info "Installing source files..."
cp "$SCRIPT_DIR"/src/*.py "$INSTALL_DIR/"
cp "$SCRIPT_DIR"/listen/*.py "$INSTALL_DIR/"
info "Sources → $INSTALL_DIR/"

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

# --- launchd plist ---
echo ""
PLIST_SRC="$SCRIPT_DIR/config/launchd/com.user.voice-listen-daemon.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.user.voice-listen-daemon.plist"
if [[ -f "$PLIST_SRC" ]]; then
    mkdir -p "$HOME/Library/LaunchAgents"
    sed "s|__HOME__|$HOME|g" "$PLIST_SRC" > "$PLIST_DST"
    info "Installed launchd plist → $PLIST_DST"
    info "  Start daemon: launchctl load $PLIST_DST"
    info "  Stop daemon:  launchctl unload $PLIST_DST"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Manual steps:"
echo "  1. Enable iTerm2 Python API:"
echo "     iTerm2 → Preferences → General → Magic → Enable Python API"
echo "  2. Grant microphone access to Terminal/iTerm2"
echo "  3. Install Hammerspoon (for push-to-talk hotkey):"
echo "     brew install --cask hammerspoon"
echo "  4. Reload zsh: source ~/.zshrc"
echo ""
echo "Quick start:"
echo "  cc firmware       # Launch a named Claude Code session"
echo "  cc frontend       # Another named session"
echo "  voice-route --list              # See sessions"
echo "  voice-route --text 'tell firmware: check GPIO'  # Route text"
echo "  voice-route --hotkey            # Record + transcribe + route"
