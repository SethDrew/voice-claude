# Voice Router for Claude Code

Route voice commands to named Claude Code sessions running in iTerm2 tabs.

## Architecture

```
[Wake Word Daemon]  --or--  [Hammerspoon Hotkey]  --or--  [Phone HTTP POST]
        |                          |                            |
        v                          v                            v
  +--------------------------------------------------------------+
  |                    Command Parser                             |
  |  "tell firmware: check GPIO" -> target=firmware, text="..."  |
  +--------------------------------------------------------------+
                            |
                            v
  +--------------------------------------------------------------+
  |                    iTerm2 Router                              |
  |  Find session by cc_name user variable -> async_send_text()  |
  +--------------------------------------------------------------+
                            |
                            v
  Named iTerm2 tabs launched via `cc <name>` wrapper
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/SethDrew/voice-claude.git
cd voice-claude
./setup.sh

# Source updated PATH
source ~/.zshrc

# Launch named Claude Code sessions
cc firmware    # tab 1
cc frontend    # tab 2
cc backend     # tab 3

# Route commands
voice-route --text "tell firmware: check GPIO pins"
voice-route --text "go to frontend"
voice-route --list
```

## Input Methods

### 1. Hotkey (Double-tap Fn)

Requires Hammerspoon. Double-tap the Fn key to:
1. Record speech (auto-stops after silence)
2. Transcribe with Whisper
3. Parse target session and command
4. Route to the correct iTerm2 tab

### 2. CLI

```bash
# Record + transcribe + route
voice-route --hotkey

# Route pre-transcribed text
voice-route --text "tell firmware: check GPIO"

# List available sessions
voice-route --list
```

### 3. Wake Word Daemon

Listens continuously for wake words ("hey skynet", "hey destroyer", "hey code").

```bash
# Start daemon
voice-daemon

# Or install as launchd service (edit plist paths first)
cp config/launchd/com.user.voice-router.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.user.voice-router.plist
```

### 4. Phone Server

HTTP endpoint for routing from a phone or other device.

```bash
voice-phone-server  # starts on port 7890

# From phone/curl:
curl -X POST http://<your-ip>:7890/voice \
  -H 'Content-Type: application/json' \
  -d '{"text": "tell firmware: check GPIO"}'

# List sessions
curl http://<your-ip>:7890/sessions
```

## Command Grammar

| Voice Input | Parsed As |
|---|---|
| `tell firmware: check GPIO` | target=firmware, text="check GPIO" |
| `tell firmware check GPIO` | target=firmware, text="check GPIO" |
| `go to frontend` | target=frontend (focus only) |
| `check the logs` | target=last-active, text="check the logs" |
| `slash commit` | target=last-active, text="/commit" |
| `hey skynet, tell firmware: check GPIO` | (wake phrase stripped first) |

## Named Sessions

The `cc` command launches Claude Code with an iTerm2 user variable (`cc_name`) that the router uses to find sessions:

```bash
cc firmware       # sets cc_name=firmware
cc frontend       # sets cc_name=frontend
cc my-project     # sets cc_name=my-project
```

**Important:** Add `DISABLE_AUTO_TITLE="true"` to `~/.zshrc` (setup.sh does this) to prevent oh-my-zsh from overwriting tab titles.

## Wake Word Training

1. Use [OpenWakeWord](https://github.com/dscripka/openWakeWord) training notebooks on Google Colab
2. Train models for your wake phrases (e.g., "hey skynet")
3. Export as `.onnx` files
4. Place in `models/` directory (or `~/.local/share/voice-router/models/` after install)

## File Layout

```
~/.local/share/voice-router/    # Runtime: Python modules + venv
~/.local/share/listen/          # STT: Whisper-based transcription
~/.local/bin/                   # Launcher scripts (on PATH)
~/.hammerspoon/init.lua         # Hotkey config (appended)
```

## Requirements

- macOS with iTerm2
- Python 3.10+
- Homebrew (for portaudio)
- Claude Code CLI
- Hammerspoon (optional, for hotkey)
- Microphone access

## License

MIT
