# Voice Router for Claude Code

Route voice commands to named Claude Code sessions running in iTerm2 tabs. Hold a key, speak a command, release — your words land in the right session.

## Architecture

```
[Push-to-Talk Hotkey]  --or--  [CLI]  --or--  [Phone HTTP POST]
        |                        |                    |
        v                        v                    v
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
# Install (requires Python 3.10+, Homebrew, iTerm2)
git clone https://github.com/SethDrew/voice-claude.git
cd voice-claude
./setup.sh
source ~/.zshrc

# Install Hammerspoon for push-to-talk hotkey
brew install --cask hammerspoon

# Start the listen daemon (keeps Whisper model warm for instant transcription)
voice-listen-daemon &

# Launch named Claude Code sessions in iTerm2 tabs
cc firmware    # tab 1
cc frontend    # tab 2
cc backend     # tab 3

# Route commands
voice-route --text "tell firmware: check GPIO pins"
voice-route --text "go to frontend"
voice-route --list
```

## Input Methods

### 1. Push-to-Talk (Hold Option key)

Requires Hammerspoon and the listen daemon running in the background.

1. **Hold Option** — mic opens instantly, "Listening..." alert appears
2. **Speak** — "tell firmware check the GPIO pins"
3. **Release Option** — recording stops, Whisper transcribes, command routes to target session

The listen daemon (`voice-listen-daemon`) keeps the Whisper model pre-loaded in memory, eliminating cold-start latency. Start it once and leave it running.

### 2. CLI

```bash
# Record + transcribe + route (standalone, no daemon needed)
voice-route --hotkey

# Route pre-transcribed text
voice-route --text "tell firmware: check GPIO"

# List available sessions
voice-route --list
```

### 3. Phone Server

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

## Named Sessions

The `cc` command launches Claude Code with an iTerm2 user variable (`cc_name`) that the router uses to find sessions:

```bash
cc firmware       # sets cc_name=firmware
cc frontend       # sets cc_name=frontend
cc my-project     # sets cc_name=my-project
```

**Important:** Add `DISABLE_AUTO_TITLE="true"` to `~/.zshrc` (setup.sh does this) to prevent oh-my-zsh from overwriting tab titles.

## File Layout

```
src/
  main.py              # CLI entry point (voice-route)
  router.py            # iTerm2 session discovery + routing
  parser.py            # Natural language command parsing
  daemon.py            # Wake word detection daemon
  listen_daemon.py     # Warm listen daemon (pre-loaded Whisper)
  phone_server.py      # Flask REST API
listen/
  listen.py            # Whisper-based STT with waveform meter
  config.py            # Audio/model configuration
bin/
  cc                   # Named session launcher
  voice-route          # CLI entry point
  voice-listen-daemon  # Warm daemon launcher
  voice-daemon         # Wake word daemon launcher
  voice-phone-server   # Phone server launcher
config/
  hammerspoon/         # Push-to-talk hotkey config
  launchd/             # Auto-start plist
```

After install:
```
~/.local/share/voice-router/    # Runtime: Python modules + venv + daemon state
~/.local/share/listen/          # STT: Whisper-based transcription + venv
~/.local/bin/                   # Launcher scripts (on PATH)
~/.hammerspoon/init.lua         # Hotkey config (appended by setup.sh)
```

## Requirements

- macOS with iTerm2
- Python 3.10+
- Homebrew (for portaudio, ffmpeg)
- ffmpeg (for Whisper audio decoding)
- Claude Code CLI
- Hammerspoon (for push-to-talk hotkey)
- Microphone access granted to iTerm2
- iTerm2 Python API enabled (Preferences > General > Magic)

## Future Work

- **Wake word detection**: Always-listening mode using OpenWakeWord with custom ONNX models. The daemon infrastructure (`src/daemon.py`) is in place but requires training wake word models (e.g., "hey skynet") via [OpenWakeWord](https://github.com/dscripka/openWakeWord) Colab notebooks and placing the `.onnx` files in `models/`.
- **Faster STT**: faster-whisper support exists but needs testing with the daemon workflow.

## License

MIT
