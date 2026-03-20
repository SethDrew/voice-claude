# Voice Router for Claude Code

Route voice commands to named Claude Code sessions running in iTerm2 tabs. Hold a key, speak a command, release — your words land in the right session.

## Architecture

```
[Push-to-Talk Hotkey]  --or--  [CLI]  --or--  [Phone HTTP POST]
        |                        |                    |
        v                        v                    v
  +--------------------------------------------------------------+
  |              Speech-to-Text (MLX Whisper)                     |
  |  large-v3-turbo on Apple Silicon, coding vocabulary primed   |
  +--------------------------------------------------------------+
                            |
                            v
  +--------------------------------------------------------------+
  |         Command Parser (regex + fuzzy + LLM fallback)        |
  |  "tell firmware: check GPIO" -> target=firmware, text="..."  |
  +--------------------------------------------------------------+
                            |
                            v
  +--------------------------------------------------------------+
  |                    iTerm2 Router                              |
  |  Fuzzy session matching via rapidfuzz edit distance           |
  +--------------------------------------------------------------+
                            |
                            v
  Named Claude Code sessions (auto-named via claude -p)
```

## Quick Start

```bash
# Install (requires Python 3.10+, Homebrew, iTerm2, Apple Silicon Mac)
git clone https://github.com/SethDrew/voice-claude.git
cd voice-claude
./setup.sh
source ~/.zshrc

# Install Hammerspoon for push-to-talk hotkey
brew install --cask hammerspoon

# Start the listen daemon (first run downloads Whisper model ~1.5GB)
voice-listen-daemon &

# Open Claude Code sessions — they auto-name after your first message
claude    # tab 1 — auto-named by claude -p (e.g., "spark")
claude    # tab 2 — auto-named (e.g., "anvil")

# Or explicitly name them
cc firmware    # tab with cc_name=firmware

# Route commands
voice-route --text "tell spark: check the GPIO pins"
voice-route --text "go to anvil"
voice-route --list
```

## Input Methods

### 1. Push-to-Talk (Hold Right Option key)

Requires Hammerspoon and the listen daemon running in the background.

1. **Hold Right Option** — mic opens instantly
2. **Speak** — "tell firmware check the GPIO pins"
3. **Release Right Option** — recording stops, Whisper transcribes, command routes to target session

The listen daemon keeps the Whisper large-v3-turbo model pre-loaded in memory via MLX (Apple Silicon GPU acceleration). Transcription takes ~0.3-1s for typical voice commands.

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

```bash
voice-phone-server  # starts on port 7890

curl -X POST http://<your-ip>:7890/voice \
  -H 'Content-Type: application/json' \
  -d '{"text": "tell firmware: check GPIO"}'
```

## Command Grammar

The parser supports multiple natural patterns. You don't need to memorize a specific syntax:

| Voice Input | Parsed As |
|---|---|
| `tell firmware: check GPIO` | target=firmware, text="check GPIO" |
| `tell firmware check GPIO` | target=firmware, text="check GPIO" |
| `ask spark run the tests` | target=spark, text="run the tests" |
| `hey anvil check the logs` | target=anvil, text="check the logs" |
| `for firmware, check GPIO` | target=firmware, text="check GPIO" |
| `firmware check GPIO` | target=firmware (target-first, if known session) |
| `go to frontend` | target=frontend (focus only) |
| `switch to spark` | target=spark (focus only) |
| `check the logs` | target=last-active, text="check the logs" |
| `slash commit` | target=last-active, text="/commit" |

Routing verbs: `tell`, `ask`, `send`, `message`, `ping`, `hey`, `yo`, `talk to`, `for`

Focus verbs: `go to`, `switch to`, `focus`

### Fuzzy Name Matching

The router uses [rapidfuzz](https://github.com/rapidfuzz/rapidfuzz) edit-distance matching, so Whisper mishearings are handled gracefully:

- "built-up" matches session "built-app"
- "firm wear" matches session "firmware" (words are joined automatically)
- "front end" matches session "frontend"

## Session Naming

Sessions are automatically named when you use Claude Code. A `Stop` hook calls `claude -p` to generate a short, unique codename (like "spark", "anvil", "drift") after your first message. The name appears in the iTerm2 tab title and is used for voice routing.

You can also name sessions explicitly:

```bash
cc firmware       # sets cc_name=firmware via the cc wrapper
claude -n spark   # Claude Code's built-in --name flag
```

**iTerm2 setup for tab titles:** Go to iTerm2 → Settings → Profiles → General → Title and ensure "Session Name" is included in the title format.

## Requirements

### Required

- **macOS with Apple Silicon** (M1/M2/M3/M4/M5 — needed for MLX Whisper GPU acceleration)
- **iTerm2** with Python API enabled (Settings → General → Magic → Enable Python API)
- **Python 3.10+** (recommend 3.13 via `brew install python@3.13`)
- **Homebrew** (for portaudio, ffmpeg)
- **ffmpeg** (`brew install ffmpeg` — Whisper needs it for audio decoding)
- **Claude Code CLI** (for session auto-naming via `claude -p`)
- **Hammerspoon** (`brew install --cask hammerspoon` — for push-to-talk hotkey)
- **Microphone access** granted to iTerm2

### Optional (Enhanced Routing)

- **[Ollama](https://ollama.com/)** — Enables LLM-based intent parsing for natural voice commands that don't match standard patterns. When installed, the parser falls back to a local LLM if regex doesn't match.

```bash
brew install ollama
ollama pull qwen2.5:1.5b
# Start Ollama server (runs in background)
brew services start ollama
```

With Ollama, you can say things like "the firmware session should check the GPIO pins" or "can you have spark look at the error logs" — natural speech that regex can't parse.

### Non-Apple-Silicon Macs

The system falls back to faster-whisper (CPU) on Intel Macs. Transcription will be slower (~2-5s vs ~0.5s) and uses the `base` model instead of `large-v3-turbo`. setup.sh handles this automatically.

## File Layout

```
src/
  main.py              # CLI entry point (voice-route)
  router.py            # iTerm2 session discovery + fuzzy routing
  parser.py            # Command parsing (regex + rapidfuzz + LLM fallback)
  llm_router.py        # Optional Ollama-based intent parsing
  listen_daemon.py     # Warm listen daemon (MLX Whisper large-v3-turbo)
  daemon.py            # Wake word detection daemon
  phone_server.py      # Flask REST API
listen/
  listen.py            # Standalone STT with waveform meter
  config.py            # Audio/model configuration
bin/
  cc                   # Named session launcher
  voice-route          # CLI entry point
  voice-listen-daemon  # Warm daemon launcher
  auto-name-session    # Claude Code Stop hook for session naming
  voice-daemon         # Wake word daemon launcher
  voice-phone-server   # Phone server launcher
config/
  hammerspoon/         # Push-to-talk hotkey config (Right Option key)
  launchd/             # Auto-start plist template
```

After install:
```
~/.local/share/voice-router/    # Runtime: router modules + venv + daemon state
~/.local/share/listen/          # STT: Whisper models + venv
~/.local/bin/                   # Launcher scripts (on PATH)
~/.hammerspoon/init.lua         # Hotkey config (appended by setup.sh)
```

## Claude Code Settings

The auto-naming hook requires these settings in `~/.claude/settings.json`:

```json
{
  "env": {
    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1"
  },
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/.local/bin/auto-name-session",
            "timeout": 20
          }
        ]
      }
    ]
  }
}
```

`CLAUDE_CODE_DISABLE_TERMINAL_TITLE` prevents Claude from overwriting the auto-generated session name on each response. The hook timeout of 20s gives `claude -p` enough time to generate a name.

## Future Work

- **Wake word detection**: Always-listening mode using OpenWakeWord with custom ONNX models. Infrastructure exists in `src/daemon.py` — needs trained `.onnx` models.
- **Process supervision**: launchd plist for auto-starting the listen daemon on login.
- **`voice-route --doctor`**: Diagnostic command to check all prerequisites and configuration.

## License

MIT
