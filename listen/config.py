#!/usr/bin/env python3
"""
listen configuration
Edit this file directly and reinstall to change defaults.
"""

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# Whisper model - see https://github.com/openai/whisper#available-models-and-languages
# tiny   - ~75MB  - fastest, lowest quality
# base   - ~140MB - good balance
# small  - ~460MB - better quality
# medium - ~1.5GB - high quality
# large  - ~3GB   - best quality
MODEL = "tiny"

# Default language - set to None for auto-detect
# Examples: "en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"
LANGUAGE = "en"

# Features - set to False to disable
ENABLE_FILE_MODE = True    # -f, --file: transcribe from file
ENABLE_SIGNAL_MODE = True  # --signal-mode: SIGUSR1 to stop
ENABLE_VAD = True          # --vad: voice activity detection
ENABLE_CODEVOICE = True    # --codevoice: full-width UI
ENABLE_JSON = True         # -j, --json: JSON output
ENABLE_OUTPUT_FILE = True  # -o, --output: write to file

# VAD settings
VAD_THRESHOLD = 0.015       # volume threshold for silence detection
VAD_DEFAULT_DURATION = 2.0  # seconds of silence before auto-stop

# UI settings
COLOR_OUTPUT = True   # ANSI colors
SHOW_VERBOSE = False  # Debug info

# Claude command
CLAUDE_CMD = "claude"
