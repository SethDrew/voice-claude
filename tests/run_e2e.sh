#!/bin/bash
# Run end-to-end integration tests for voice-claude
#
# Uses the listen venv which has mlx_whisper + sounddevice + numpy + pytest.
# Tests that need BlackHole or iTerm2 skip gracefully if unavailable.
#
# Usage:
#   ./tests/run_e2e.sh              # run all e2e tests
#   ./tests/run_e2e.sh -k "not slow" # skip slow tests
#   ./tests/run_e2e.sh -k integration # run only integration tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LISTEN_VENV="$HOME/.local/share/listen/venv"

if [ ! -f "$LISTEN_VENV/bin/python" ]; then
    echo "Error: Listen venv not found at $LISTEN_VENV"
    echo "Install it first: see project README"
    exit 1
fi

if ! "$LISTEN_VENV/bin/python" -c "import pytest" 2>/dev/null; then
    echo "Error: pytest not installed in listen venv"
    echo "Run: $LISTEN_VENV/bin/pip install pytest rapidfuzz"
    exit 1
fi

echo "=== Voice-Claude E2E Integration Tests ==="
echo "Python: $($LISTEN_VENV/bin/python --version 2>&1)"
echo "Project: $PROJECT_DIR"
echo ""

exec "$LISTEN_VENV/bin/python" -m pytest \
    "$SCRIPT_DIR/test_e2e_pipeline.py" \
    "$SCRIPT_DIR/test_audio_quality.py" \
    -v --tb=short \
    "$@"
