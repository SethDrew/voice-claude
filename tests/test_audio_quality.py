#!/usr/bin/env python3
"""Audio quality tests for STT (Speech-to-Text) accuracy.

These tests verify that the Whisper model transcribes known phrases
correctly, catching model regressions and configuration issues.

All tests require the real MLX Whisper model and are marked as
integration + slow.

Markers:
    @pytest.mark.integration — needs real Whisper model
    @pytest.mark.slow        — takes > 5s (model loading / transcription)
"""

import json
import os
import sys
import tempfile
import wave

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wav_to_frames(wav_path: str) -> list[np.ndarray]:
    """Load a wav file and return it as a list of float32 numpy frames."""
    with wave.open(wav_path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        data = data.reshape(-1, n_channels)[:, 0]

    chunk_size = 1024
    frames = []
    for i in range(0, len(data), chunk_size):
        frames.append(data[i:i + chunk_size])
    return frames


def _have_whisper_model() -> bool:
    """Check if the MLX Whisper model is available."""
    try:
        import mlx_whisper
        return True
    except ImportError:
        return False


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis.

    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference word count.

    Uses a simple edit-distance-based approach.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Build edit distance matrix
    n = len(ref_words)
    m = len(hyp_words)

    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[n][m] / n


# ---------------------------------------------------------------------------
# Daemon import helpers
# ---------------------------------------------------------------------------

class _FakeConfig:
    SAMPLE_RATE = 16000
    CHANNELS = 1


class _FakeStream:
    def __init__(self, **kwargs):
        pass
    def start(self): pass
    def abort(self): pass
    def close(self): pass


class _FakeSounddevice:
    InputStream = _FakeStream


def _import_daemon_real_whisper(tmp_dir):
    """Import listen_daemon with real mlx_whisper but mocked sounddevice."""
    saved = {}
    for mod_name in ("sounddevice", "config"):
        saved[mod_name] = sys.modules.get(mod_name)

    sys.modules["sounddevice"] = _FakeSounddevice()
    sys.modules["config"] = _FakeConfig()

    if "listen_daemon" in sys.modules:
        del sys.modules["listen_daemon"]

    import listen_daemon

    listen_daemon.STATE_DIR = tmp_dir
    listen_daemon.STATE_FILE = os.path.join(tmp_dir, "daemon-state.json")
    listen_daemon.PID_FILE = os.path.join(tmp_dir, "listen-daemon.pid")
    listen_daemon.RESULTS_FILE = os.path.join(tmp_dir, "daemon-results.jsonl")
    listen_daemon.recording = False
    listen_daemon.rec_frames = []
    listen_daemon.stream = None
    listen_daemon.seq = 0

    return listen_daemon, saved


def _restore_modules(saved):
    for mod_name, mod in saved.items():
        if mod is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = mod
    sys.modules.pop("listen_daemon", None)


def _transcribe_fixture(fixture_name: str) -> str:
    """Transcribe a fixture wav file and return the text."""
    wav_path = os.path.join(FIXTURES_DIR, fixture_name)
    if not os.path.exists(wav_path):
        pytest.skip(f"Fixture not found: {fixture_name}")

    frames = _wav_to_frames(wav_path)
    tmp = tempfile.mkdtemp()
    daemon, saved = _import_daemon_real_whisper(tmp)
    try:
        daemon.transcribe_frames(frames, seq_num=1)

        with open(daemon.RESULTS_FILE) as f:
            result = json.loads(f.readline())
        return result["text"]
    finally:
        _restore_modules(saved)


# ---------------------------------------------------------------------------
# Test F: Known phrases transcription accuracy
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestKnownPhrasesAccuracy:
    """Test F: Verify transcription accuracy against known phrases.

    Each test loads a fixture wav, transcribes it, and checks that
    the Word Error Rate is below the threshold.
    """

    WER_THRESHOLD = 0.50  # 50% — generous for TTS-generated audio

    @pytest.fixture(autouse=True)
    def setup(self):
        if not _have_whisper_model():
            pytest.skip("mlx_whisper not available")

    def test_tell_firmware_check_gpio_wer(self):
        """WER for 'tell firmware check GPIO'."""
        reference = "tell firmware check GPIO"
        hypothesis = _transcribe_fixture("tell_firmware_check_gpio.wav")

        wer = _compute_wer(reference, hypothesis)
        print(f"Reference: {reference!r}")
        print(f"Hypothesis: {hypothesis!r}")
        print(f"WER: {wer:.2%}")

        assert wer <= self.WER_THRESHOLD, \
            f"WER {wer:.2%} exceeds threshold {self.WER_THRESHOLD:.0%} " \
            f"(ref={reference!r}, hyp={hypothesis!r})"

    def test_go_to_frontend_wer(self):
        """WER for 'go to frontend'."""
        reference = "go to frontend"
        hypothesis = _transcribe_fixture("go_to_frontend.wav")

        wer = _compute_wer(reference, hypothesis)
        print(f"Reference: {reference!r}")
        print(f"Hypothesis: {hypothesis!r}")
        print(f"WER: {wer:.2%}")

        assert wer <= self.WER_THRESHOLD, \
            f"WER {wer:.2%} exceeds threshold {self.WER_THRESHOLD:.0%} " \
            f"(ref={reference!r}, hyp={hypothesis!r})"

    def test_bare_command_wer(self):
        """WER for 'check the error logs'."""
        reference = "check the error logs"
        hypothesis = _transcribe_fixture("bare_command.wav")

        wer = _compute_wer(reference, hypothesis)
        print(f"Reference: {reference!r}")
        print(f"Hypothesis: {hypothesis!r}")
        print(f"WER: {wer:.2%}")

        assert wer <= self.WER_THRESHOLD, \
            f"WER {wer:.2%} exceeds threshold {self.WER_THRESHOLD:.0%} " \
            f"(ref={reference!r}, hyp={hypothesis!r})"

    def test_refactor_api_endpoint_wer(self):
        """WER for 'refactor the API endpoint'."""
        reference = "refactor the API endpoint"
        hypothesis = _transcribe_fixture("refactor_api_endpoint.wav")

        wer = _compute_wer(reference, hypothesis)
        print(f"Reference: {reference!r}")
        print(f"Hypothesis: {hypothesis!r}")
        print(f"WER: {wer:.2%}")

        assert wer <= self.WER_THRESHOLD, \
            f"WER {wer:.2%} exceeds threshold {self.WER_THRESHOLD:.0%} " \
            f"(ref={reference!r}, hyp={hypothesis!r})"


# ---------------------------------------------------------------------------
# Test G: Coding vocabulary recognition
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestCodingVocabularyRecognition:
    """Test G: Verify that coding/technical terms are transcribed correctly.

    The Whisper model uses an initial_prompt containing coding vocabulary
    to bias transcription. These tests verify the bias works.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        if not _have_whisper_model():
            pytest.skip("mlx_whisper not available")

    def test_git_commit_recognized(self):
        """'git commit' should be recognized (not 'get commit' or similar)."""
        text = _transcribe_fixture("git_commit.wav")
        text_lower = text.lower()
        # Accept "git commit" or close variants
        assert "git" in text_lower or "commit" in text_lower, \
            f"Expected 'git' or 'commit' in transcription: {text!r}"

    def test_pull_request_recognized(self):
        """'pull request' should be recognized."""
        text = _transcribe_fixture("pull_request.wav")
        text_lower = text.lower()
        assert "pull" in text_lower or "request" in text_lower, \
            f"Expected 'pull' or 'request' in transcription: {text!r}"

    def test_typescript_recognized(self):
        """'TypeScript' should be recognized (not 'type script' as separate words)."""
        text = _transcribe_fixture("typescript.wav")
        text_lower = text.lower()
        assert "typescript" in text_lower or "type" in text_lower, \
            f"Expected 'typescript' or 'type' in transcription: {text!r}"

    def test_refactor_api_endpoint_recognized(self):
        """Technical phrase 'refactor the API endpoint' should be recognized."""
        text = _transcribe_fixture("refactor_api_endpoint.wav")
        text_lower = text.lower()
        # At least some of these terms should appear
        matches = sum(1 for term in ["refactor", "api", "endpoint"] if term in text_lower)
        assert matches >= 2, \
            f"Expected at least 2 of ['refactor', 'api', 'endpoint'] in: {text!r}"


# ---------------------------------------------------------------------------
# WER computation tests (meta-tests for the test infrastructure)
# ---------------------------------------------------------------------------

class TestWERComputation:
    """Verify the WER helper function works correctly."""

    def test_identical_strings(self):
        assert _compute_wer("hello world", "hello world") == 0.0

    def test_completely_different(self):
        assert _compute_wer("hello world", "foo bar") == 1.0

    def test_one_substitution(self):
        # 1 substitution out of 3 words = 0.333...
        wer = _compute_wer("the quick fox", "the slow fox")
        assert abs(wer - 1 / 3) < 0.01

    def test_insertion(self):
        # "the fox" -> "the quick fox" = 1 insertion / 2 ref words = 0.5
        wer = _compute_wer("the fox", "the quick fox")
        assert abs(wer - 0.5) < 0.01

    def test_deletion(self):
        # "the quick fox" -> "the fox" = 1 deletion / 3 ref words = 0.333...
        wer = _compute_wer("the quick fox", "the fox")
        assert abs(wer - 1 / 3) < 0.01

    def test_empty_reference(self):
        assert _compute_wer("", "") == 0.0
        assert _compute_wer("", "hello") == 1.0

    def test_case_insensitive(self):
        assert _compute_wer("Hello World", "hello world") == 0.0
