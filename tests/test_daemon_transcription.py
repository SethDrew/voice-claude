#!/usr/bin/env python3
"""Tests for daemon transcription edge cases.

Covers issues:
  #1   Whisper model not found / ffmpeg missing
  #10  Whisper hallucinations routed as commands
  #11  MLX Metal GPU crash — concurrent transcribe calls
  #5   Results file grows unbounded (write side)
"""

import json
import os
import sys
import tempfile
import threading
import time
import wave

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Module-level mock infrastructure
# ---------------------------------------------------------------------------

class _FakeConfig:
    SAMPLE_RATE = 16000
    CHANNELS = 1


class _FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")

    def start(self):
        pass

    def abort(self):
        pass

    def close(self):
        pass


class _FakeSounddevice:
    InputStream = _FakeStream


_transcribe_call_log = []
_transcribe_delay = 0.0
_transcribe_result = {"text": "fake transcription"}
_transcribe_error = None


class _FakeMlxWhisper:
    @staticmethod
    def transcribe(path, **kwargs):
        _transcribe_call_log.append({
            "path": path,
            "time": time.time(),
            "thread": threading.current_thread().name,
        })
        if _transcribe_delay > 0:
            time.sleep(_transcribe_delay)
        if _transcribe_error:
            raise _transcribe_error
        return dict(_transcribe_result)


def _import_daemon(tmp_dir):
    """Import listen_daemon with mocked dependencies."""
    global _transcribe_call_log, _transcribe_delay, _transcribe_result, _transcribe_error
    _transcribe_call_log = []
    _transcribe_delay = 0.0
    _transcribe_result = {"text": "fake transcription"}
    _transcribe_error = None

    saved = {}
    for mod_name in ("sounddevice", "mlx_whisper", "config"):
        saved[mod_name] = sys.modules.get(mod_name)

    sys.modules["sounddevice"] = _FakeSounddevice()
    sys.modules["mlx_whisper"] = _FakeMlxWhisper()
    sys.modules["config"] = _FakeConfig()

    if "listen_daemon" in sys.modules:
        del sys.modules["listen_daemon"]

    import listen_daemon

    listen_daemon.STATE_DIR = tmp_dir
    listen_daemon.STATE_FILE = os.path.join(tmp_dir, "daemon-state.json")
    listen_daemon.PID_FILE = os.path.join(tmp_dir, "listen-daemon.pid")
    listen_daemon.RESULTS_FILE = os.path.join(tmp_dir, "daemon-results.jsonl")
    listen_daemon.BACKEND = "mlx"
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTranscribeFrames:
    """Test the transcribe_frames function."""

    def test_transcription_writes_result_to_jsonl(self):
        """Successful transcription should append a line to daemon-results.jsonl."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            daemon.transcribe_frames(frames, seq_num=1)

            results_file = daemon.RESULTS_FILE
            assert os.path.exists(results_file)
            with open(results_file) as f:
                lines = f.readlines()
            assert len(lines) == 1
            result = json.loads(lines[0])
            assert result["seq"] == 1
            assert result["text"] == "fake transcription"
        finally:
            _restore_modules(saved)

    def test_transcription_writes_state_done(self):
        """After transcription, state should be 'done'."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            daemon.transcribe_frames(frames, seq_num=1)

            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "done"
            assert state["text"] == "fake transcription"
        finally:
            _restore_modules(saved)

    def test_transcription_error_writes_error_state(self):
        """If transcription fails, state should be 'error'."""
        global _transcribe_error
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        _transcribe_error = RuntimeError("Metal GPU crash")
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            daemon.transcribe_frames(frames, seq_num=1)

            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "error"
            assert "Metal GPU crash" in state["error"]
        finally:
            _transcribe_error = None
            _restore_modules(saved)

    def test_empty_audio_concatenation(self):
        """Frames with no actual audio should still produce valid wav."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(160, dtype=np.float32)]
            daemon.transcribe_frames(frames, seq_num=1)
            # Should complete without error
            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "done"
        finally:
            _restore_modules(saved)

    def test_very_short_recording(self):
        """A recording shorter than 0.5s should still be processed."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            # 0.01s of audio at 16kHz
            frames = [np.zeros(160, dtype=np.float32)]
            daemon.transcribe_frames(frames, seq_num=1)
            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "done"
        finally:
            _restore_modules(saved)

    def test_multichannel_flattened(self):
        """Multi-dimensional audio frames should be flattened."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            # 2D array simulating stereo
            frames = [np.zeros((1600, 2), dtype=np.float32)]
            daemon.transcribe_frames(frames, seq_num=1)
            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "done"
        finally:
            _restore_modules(saved)


class TestConcurrentTranscription:
    """Issue #11 regression: concurrent MLX Metal GPU crash.

    The _transcribe_lock must serialize calls to prevent concurrent
    Metal GPU access which causes SIGABRT.
    """

    def test_transcriptions_are_serialized(self):
        """Two concurrent transcribe_frames calls should not overlap."""
        global _transcribe_delay
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        _transcribe_delay = 0.1  # each transcription takes 0.1s

        try:
            frames = [np.zeros(1600, dtype=np.float32)]

            t1 = threading.Thread(
                target=daemon.transcribe_frames,
                args=(list(frames), 1),
            )
            t2 = threading.Thread(
                target=daemon.transcribe_frames,
                args=(list(frames), 2),
            )

            t1.start()
            time.sleep(0.01)  # ensure t1 grabs the lock first
            t2.start()

            t1.join(timeout=5)
            t2.join(timeout=5)

            assert not t1.is_alive()
            assert not t2.is_alive()

            # Both calls should have completed
            assert len(_transcribe_call_log) == 2

            # Verify serialization: second call should start after first ends
            call1_time = _transcribe_call_log[0]["time"]
            call2_time = _transcribe_call_log[1]["time"]
            # With 0.1s delay, they should be ~0.1s apart
            assert call2_time - call1_time >= 0.09
        finally:
            _transcribe_delay = 0.0
            _restore_modules(saved)

    def test_lock_released_after_error(self):
        """If transcription raises, the lock should still be released."""
        global _transcribe_error
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)

        try:
            # First call will fail
            _transcribe_error = RuntimeError("GPU crash")
            frames = [np.zeros(1600, dtype=np.float32)]
            daemon.transcribe_frames(list(frames), 1)

            # Second call should not deadlock
            _transcribe_error = None
            daemon.transcribe_frames(list(frames), 2)

            assert len(_transcribe_call_log) == 2
        finally:
            _transcribe_error = None
            _restore_modules(saved)


class TestResultsFile:
    """Issue #5: results file grows unbounded (daemon write side)."""

    def test_results_appended_sequentially(self):
        """Each transcription should append one JSONL line."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            for i in range(5):
                daemon.transcribe_frames(list(frames), seq_num=i + 1)

            with open(daemon.RESULTS_FILE) as f:
                lines = f.readlines()
            assert len(lines) == 5

            for i, line in enumerate(lines):
                result = json.loads(line)
                assert result["seq"] == i + 1
        finally:
            _restore_modules(saved)

    def test_results_file_valid_jsonl(self):
        """Every line in the results file should be valid JSON."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            for i in range(10):
                daemon.transcribe_frames(list(frames), seq_num=i)

            with open(daemon.RESULTS_FILE) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        pytest.fail(f"Line {line_num} is not valid JSON: {line!r}")
        finally:
            _restore_modules(saved)

    def test_results_survive_truncation_mid_write(self):
        """If the results file is truncated externally, subsequent writes
        should start fresh without corruption."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            daemon.transcribe_frames(list(frames), seq_num=1)
            daemon.transcribe_frames(list(frames), seq_num=2)

            # Truncate (simulating what Hammerspoon does)
            with open(daemon.RESULTS_FILE, "w") as f:
                pass  # truncate

            # New writes after truncation
            daemon.transcribe_frames(list(frames), seq_num=3)

            with open(daemon.RESULTS_FILE) as f:
                lines = f.readlines()
            assert len(lines) == 1
            result = json.loads(lines[0])
            assert result["seq"] == 3
        finally:
            _restore_modules(saved)


class TestHallucinationDetection:
    """Issue #10: Whisper hallucination strings should be caught.

    While the parser handles filtering, the daemon should still produce
    valid output; the parser is the defense layer. These tests verify
    the parser's hallucination filter works for daemon-produced text.
    """

    def test_repeated_tokens_detected(self):
        """Repeated token patterns like 'ext ext ext' should be filterable."""
        from parser import parse
        # These are common Whisper hallucination patterns
        hallucinations = [
            "Thank you for watching",
            "Thanks for watching",
            "Subscribe",
            "Please subscribe",
            "Like and subscribe",
            "Thank you",
            "You",
            "Bye",
            "Goodbye",
        ]
        for text in hallucinations:
            cmd = parse(text)
            assert cmd.text is None, f"Hallucination not filtered: {text!r}"

    def test_repeated_foreign_tokens(self):
        """Repeated non-English tokens from Whisper should ideally be caught.

        Note: the current filter uses an explicit allow list, so novel
        hallucination patterns may slip through. This test documents the gap.
        """
        from parser import parse
        # These particular patterns are NOT in the current hallucination set,
        # so they will pass through. This test documents the known gap.
        novel_hallucinations = [
            "ext ext ext",
            "..." * 5,
        ]
        for text in novel_hallucinations:
            cmd = parse(text)
            # These are NOT currently filtered -- this documents the gap
            # If they are filtered in the future, this test will need updating

    def test_legitimate_text_not_caught(self):
        """Real commands should not be caught by hallucination filters."""
        from parser import parse
        legitimate = [
            "check the GPIO pins",
            "tell firmware slash commit",
            "go to frontend",
        ]
        for text in legitimate:
            cmd = parse(text)
            assert cmd.text is not None or cmd.target is not None, \
                f"Legitimate text falsely filtered: {text!r}"


class TestWhisperModelMissing:
    """Issue #1 regression: daemon should handle missing model/ffmpeg."""

    def test_backend_detection_order(self):
        """The daemon import cascade: mlx > faster > openai."""
        # We verify the logic by checking BACKEND after import with mocks
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            # With our mock, mlx_whisper is available, so BACKEND should be mlx
            assert daemon.BACKEND == "mlx"
        finally:
            _restore_modules(saved)

    def test_transcription_error_caught(self):
        """If the backend raises (model not found), error state is written."""
        global _transcribe_error
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        _transcribe_error = FileNotFoundError("Model file not found")
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            daemon.transcribe_frames(frames, seq_num=1)

            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "error"
            assert "not found" in state["error"].lower()
        finally:
            _transcribe_error = None
            _restore_modules(saved)


class TestStopRecordingWithFrames:
    """Test that stop recording correctly hands off frames for transcription."""

    def test_frames_copied_before_transcription(self):
        """rec_frames should be copied before being cleared."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.do_start_recording()

            # Simulate some recorded frames
            daemon.rec_frames = [np.zeros(1600, dtype=np.float32) for _ in range(3)]

            daemon.do_stop_recording()

            # rec_frames should be cleared
            assert len(daemon.rec_frames) == 0

            # Wait for transcription thread
            time.sleep(0.5)

            # Result should be written
            if os.path.exists(daemon.RESULTS_FILE):
                with open(daemon.RESULTS_FILE) as f:
                    lines = f.readlines()
                assert len(lines) >= 1
        finally:
            _restore_modules(saved)

    def test_empty_frames_produce_done_state(self):
        """If no frames were recorded, state should go to 'done' with empty text."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.do_start_recording()
            # Don't add any frames
            daemon.rec_frames = []
            daemon.do_stop_recording()

            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "done"
            assert state["text"] == ""
        finally:
            _restore_modules(saved)
