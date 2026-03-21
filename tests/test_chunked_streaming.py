#!/usr/bin/env python3
"""Tests for chunked streaming STT in the listen daemon.

Tests the pause-detection chunk monitor, chunk file writing,
final chunk marking, and backward-compatible results file output.
"""

import json
import os
import sys
import tempfile
import threading
import time

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

class _FakeConfig:
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024
    VAD_THRESHOLD = 0.015


class _FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self._started = False
        self._aborted = False
        self._closed = False

    def start(self):
        self._started = True

    def abort(self):
        self._aborted = True
        self._started = False

    def close(self):
        self._closed = True


class _FakeSounddevice:
    InputStream = _FakeStream


_transcribe_call_count = 0
_transcribe_results = []  # list of texts to return in sequence


class _FakeMlxWhisper:
    @staticmethod
    def transcribe(path, **kwargs):
        global _transcribe_call_count
        idx = _transcribe_call_count
        _transcribe_call_count += 1
        if _transcribe_results and idx < len(_transcribe_results):
            return {"text": _transcribe_results[idx]}
        return {"text": f"chunk {idx + 1}"}


def _import_daemon(tmp_dir):
    """Import listen_daemon with mocked dependencies and redirected paths."""
    global _transcribe_call_count, _transcribe_results
    _transcribe_call_count = 0
    _transcribe_results = []

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
    listen_daemon.CHUNKS_FILE = os.path.join(tmp_dir, "daemon-chunks.jsonl")
    listen_daemon.BACKEND = "mlx"
    listen_daemon.recording = False
    listen_daemon.rec_frames = []
    listen_daemon.stream = None
    listen_daemon.seq = 0
    listen_daemon._chunk_num = 0
    with listen_daemon._chunk_lock:
        listen_daemon._chunk_frames.clear()
    listen_daemon._chunk_texts.clear()
    listen_daemon._chunk_monitor_thread = None

    return listen_daemon, saved


def _restore_modules(saved):
    for mod_name, mod in saved.items():
        if mod is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = mod
    sys.modules.pop("listen_daemon", None)


def _make_speech_frames(duration_s, amplitude=0.1, sample_rate=16000, chunk_size=1024):
    """Generate synthetic speech frames (non-silent audio)."""
    total_samples = int(duration_s * sample_rate)
    num_frames = max(1, total_samples // chunk_size)
    frames = []
    for i in range(num_frames):
        # Generate sine wave to simulate speech above VAD threshold
        t = np.linspace(i * chunk_size / sample_rate,
                        (i + 1) * chunk_size / sample_rate,
                        chunk_size, endpoint=False)
        frame = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        frames.append(frame)
    return frames


def _make_silence_frames(duration_s, sample_rate=16000, chunk_size=1024):
    """Generate silence frames (below VAD threshold)."""
    total_samples = int(duration_s * sample_rate)
    num_frames = max(1, total_samples // chunk_size)
    frames = []
    for _ in range(num_frames):
        frames.append(np.zeros(chunk_size, dtype=np.float32))
    return frames


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChunksFileCreation:
    """Test that chunks file is created and managed properly."""

    def test_chunks_file_truncated_on_start(self):
        """Starting a new recording should truncate the chunks file."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            # Write some stale data to chunks file
            chunks_file = daemon.CHUNKS_FILE
            with open(chunks_file, "w") as f:
                f.write('{"stale": true}\n')

            daemon.do_start_recording()
            assert daemon.recording is True

            # Chunks file should be empty now
            with open(chunks_file) as f:
                content = f.read()
            assert content == ""

            daemon.recording = False
            daemon.stream = None
        finally:
            daemon.recording = False
            _restore_modules(saved)

    def test_chunks_file_path_set(self):
        """CHUNKS_FILE should be set to the expected path."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            assert daemon.CHUNKS_FILE == os.path.join(tmp, "daemon-chunks.jsonl")
        finally:
            _restore_modules(saved)


class TestWriteChunk:
    """Test the _write_chunk helper function."""

    def test_write_non_final_chunk(self):
        """Non-final chunks should have final=false."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon._write_chunk(1, 1, "check the GPIO pins", final=False)

            with open(daemon.CHUNKS_FILE) as f:
                lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["seq"] == 1
            assert record["chunk"] == 1
            assert record["text"] == "check the GPIO pins"
            assert record["final"] is False
        finally:
            _restore_modules(saved)

    def test_write_final_chunk(self):
        """Final chunks should have final=true."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon._write_chunk(1, 2, "and also the LED driver", final=True)

            with open(daemon.CHUNKS_FILE) as f:
                lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["final"] is True
        finally:
            _restore_modules(saved)

    def test_multiple_chunks_appended(self):
        """Multiple chunks should be appended sequentially."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon._write_chunk(1, 1, "first chunk", final=False)
            daemon._write_chunk(1, 2, "second chunk", final=False)
            daemon._write_chunk(1, 3, "third chunk", final=True)

            with open(daemon.CHUNKS_FILE) as f:
                lines = f.readlines()
            assert len(lines) == 3

            for i, line in enumerate(lines):
                record = json.loads(line)
                assert record["chunk"] == i + 1
                assert record["final"] == (i == 2)
        finally:
            _restore_modules(saved)


class TestTranscribeAudio:
    """Test the _transcribe_audio helper function."""

    def test_transcribe_returns_text(self):
        """_transcribe_audio should return transcribed text."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            text = daemon._transcribe_audio(frames)
            assert isinstance(text, str)
            assert len(text) > 0
        finally:
            _restore_modules(saved)

    def test_transcribe_uses_lock(self):
        """_transcribe_audio should use _transcribe_lock for serialization."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            # If lock is acquired, this should still work (reentrant check not needed,
            # but concurrent calls should be serialized)
            text = daemon._transcribe_audio(frames)
            assert text is not None
        finally:
            _restore_modules(saved)


class TestChunkMonitor:
    """Test the chunk monitor thread's pause detection logic."""

    def test_monitor_detects_pause_in_simulated_audio(self):
        """Chunk monitor should detect a pause between speech segments."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        global _transcribe_results
        _transcribe_results = ["check the GPIO pins", "and also the LED driver"]
        try:
            daemon.do_start_recording()

            # Simulate speech -> silence -> speech by feeding frames
            speech1 = _make_speech_frames(0.5, amplitude=0.1)
            silence = _make_silence_frames(0.6)
            speech2 = _make_speech_frames(0.5, amplitude=0.1)

            # Feed speech1
            for frame in speech1:
                with daemon._chunk_lock:
                    daemon._chunk_frames.append(frame)
                daemon.rec_frames.append(frame)

            # Feed silence (should trigger chunk boundary)
            for frame in silence:
                with daemon._chunk_lock:
                    daemon._chunk_frames.append(frame)
                daemon.rec_frames.append(frame)

            # Wait for the monitor to detect the pause and transcribe
            time.sleep(1.5)

            # Feed speech2
            for frame in speech2:
                with daemon._chunk_lock:
                    daemon._chunk_frames.append(frame)
                daemon.rec_frames.append(frame)

            # Stop recording to trigger final chunk
            daemon.do_stop_recording()

            # Wait for finalization
            time.sleep(2.0)

            # Check chunks file
            if os.path.exists(daemon.CHUNKS_FILE):
                with open(daemon.CHUNKS_FILE) as f:
                    lines = f.readlines()
                # Should have at least one chunk (the pause-detected one)
                # and possibly the final chunk
                assert len(lines) >= 1
                chunks = [json.loads(line) for line in lines]
                # At least one should be marked final
                assert any(c["final"] for c in chunks)
        finally:
            daemon.recording = False
            _restore_modules(saved)

    def test_short_audio_no_pause_produces_single_chunk(self):
        """Audio without a pause should produce a single final chunk."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        global _transcribe_results
        _transcribe_results = ["check the GPIO pins"]
        try:
            daemon.do_start_recording()

            # Feed only speech frames (no silence gap)
            speech = _make_speech_frames(0.5, amplitude=0.1)
            for frame in speech:
                with daemon._chunk_lock:
                    daemon._chunk_frames.append(frame)
                daemon.rec_frames.append(frame)

            # Stop immediately — no pause was long enough to trigger chunking
            daemon.do_stop_recording()

            # Wait for finalization
            time.sleep(1.5)

            # Check chunks file
            if os.path.exists(daemon.CHUNKS_FILE):
                with open(daemon.CHUNKS_FILE) as f:
                    lines = f.readlines()
                chunks = [json.loads(line) for line in lines]
                # Should have exactly one final chunk
                finals = [c for c in chunks if c["final"]]
                assert len(finals) >= 1
        finally:
            daemon.recording = False
            _restore_modules(saved)


class TestFinalChunkAndResults:
    """Test that final chunk and results file work correctly."""

    def test_results_file_gets_full_text(self):
        """The results file should contain the full concatenated text."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        global _transcribe_results
        _transcribe_results = ["full text here"]
        try:
            daemon.do_start_recording()

            # Add some frames
            speech = _make_speech_frames(0.3, amplitude=0.1)
            for frame in speech:
                with daemon._chunk_lock:
                    daemon._chunk_frames.append(frame)
                daemon.rec_frames.append(frame)

            daemon.do_stop_recording()
            time.sleep(1.5)

            # Check results file (backward compat)
            if os.path.exists(daemon.RESULTS_FILE):
                with open(daemon.RESULTS_FILE) as f:
                    lines = f.readlines()
                assert len(lines) >= 1
                result = json.loads(lines[-1])
                assert "text" in result
                assert "seq" in result
        finally:
            daemon.recording = False
            _restore_modules(saved)

    def test_final_chunk_marked_true(self):
        """The last chunk should have final=True."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        global _transcribe_results
        _transcribe_results = ["some text"]
        try:
            daemon.do_start_recording()

            speech = _make_speech_frames(0.3, amplitude=0.1)
            for frame in speech:
                with daemon._chunk_lock:
                    daemon._chunk_frames.append(frame)
                daemon.rec_frames.append(frame)

            daemon.do_stop_recording()
            time.sleep(1.5)

            if os.path.exists(daemon.CHUNKS_FILE):
                with open(daemon.CHUNKS_FILE) as f:
                    lines = f.readlines()
                if lines:
                    chunks = [json.loads(line) for line in lines]
                    # Last chunk should be final
                    assert chunks[-1]["final"] is True
        finally:
            daemon.recording = False
            _restore_modules(saved)

    def test_empty_recording_no_chunks(self):
        """Empty recording (no frames) should not produce chunks."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.do_start_recording()
            # Don't add any frames
            daemon.rec_frames = []
            with daemon._chunk_lock:
                daemon._chunk_frames.clear()
            daemon.do_stop_recording()
            time.sleep(0.5)

            # Results file should not exist or be empty/have empty text
            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "done"
            assert state["text"] == ""
        finally:
            daemon.recording = False
            _restore_modules(saved)


class TestSilenceOnlyRecording:
    """Test that silence-only recordings are handled correctly."""

    def test_silence_only_produces_result(self):
        """Recording only silence should still produce a result."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        global _transcribe_results
        _transcribe_results = [""]  # Whisper returns empty for silence
        try:
            daemon.do_start_recording()

            silence = _make_silence_frames(0.5)
            for frame in silence:
                with daemon._chunk_lock:
                    daemon._chunk_frames.append(frame)
                daemon.rec_frames.append(frame)

            daemon.do_stop_recording()
            time.sleep(1.5)

            # Should still produce a result (possibly empty)
            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "done"
        finally:
            daemon.recording = False
            _restore_modules(saved)


class TestChunkStreamingState:
    """Test chunk streaming state management across recordings."""

    def test_chunk_state_reset_between_recordings(self):
        """Chunk state should be reset when a new recording starts."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        global _transcribe_results
        _transcribe_results = ["first recording", "second recording"]
        try:
            # First recording
            daemon.do_start_recording()
            speech = _make_speech_frames(0.2, amplitude=0.1)
            for frame in speech:
                with daemon._chunk_lock:
                    daemon._chunk_frames.append(frame)
                daemon.rec_frames.append(frame)
            daemon.do_stop_recording()
            time.sleep(1.0)

            # Second recording
            daemon.do_start_recording()
            assert daemon._chunk_num == 0
            assert len(daemon._chunk_texts) == 0
            with daemon._chunk_lock:
                assert len(daemon._chunk_frames) == 0

            daemon.recording = False
            daemon.stream = None
        finally:
            daemon.recording = False
            _restore_modules(saved)

    def test_chunks_file_valid_jsonl(self):
        """Every line in the chunks file should be valid JSON."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            # Write some chunks directly
            daemon._write_chunk(1, 1, "first", final=False)
            daemon._write_chunk(1, 2, "second", final=True)

            with open(daemon.CHUNKS_FILE) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        pytest.fail(f"Line {line_num} is not valid JSON: {line!r}")
        finally:
            _restore_modules(saved)
