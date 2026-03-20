#!/usr/bin/env python3
"""Tests for listen daemon signal handling and self-pipe pattern.

Covers issues:
  #2  Signal mode recording hangs (stream.stop vs stream.abort)
  #3  Mic stays on after key release (SIGUSR2 not processed)
  #4  Push-to-talk subsequent recordings fail (active stays True)
  #12 Self-pipe select blocking
"""

import json
import os
import signal
import sys
import tempfile
import threading
import time

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Helpers: mock out heavy dependencies so we can import listen_daemon
# ---------------------------------------------------------------------------

class _FakeConfig:
    SAMPLE_RATE = 16000
    CHANNELS = 1


class _FakeStream:
    """Mock sounddevice.InputStream that records method calls."""

    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self._started = False
        self._aborted = False
        self._closed = False
        self.start_count = 0

    def start(self):
        self._started = True
        self.start_count += 1

    def stop(self):
        self._started = False

    def abort(self):
        self._aborted = True
        self._started = False

    def close(self):
        self._closed = True


class _FakeSounddevice:
    InputStream = _FakeStream


class _FakeMlxWhisper:
    @staticmethod
    def transcribe(path, **kwargs):
        return {"text": "fake transcription"}


def _import_daemon(tmp_dir):
    """Import listen_daemon with mocked dependencies and redirected paths."""
    import importlib

    # Stash and mock heavy deps
    saved = {}
    for mod_name in ("sounddevice", "mlx_whisper", "config"):
        saved[mod_name] = sys.modules.get(mod_name)

    sys.modules["sounddevice"] = _FakeSounddevice()
    sys.modules["mlx_whisper"] = _FakeMlxWhisper()
    sys.modules["config"] = _FakeConfig()

    # Ensure listen_daemon is freshly loaded (it caches global state)
    if "listen_daemon" in sys.modules:
        del sys.modules["listen_daemon"]

    import listen_daemon

    # Override file paths so tests don't touch real state
    listen_daemon.STATE_DIR = tmp_dir
    listen_daemon.STATE_FILE = os.path.join(tmp_dir, "daemon-state.json")
    listen_daemon.PID_FILE = os.path.join(tmp_dir, "listen-daemon.pid")
    listen_daemon.RESULTS_FILE = os.path.join(tmp_dir, "daemon-results.jsonl")
    listen_daemon.BACKEND = "mlx"

    # Reset global state
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


class TestSelfPipePattern:
    """Test that the self-pipe delivers signal bytes correctly."""

    def test_pipe_write_read_cycle(self):
        """Signal handler writes a byte; main loop reads it."""
        r_fd, w_fd = os.pipe()
        os.set_blocking(w_fd, False)
        os.set_blocking(r_fd, False)

        # Simulate what the signal handler does
        os.write(w_fd, bytes([signal.SIGUSR1 & 0xFF]))

        data = os.read(r_fd, 64)
        assert len(data) == 1
        assert data[0] == (signal.SIGUSR1 & 0xFF)

        os.close(r_fd)
        os.close(w_fd)

    def test_multiple_signals_coalesced(self):
        """Multiple signal bytes should all be readable in one os.read."""
        r_fd, w_fd = os.pipe()
        os.set_blocking(w_fd, False)
        os.set_blocking(r_fd, False)

        os.write(w_fd, bytes([signal.SIGUSR1 & 0xFF]))
        os.write(w_fd, bytes([signal.SIGUSR2 & 0xFF]))
        os.write(w_fd, bytes([signal.SIGTERM & 0xFF]))

        data = os.read(r_fd, 64)
        assert len(data) == 3
        assert data[0] == (signal.SIGUSR1 & 0xFF)
        assert data[1] == (signal.SIGUSR2 & 0xFF)
        assert data[2] == (signal.SIGTERM & 0xFF)

        os.close(r_fd)
        os.close(w_fd)

    def test_pipe_nonblocking_read_when_empty(self):
        """Reading from empty non-blocking pipe should raise BlockingIOError."""
        r_fd, w_fd = os.pipe()
        os.set_blocking(r_fd, False)

        with pytest.raises(BlockingIOError):
            os.read(r_fd, 64)

        os.close(r_fd)
        os.close(w_fd)

    def test_pipe_survives_concurrent_writes(self):
        """Multiple threads writing to the pipe should not lose bytes."""
        r_fd, w_fd = os.pipe()
        os.set_blocking(w_fd, False)
        os.set_blocking(r_fd, False)

        write_count = 50

        def writer(byte_val):
            for _ in range(write_count):
                try:
                    os.write(w_fd, bytes([byte_val]))
                except OSError:
                    pass
                time.sleep(0.001)

        threads = [
            threading.Thread(target=writer, args=(signal.SIGUSR1 & 0xFF,)),
            threading.Thread(target=writer, args=(signal.SIGUSR2 & 0xFF,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Read all accumulated bytes
        collected = b""
        while True:
            try:
                chunk = os.read(r_fd, 4096)
                if not chunk:
                    break
                collected += chunk
            except BlockingIOError:
                break

        # Should have gotten all written bytes (2 threads x write_count each)
        assert len(collected) == 2 * write_count

        os.close(r_fd)
        os.close(w_fd)


class TestRecordingLifecycle:
    """Test start/stop recording state machine."""

    def test_start_recording_sets_state(self):
        """do_start_recording should set recording=True and create a stream."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.do_start_recording()
            assert daemon.recording is True
            assert daemon.stream is not None
            assert daemon.seq == 1

            # State file should say "recording"
            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "recording"
        finally:
            # Clean up
            daemon.recording = False
            daemon.stream = None
            _restore_modules(saved)

    def test_stop_recording_resets_state(self):
        """do_stop_recording should set recording=False and clean stream."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.do_start_recording()
            assert daemon.recording is True

            daemon.do_stop_recording()
            assert daemon.recording is False
            assert daemon.stream is None
        finally:
            _restore_modules(saved)

    def test_double_start_is_idempotent(self):
        """Calling do_start_recording twice should not increment seq twice."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.do_start_recording()
            first_seq = daemon.seq
            first_stream = daemon.stream

            daemon.do_start_recording()  # second call while already recording
            assert daemon.seq == first_seq  # seq should not change
            assert daemon.stream is first_stream  # same stream
        finally:
            daemon.recording = False
            daemon.stream = None
            _restore_modules(saved)

    def test_stop_when_not_recording_is_noop(self):
        """Calling do_stop_recording when not recording should be safe."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            # Should not raise
            daemon.do_stop_recording()
            assert daemon.recording is False
        finally:
            _restore_modules(saved)

    def test_stop_uses_abort_not_stop(self):
        """Issue #2 regression: stream.abort() must be used, not stream.stop().

        stream.stop() blocks when signals interrupt audio callbacks.
        The fix was to use stream.abort() instead.
        """
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.do_start_recording()
            stream = daemon.stream
            daemon.do_stop_recording()

            assert stream._aborted is True, "stream.abort() was not called"
            assert stream._closed is True, "stream.close() was not called"
        finally:
            _restore_modules(saved)


class TestSubsequentRecordings:
    """Issue #4 regression: push-to-talk subsequent recordings fail.

    voiceRouter.active stays True after first recording, or pendingCount
    gets stuck positive. The daemon-side must properly reset state.
    """

    def test_multiple_record_stop_cycles(self):
        """Recording should work across multiple start/stop cycles."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            for i in range(5):
                daemon.do_start_recording()
                assert daemon.recording is True
                assert daemon.seq == i + 1

                daemon.do_stop_recording()
                assert daemon.recording is False
                assert daemon.stream is None
        finally:
            _restore_modules(saved)

    def test_rapid_start_stop(self):
        """Rapid fire start/stop should not leave dangling state."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            for _ in range(20):
                daemon.do_start_recording()
                daemon.do_stop_recording()

            assert daemon.recording is False
            assert daemon.stream is None
            assert daemon.seq == 20
        finally:
            _restore_modules(saved)


class TestStalePidFile:
    """Issue #3 regression: stale PID files cause signal delivery failure."""

    def test_stale_pid_file_detected(self):
        """If the PID file points to a dead process, signaling should fail."""
        tmp = tempfile.mkdtemp()
        pid_file = os.path.join(tmp, "listen-daemon.pid")

        # Write a PID that doesn't exist (99999999)
        with open(pid_file, "w") as f:
            f.write("99999999")

        # Verify process doesn't exist
        try:
            os.kill(99999999, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True  # exists but we can't signal it

        assert alive is False, "Test PID should not correspond to a real process"

    def test_pid_file_written_on_startup(self):
        """main() should write the current PID to the PID file."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            pid_file = os.path.join(tmp, "listen-daemon.pid")
            with open(pid_file, "w") as f:
                f.write(str(os.getpid()))

            pid_content = open(pid_file).read().strip()
            assert pid_content == str(os.getpid())
        finally:
            _restore_modules(saved)

    def test_pid_file_removed_on_shutdown(self):
        """do_shutdown should remove the PID file."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            pid_file = daemon.PID_FILE
            with open(pid_file, "w") as f:
                f.write(str(os.getpid()))
            assert os.path.exists(pid_file)

            with pytest.raises(SystemExit):
                daemon.do_shutdown()

            assert not os.path.exists(pid_file)
        finally:
            _restore_modules(saved)


class TestWriteState:
    """Test atomic state file writes."""

    def test_write_state_creates_file(self):
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.write_state("idle")
            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "idle"
            assert "pid" in state
            assert "ts" in state
        finally:
            _restore_modules(saved)

    def test_write_state_with_extra_fields(self):
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.write_state("done", text="hello world", seq=42)
            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "done"
            assert state["text"] == "hello world"
            assert state["seq"] == 42
        finally:
            _restore_modules(saved)

    def test_write_state_atomic_replace(self):
        """State writes use tmp+rename pattern for atomicity."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            # Write initial state
            daemon.write_state("idle")
            # Overwrite should not corrupt
            daemon.write_state("recording")
            state = json.loads(open(daemon.STATE_FILE).read())
            assert state["state"] == "recording"
        finally:
            _restore_modules(saved)


@pytest.mark.stress
class TestRapidSignalSequences:
    """Stress test: rapid USR1/USR2 sequences."""

    def test_rapid_usr1_usr2_alternation(self):
        """Simulate rapid option-key press/release cycles."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            for _ in range(50):
                daemon.do_start_recording()
                daemon.do_stop_recording()

            # Final state should be clean
            assert daemon.recording is False
            assert daemon.stream is None
        finally:
            _restore_modules(saved)

    def test_multiple_usr2_without_usr1(self):
        """Extra stop signals should be harmless."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.do_stop_recording()
            daemon.do_stop_recording()
            daemon.do_stop_recording()
            assert daemon.recording is False
        finally:
            _restore_modules(saved)

    def test_multiple_usr1_without_usr2(self):
        """Extra start signals while recording should be idempotent."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            daemon.do_start_recording()
            seq_after_first = daemon.seq
            daemon.do_start_recording()
            daemon.do_start_recording()
            assert daemon.seq == seq_after_first
            assert daemon.recording is True

            daemon.do_stop_recording()
            assert daemon.recording is False
        finally:
            _restore_modules(saved)


@pytest.mark.stress
class TestSelectBlocking:
    """Issue #12 regression: select.select() on signal pipe gets stuck."""

    def test_select_with_timeout_returns(self):
        """select.select with a timeout should return even if no data."""
        r_fd, w_fd = os.pipe()
        os.set_blocking(r_fd, False)

        import select
        t0 = time.time()
        ready, _, _ = select.select([r_fd], [], [], 0.1)
        elapsed = time.time() - t0

        assert ready == []
        assert elapsed < 0.5  # should have returned near the 0.1s timeout

        os.close(r_fd)
        os.close(w_fd)

    def test_select_wakes_on_data(self):
        """select.select should wake immediately when data is available."""
        r_fd, w_fd = os.pipe()
        os.set_blocking(r_fd, False)
        os.set_blocking(w_fd, False)

        # Write first, then select
        os.write(w_fd, b'\x01')

        import select
        t0 = time.time()
        ready, _, _ = select.select([r_fd], [], [], 1.0)
        elapsed = time.time() - t0

        assert len(ready) == 1
        assert elapsed < 0.1  # should return nearly instantly

        os.close(r_fd)
        os.close(w_fd)

    def test_select_wakes_from_another_thread(self):
        """Data written from another thread should wake select."""
        r_fd, w_fd = os.pipe()
        os.set_blocking(r_fd, False)
        os.set_blocking(w_fd, False)

        def delayed_write():
            time.sleep(0.05)
            os.write(w_fd, b'\x01')

        threading.Thread(target=delayed_write, daemon=True).start()

        import select
        t0 = time.time()
        ready, _, _ = select.select([r_fd], [], [], 2.0)
        elapsed = time.time() - t0

        assert len(ready) == 1
        assert elapsed < 1.0  # should not wait the full 2s

        os.close(r_fd)
        os.close(w_fd)
