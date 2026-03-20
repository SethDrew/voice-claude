#!/usr/bin/env python3
"""End-to-end tests for the results pipeline (daemon -> JSONL -> consumer).

Covers issues:
  #4   pendingCount gets stuck positive
  #5   Results file grows unbounded
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
# Mock infrastructure (same pattern as test_daemon_signals.py)
# ---------------------------------------------------------------------------

class _FakeConfig:
    SAMPLE_RATE = 16000
    CHANNELS = 1


class _FakeStream:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def abort(self):
        pass

    def close(self):
        pass


class _FakeSounddevice:
    InputStream = _FakeStream


class _FakeMlxWhisper:
    @staticmethod
    def transcribe(path, **kwargs):
        return {"text": "test transcription"}


def _import_daemon(tmp_dir):
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
# Simulated Hammerspoon consumer
# ---------------------------------------------------------------------------

class SimulatedConsumer:
    """Simulates the Hammerspoon Lua consumer (consumeResults function).

    This replicates the logic from init.lua:
    - Reads JSONL file line by line
    - Tracks lastResultLine to avoid re-reading
    - Decrements pendingCount for each result
    - Collects text into a route queue
    """

    def __init__(self, results_file):
        self.results_file = results_file
        self.last_result_line = 0
        self.pending_count = 0
        self.route_queue = []

    def consume(self):
        """Read new results from the JSONL file."""
        if not os.path.exists(self.results_file):
            return

        with open(self.results_file) as f:
            lines = f.readlines()

        line_num = 0
        for line in lines:
            line_num += 1
            if line_num > self.last_result_line:
                try:
                    result = json.loads(line.strip())
                    if result.get("text") and len(result["text"]) > 0:
                        self.route_queue.append(result["text"])
                        self.pending_count = max(0, self.pending_count - 1)
                    else:
                        self.pending_count = max(0, self.pending_count - 1)
                except json.JSONDecodeError:
                    pass
                self.last_result_line = line_num

    def truncate_if_idle(self):
        """Truncate results file when idle (pendingCount <= 0 and queue empty)."""
        if self.pending_count <= 0 and len(self.route_queue) == 0:
            with open(self.results_file, "w") as f:
                pass  # truncate
            self.last_result_line = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDaemonWritesConsumerReads:
    """Test that daemon writes to JSONL and consumer reads in order."""

    def test_single_result_round_trip(self):
        """Daemon writes one result, consumer reads it."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            daemon.transcribe_frames(frames, seq_num=1)

            consumer = SimulatedConsumer(daemon.RESULTS_FILE)
            consumer.pending_count = 1
            consumer.consume()

            assert len(consumer.route_queue) == 1
            assert consumer.route_queue[0] == "test transcription"
            assert consumer.pending_count == 0
        finally:
            _restore_modules(saved)

    def test_multiple_results_in_order(self):
        """Multiple results should be read in sequence order."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            for i in range(5):
                daemon.transcribe_frames(list(frames), seq_num=i + 1)

            consumer = SimulatedConsumer(daemon.RESULTS_FILE)
            consumer.pending_count = 5
            consumer.consume()

            assert len(consumer.route_queue) == 5
            assert consumer.pending_count == 0
        finally:
            _restore_modules(saved)

    def test_incremental_consumption(self):
        """Consumer should pick up new results on subsequent consume() calls."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            consumer = SimulatedConsumer(daemon.RESULTS_FILE)

            # Write first result
            daemon.transcribe_frames(list(frames), seq_num=1)
            consumer.pending_count = 1
            consumer.consume()
            assert len(consumer.route_queue) == 1
            assert consumer.pending_count == 0

            # Write second result
            daemon.transcribe_frames(list(frames), seq_num=2)
            consumer.pending_count = 1
            consumer.consume()
            assert len(consumer.route_queue) == 2
            assert consumer.pending_count == 0
        finally:
            _restore_modules(saved)


class TestRapidWrites:
    """Test that rapid writes don't lose results."""

    def test_rapid_sequential_writes(self):
        """50 rapid sequential transcriptions should all be readable."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            for i in range(50):
                daemon.transcribe_frames(list(frames), seq_num=i + 1)

            consumer = SimulatedConsumer(daemon.RESULTS_FILE)
            consumer.pending_count = 50
            consumer.consume()

            assert len(consumer.route_queue) == 50
            assert consumer.pending_count == 0
        finally:
            _restore_modules(saved)

    @pytest.mark.stress
    def test_concurrent_writes(self):
        """Multiple threads writing results concurrently."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            threads = []
            for i in range(10):
                t = threading.Thread(
                    target=daemon.transcribe_frames,
                    args=(list(frames), i + 1),
                )
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            # All threads should have completed
            for t in threads:
                assert not t.is_alive()

            consumer = SimulatedConsumer(daemon.RESULTS_FILE)
            consumer.pending_count = 10
            consumer.consume()

            # Due to lock serialization, all 10 should be present
            assert len(consumer.route_queue) == 10
        finally:
            _restore_modules(saved)


class TestFileTruncation:
    """Issue #5: Results file grows unbounded.

    The Hammerspoon consumer truncates the file when idle.
    Test that truncation doesn't lose in-flight results.
    """

    def test_truncation_resets_line_counter(self):
        """After truncation, lastResultLine should reset to 0."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            consumer = SimulatedConsumer(daemon.RESULTS_FILE)

            # Write and consume some results
            daemon.transcribe_frames(list(frames), seq_num=1)
            daemon.transcribe_frames(list(frames), seq_num=2)
            consumer.pending_count = 2
            consumer.consume()
            assert consumer.last_result_line == 2

            # Simulate idle: process queue, then truncate
            consumer.route_queue.clear()
            consumer.truncate_if_idle()
            assert consumer.last_result_line == 0

            # New results after truncation
            daemon.transcribe_frames(list(frames), seq_num=3)
            consumer.pending_count = 1
            consumer.consume()
            assert len(consumer.route_queue) == 1
            assert consumer.pending_count == 0
        finally:
            _restore_modules(saved)

    def test_truncation_while_pending_avoidance(self):
        """File should NOT be truncated while results are pending."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            consumer = SimulatedConsumer(daemon.RESULTS_FILE)

            daemon.transcribe_frames(list(frames), seq_num=1)
            consumer.pending_count = 2  # still expecting another result

            consumer.consume()
            assert consumer.pending_count == 1  # decremented by 1

            # Should NOT truncate because pending > 0
            consumer.truncate_if_idle()
            # File should still exist with content
            with open(daemon.RESULTS_FILE) as f:
                lines = f.readlines()
            assert len(lines) == 1  # the one result is still there
        finally:
            _restore_modules(saved)

    def test_truncation_after_all_consumed(self):
        """File should be truncated after all results are consumed and routed."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            consumer = SimulatedConsumer(daemon.RESULTS_FILE)

            daemon.transcribe_frames(list(frames), seq_num=1)
            consumer.pending_count = 1
            consumer.consume()

            # Process the queue
            consumer.route_queue.clear()

            consumer.truncate_if_idle()

            # File should be empty
            with open(daemon.RESULTS_FILE) as f:
                content = f.read()
            assert content == ""
        finally:
            _restore_modules(saved)


class TestPendingCountAccuracy:
    """Issue #4 regression: pendingCount gets stuck positive."""

    def test_pending_count_decrements_correctly(self):
        """Each consumed result should decrement pendingCount by 1."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            consumer = SimulatedConsumer(daemon.RESULTS_FILE)

            for i in range(3):
                daemon.transcribe_frames(list(frames), seq_num=i + 1)
                consumer.pending_count += 1

            assert consumer.pending_count == 3
            consumer.consume()
            assert consumer.pending_count == 0
        finally:
            _restore_modules(saved)

    def test_pending_count_never_goes_negative(self):
        """pendingCount should floor at 0 (max(0, pending - 1))."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            consumer = SimulatedConsumer(daemon.RESULTS_FILE)

            daemon.transcribe_frames(list(frames), seq_num=1)
            consumer.pending_count = 0  # Already 0
            consumer.consume()

            assert consumer.pending_count == 0  # Should not go negative
        finally:
            _restore_modules(saved)

    def test_empty_transcription_still_decrements(self):
        """Empty transcription results should still decrement pendingCount."""
        tmp = tempfile.mkdtemp()
        results_file = os.path.join(tmp, "daemon-results.jsonl")

        # Write an empty transcription result directly
        with open(results_file, "a") as f:
            f.write(json.dumps({"seq": 1, "text": ""}) + "\n")

        consumer = SimulatedConsumer(results_file)
        consumer.pending_count = 1
        consumer.consume()

        assert consumer.pending_count == 0
        assert len(consumer.route_queue) == 0  # empty text not added to queue

    @pytest.mark.stress
    def test_rapid_record_stop_cycles_pending_accurate(self):
        """Rapid record/stop cycles: pendingCount should match reality.

        The transcription threads need the STATE_DIR to persist, so we
        use a more robust temp directory and ensure it stays around.
        We also add a small delay between cycles to let threads start,
        since the transcription lock serializes them.
        """
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            # Ensure the state dir stays around for transcription threads
            os.makedirs(tmp, exist_ok=True)

            consumer = SimulatedConsumer(daemon.RESULTS_FILE)
            num_cycles = 10  # Fewer cycles for reliability

            # Simulate rapid push-to-talk cycles
            for i in range(num_cycles):
                daemon.do_start_recording()
                # Simulate some audio frames
                daemon.rec_frames = [np.zeros(1600, dtype=np.float32)]
                daemon.do_stop_recording()
                consumer.pending_count += 1
                # Small delay to let transcription thread start
                time.sleep(0.05)

            # Wait for all transcription threads to finish.
            # Each transcription takes ~0ms with the mock but is serialized
            # by the lock, so we need to wait for them all to complete.
            deadline = time.time() + 10
            while time.time() < deadline:
                if os.path.exists(daemon.RESULTS_FILE):
                    with open(daemon.RESULTS_FILE) as f:
                        lines = f.readlines()
                    if len(lines) >= num_cycles:
                        break
                time.sleep(0.1)

            consumer.consume()

            # All results should have arrived
            assert consumer.pending_count == 0
            assert len(consumer.route_queue) == num_cycles
        finally:
            _restore_modules(saved)


class TestResultsFileFormat:
    """Test that the JSONL format is correct and parseable."""

    def test_each_line_is_valid_json(self):
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            for i in range(5):
                daemon.transcribe_frames(list(frames), seq_num=i + 1)

            with open(daemon.RESULTS_FILE) as f:
                for line_num, line in enumerate(f, 1):
                    result = json.loads(line.strip())
                    assert "seq" in result
                    assert "text" in result
        finally:
            _restore_modules(saved)

    def test_seq_numbers_are_sequential(self):
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            for i in range(5):
                daemon.transcribe_frames(list(frames), seq_num=i + 1)

            with open(daemon.RESULTS_FILE) as f:
                seqs = [json.loads(line)["seq"] for line in f]
            assert seqs == [1, 2, 3, 4, 5]
        finally:
            _restore_modules(saved)

    def test_newline_terminated(self):
        """Each line should end with a newline."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon(tmp)
        try:
            frames = [np.zeros(1600, dtype=np.float32)]
            daemon.transcribe_frames(list(frames), seq_num=1)

            with open(daemon.RESULTS_FILE, "rb") as f:
                content = f.read()
            assert content.endswith(b"\n")
        finally:
            _restore_modules(saved)
