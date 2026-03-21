#!/usr/bin/env python3
"""Tests for the v2 pipeline: daemon partial file -> Hammerspoon routing.

Covers the v2 protocol where:
  - Daemon writes daemon-partial.json with {text, final, ts}
  - Hammerspoon polls the file, shows live text, routes on final:true
  - voice-route CLI delivers text to iTerm2 sessions

Bug under investigation: live streaming text shows in overlay but
routing never delivers text to a Claude Code session.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Helpers: import listen_daemon_v2 with mocked Moonshine dependency
# ---------------------------------------------------------------------------

class _FakeMicTranscriber:
    """Mock MicTranscriber that records method calls."""
    def __init__(self, **kwargs):
        self._listeners = []
        self._started = False
        self._closed = False

    def add_listener(self, listener):
        self._listeners.append(listener)

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def close(self):
        self._closed = True


class _FakeMoonshineVoice:
    MicTranscriber = _FakeMicTranscriber

    @staticmethod
    def get_model_for_language(lang):
        return ("fake_model_path", "fake_arch")


class _FakeTranscriberModule:
    class TranscriptEventListener:
        def on_line_started(self, event): pass
        def on_line_text_changed(self, event): pass
        def on_line_completed(self, event): pass


def _import_daemon_v2(tmp_dir):
    """Import listen_daemon_v2 with mocked dependencies and redirected paths."""
    import importlib

    saved = {}
    for mod_name in ("moonshine_voice", "moonshine_voice.transcriber"):
        saved[mod_name] = sys.modules.get(mod_name)

    sys.modules["moonshine_voice"] = _FakeMoonshineVoice()
    sys.modules["moonshine_voice.transcriber"] = _FakeTranscriberModule()

    if "listen_daemon_v2" in sys.modules:
        del sys.modules["listen_daemon_v2"]

    import listen_daemon_v2

    # Override file paths to use temp directory
    listen_daemon_v2.STATE_DIR = tmp_dir
    listen_daemon_v2.STATE_FILE = os.path.join(tmp_dir, "daemon-state.json")
    listen_daemon_v2.PID_FILE = os.path.join(tmp_dir, "listen-daemon.pid")
    listen_daemon_v2.RESULTS_FILE = os.path.join(tmp_dir, "daemon-results.jsonl")
    listen_daemon_v2.PARTIAL_FILE = os.path.join(tmp_dir, "daemon-partial.json")

    # Reset global state
    listen_daemon_v2.recording = False
    listen_daemon_v2.seq = 0
    listen_daemon_v2._live_text = ""
    listen_daemon_v2._finalized_lines = []

    # Create mock mic
    listen_daemon_v2._mic = _FakeMicTranscriber()

    return listen_daemon_v2, saved


def _restore_modules(saved):
    for mod_name, mod in saved.items():
        if mod is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = mod
    sys.modules.pop("listen_daemon_v2", None)


# ---------------------------------------------------------------------------
# Test 1: Daemon writes final:true to partial file on stop
# ---------------------------------------------------------------------------

class TestDaemonWritesFinalOnStop:
    """After do_stop_recording, daemon-partial.json should have final:true."""

    def test_daemon_writes_final_on_stop(self):
        """After recording start+stop, daemon-partial.json has final:true with text."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            # Simulate some transcription text
            daemon._live_text = "hello world"
            daemon._finalized_lines = ["hello", "world"]
            daemon.recording = True

            daemon.do_stop_recording()

            # Read the partial file
            partial_path = os.path.join(tmp, "daemon-partial.json")
            assert os.path.exists(partial_path), "daemon-partial.json should exist"

            with open(partial_path) as f:
                data = json.load(f)

            assert data["final"] is True, "final flag should be True after stop"
            assert data["text"] == "hello world", "text should match live text"
        finally:
            _restore_modules(saved)

    def test_final_text_is_stripped(self):
        """Final text should be stripped of whitespace."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            daemon._live_text = "  hello world  "
            daemon._finalized_lines = ["hello", "world"]
            daemon.recording = True

            daemon.do_stop_recording()

            partial_path = os.path.join(tmp, "daemon-partial.json")
            with open(partial_path) as f:
                data = json.load(f)

            assert data["text"] == "hello world"
            assert data["final"] is True
        finally:
            _restore_modules(saved)


# ---------------------------------------------------------------------------
# Test 2: Partial file has valid JSON at all times
# ---------------------------------------------------------------------------

class TestPartialFileAlwaysValidJson:
    """The partial file should never be half-written or corrupt."""

    def test_partial_write_produces_valid_json(self):
        """write_partial should produce a valid JSON file."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            daemon.write_partial("test text", final=False)

            partial_path = os.path.join(tmp, "daemon-partial.json")
            with open(partial_path) as f:
                data = json.load(f)

            assert data["text"] == "test text"
            assert data["final"] is False
            assert "ts" in data
        finally:
            _restore_modules(saved)

    def test_write_partial_does_not_crash(self):
        """write_partial should not raise exceptions."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            # This should NOT raise — if it does, the duplicate os.replace bug is present
            daemon.write_partial("test", final=False)
            daemon.write_partial("test two", final=False)
            daemon.write_partial("final text", final=True)
        finally:
            _restore_modules(saved)

    def test_rapid_partial_writes_always_valid(self):
        """Rapid sequential writes should always leave valid JSON."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            partial_path = os.path.join(tmp, "daemon-partial.json")
            for i in range(50):
                daemon.write_partial(f"word {i}", final=False)
                with open(partial_path) as f:
                    data = json.load(f)
                assert data["text"] == f"word {i}"
                assert data["final"] is False
        finally:
            _restore_modules(saved)

    def test_concurrent_partial_writes_dont_corrupt(self):
        """Multiple threads writing partials should not corrupt the file."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            partial_path = os.path.join(tmp, "daemon-partial.json")
            errors = []

            def writer(thread_id):
                for i in range(20):
                    try:
                        daemon.write_partial(f"thread-{thread_id}-{i}", final=False)
                    except Exception as e:
                        errors.append(e)

            threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"write_partial raised exceptions: {errors}"

            # Final file should be valid JSON
            with open(partial_path) as f:
                data = json.load(f)
            assert "text" in data
            assert "final" in data
        finally:
            _restore_modules(saved)


# ---------------------------------------------------------------------------
# Test 3: Final text matches what was transcribed
# ---------------------------------------------------------------------------

class TestFinalTextMatchesLiveText:
    """The final:true text should match the last live text."""

    def test_final_text_matches_live_text(self):
        """Final text in partial file should match the accumulated live text."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            # Simulate a recording session with text updates
            daemon.recording = True
            daemon._live_text = "the quick brown fox"
            daemon._finalized_lines = ["the", "quick", "brown", "fox"]

            # Write a live partial first
            daemon.write_partial("the quick brown fox", final=False)

            # Stop recording
            daemon.do_stop_recording()

            # Check final
            partial_path = os.path.join(tmp, "daemon-partial.json")
            with open(partial_path) as f:
                data = json.load(f)

            assert data["final"] is True
            assert data["text"] == "the quick brown fox"
        finally:
            _restore_modules(saved)

    def test_empty_text_filtered(self):
        """Empty/hallucination text should be filtered to empty string."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            daemon.recording = True
            daemon._live_text = "thank you"
            daemon._finalized_lines = ["thank you"]

            daemon.do_stop_recording()

            partial_path = os.path.join(tmp, "daemon-partial.json")
            with open(partial_path) as f:
                data = json.load(f)

            # Hallucination should be filtered
            assert data["final"] is True
            assert data["text"] == ""
        finally:
            _restore_modules(saved)


# ---------------------------------------------------------------------------
# Test 4: voice-route --text delivers to a session
# ---------------------------------------------------------------------------

class TestVoiceRouteDelivers:
    """voice-route --text should succeed when sessions exist."""

    @pytest.mark.integration
    def test_voice_route_text_delivers(self):
        """voice-route --text should succeed when sessions exist."""
        # First check if voice-route is available and sessions exist
        result = subprocess.run(
            ["voice-route", "--list"],
            capture_output=True, text=True, timeout=10
        )
        if "No cc sessions found" in result.stdout or result.returncode != 0:
            pytest.skip("No active sessions to test routing")

        # Try routing
        result = subprocess.run(
            ["voice-route", "--text", "test message from v2 pipeline TDD"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0, f"voice-route failed: {result.stderr}"
        assert "→" in result.stdout, f"Expected routing output, got: {result.stdout}"

    @pytest.mark.integration
    def test_voice_route_list_shows_sessions(self):
        """voice-route --list should show active sessions."""
        result = subprocess.run(
            ["voice-route", "--list"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0, f"voice-route --list failed: {result.stderr}"


# ---------------------------------------------------------------------------
# Test 5: Routing logic fires on final (simulate Hammerspoon)
# ---------------------------------------------------------------------------

class TestRoutingLogicFiresOnFinal:
    """Simulate what Hammerspoon does: read partial, detect final, route."""

    def _simulate_hammerspoon_poll(self, partial_path):
        """Simulate the Lua pollPartial logic in Python.

        Returns (should_route, text) tuple.
        """
        if not os.path.exists(partial_path):
            return (False, None)

        with open(partial_path) as f:
            raw = f.read()

        if not raw:
            return (False, None)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return (False, None)

        text = data.get("text", "")
        final = data.get("final", False)

        if final and text and len(text) > 0:
            return (True, text)

        return (False, None)

    def test_routing_fires_on_final_true(self):
        """When partial file has final:true and text, routing should fire."""
        tmp = tempfile.mkdtemp()
        partial_path = os.path.join(tmp, "daemon-partial.json")

        # Write a final partial
        with open(partial_path, "w") as f:
            json.dump({"text": "tell firmware check the status", "final": True, "ts": time.time()}, f)

        should_route, text = self._simulate_hammerspoon_poll(partial_path)
        assert should_route is True
        assert text == "tell firmware check the status"

    def test_routing_does_not_fire_on_non_final(self):
        """When partial file has final:false, routing should NOT fire."""
        tmp = tempfile.mkdtemp()
        partial_path = os.path.join(tmp, "daemon-partial.json")

        with open(partial_path, "w") as f:
            json.dump({"text": "in progress...", "final": False, "ts": time.time()}, f)

        should_route, text = self._simulate_hammerspoon_poll(partial_path)
        assert should_route is False

    def test_routing_does_not_fire_on_empty_text(self):
        """When final:true but text is empty, routing should NOT fire."""
        tmp = tempfile.mkdtemp()
        partial_path = os.path.join(tmp, "daemon-partial.json")

        with open(partial_path, "w") as f:
            json.dump({"text": "", "final": True, "ts": time.time()}, f)

        should_route, text = self._simulate_hammerspoon_poll(partial_path)
        assert should_route is False

    def test_routing_with_daemon_write_partial(self):
        """Use the actual daemon write_partial function and verify routing logic."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            partial_path = os.path.join(tmp, "daemon-partial.json")

            # Write final partial via daemon
            daemon.write_partial("run the tests", final=True)

            should_route, text = self._simulate_hammerspoon_poll(partial_path)
            assert should_route is True, "Routing should fire on final:true from write_partial"
            assert text == "run the tests"
        finally:
            _restore_modules(saved)


# ---------------------------------------------------------------------------
# Test 6: Full pipeline simulation
# ---------------------------------------------------------------------------

class TestFullPipelineSimulation:
    """Full pipeline: start -> record -> stop -> final text -> routing check."""

    def test_full_pipeline_signal_to_route(self):
        """Simulate: USR1 -> text arrives -> USR2 -> final text -> routing fires."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            partial_path = os.path.join(tmp, "daemon-partial.json")

            # Step 1: Start recording
            daemon.do_start_recording()
            assert daemon.recording is True

            # Step 2: Simulate transcription callbacks
            daemon._live_text = "deploy the firmware update"
            daemon._finalized_lines = ["deploy", "the", "firmware", "update"]
            daemon.write_partial("deploy the firmware update", final=False)

            # Verify live partial
            with open(partial_path) as f:
                data = json.load(f)
            assert data["final"] is False
            assert data["text"] == "deploy the firmware update"

            # Step 3: Stop recording
            daemon.do_stop_recording()
            assert daemon.recording is False

            # Step 4: Verify final partial
            with open(partial_path) as f:
                data = json.load(f)
            assert data["final"] is True, "Partial file should have final:true after stop"
            assert data["text"] == "deploy the firmware update"

            # Step 5: Simulate Hammerspoon reading and routing
            should_route = data["final"] and data["text"] and len(data["text"]) > 0
            assert should_route is True, "Hammerspoon should detect final and trigger routing"

        finally:
            _restore_modules(saved)

    def test_full_pipeline_results_file_also_written(self):
        """Stop recording should also write to results JSONL file."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            daemon.do_start_recording()
            daemon._live_text = "check the logs"
            daemon._finalized_lines = ["check", "the", "logs"]

            daemon.do_stop_recording()

            results_path = os.path.join(tmp, "daemon-results.jsonl")
            assert os.path.exists(results_path), "Results JSONL should exist"

            with open(results_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            result = json.loads(lines[0])
            assert result["text"] == "check the logs"
            assert result["seq"] == 1
        finally:
            _restore_modules(saved)

    def test_state_file_updated_after_stop(self):
        """write_state should be called after write_partial (no crash)."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            daemon.do_start_recording()
            daemon._live_text = "hello"
            daemon._finalized_lines = ["hello"]

            daemon.do_stop_recording()

            state_path = os.path.join(tmp, "daemon-state.json")
            with open(state_path) as f:
                state = json.load(f)
            assert state["state"] == "done", \
                "State should be 'done' after stop (write_partial must not crash before write_state)"
            assert state["text"] == "hello"
        finally:
            _restore_modules(saved)

    def test_multiple_record_stop_cycles(self):
        """Multiple recording cycles should all produce correct final partials."""
        tmp = tempfile.mkdtemp()
        daemon, saved = _import_daemon_v2(tmp)
        try:
            partial_path = os.path.join(tmp, "daemon-partial.json")

            for i in range(5):
                daemon.do_start_recording()
                daemon._live_text = f"message number {i}"
                daemon._finalized_lines = [f"message number {i}"]

                daemon.do_stop_recording()

                with open(partial_path) as f:
                    data = json.load(f)
                assert data["final"] is True
                assert data["text"] == f"message number {i}"
        finally:
            _restore_modules(saved)


# ---------------------------------------------------------------------------
# Test 7: Lua forward-reference detection
# ---------------------------------------------------------------------------

class TestLuaForwardReferences:
    """Test that the Lua code doesn't have forward-reference issues.

    We can't run Lua directly, but we can parse the file and verify
    that local functions are declared before they're referenced.
    """

    def _get_lua_path(self):
        """Get path to the Hammerspoon init.lua."""
        return os.path.join(
            os.path.dirname(__file__), '..', 'config', 'hammerspoon', 'init.lua'
        )

    def _parse_lua_locals_and_refs(self, lua_source):
        """Parse Lua source to find local function declarations and references.

        Uses nesting depth tracking to properly identify function bodies.
        In Lua, 'if/for/while/function/do' open a block, and 'end' closes one.

        Returns (declarations, references) where:
          declarations = {name: line_number}
          references = [(name, line_number, calling_function)]
        """
        import re

        declarations = {}
        references = []
        current_function = None
        # Track nesting depth within the current function
        # When depth drops to 0, we've exited the function
        func_depth = 0

        for i, line in enumerate(lua_source.split('\n'), 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith('--'):
                continue

            # Track local function declarations (top-level)
            m = re.match(r'^local\s+function\s+(\w+)\s*\(', line)
            if m:
                name = m.group(1)
                declarations[name] = i
                current_function = name
                func_depth = 1  # The function itself is depth 1
                continue

            if current_function:
                # Count block openers on this line
                # Match: if...then, for...do, while...do, function(, do (standalone)
                openers = len(re.findall(r'\bif\b.*\bthen\b', stripped))
                openers += len(re.findall(r'\bfor\b.*\bdo\b', stripped))
                openers += len(re.findall(r'\bwhile\b.*\bdo\b', stripped))
                openers += len(re.findall(r'\bfunction\s*\(', stripped))
                # 'do' as standalone block (not part of for/while)
                if re.match(r'^do\b', stripped):
                    openers += 1

                # Count 'end' closers on this line
                closers = len(re.findall(r'\bend\b', stripped))

                func_depth += openers - closers

                if func_depth <= 0:
                    current_function = None
                    func_depth = 0
                    continue

                # Track function calls within function bodies
                for m in re.finditer(r'\b(\w+)\s*\(', line):
                    called = m.group(1)
                    # Skip Lua keywords and built-ins
                    if called in ('if', 'for', 'while', 'function', 'return',
                                  'local', 'not', 'and', 'or', 'pcall', 'type',
                                  'print', 'require', 'pairs', 'ipairs', 'tostring',
                                  'tonumber', 'math', 'string', 'table', 'io', 'os',
                                  'error', 'assert', 'select', 'next'):
                        continue
                    references.append((called, i, current_function))

        return declarations, references

    def test_no_forward_references_to_local_functions(self):
        """Local functions should be declared before they are called.

        In Lua, 'local function foo()' creates a local that is only
        visible AFTER the declaration. If pollPartial() references
        routeFinalText() but routeFinalText is declared later as a local,
        Lua will look it up as a global (which is nil).
        """
        lua_path = self._get_lua_path()
        if not os.path.exists(lua_path):
            pytest.skip("init.lua not found")

        with open(lua_path) as f:
            source = f.read()

        declarations, references = self._parse_lua_locals_and_refs(source)

        forward_refs = []
        for called_name, call_line, caller_func in references:
            if called_name in declarations:
                decl_line = declarations[called_name]
                # The call is inside a function that's defined BEFORE the callee
                caller_decl = declarations.get(caller_func, 0)
                if caller_decl < decl_line:
                    # The calling function is declared before the callee
                    # This means the callee's local isn't in scope
                    forward_refs.append(
                        f"  {caller_func} (line {caller_decl}) calls "
                        f"{called_name} (declared line {decl_line})"
                    )

        assert len(forward_refs) == 0, (
            "Found forward references to local functions in init.lua:\n"
            + "\n".join(forward_refs)
            + "\n\nIn Lua, local functions must be declared before they "
            "are referenced. Move declarations above their callers, or "
            "use forward declarations (local routeFinalText; ... routeFinalText = function() ... end)."
        )
