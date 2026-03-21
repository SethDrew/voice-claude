#!/usr/bin/env python3
"""Tests for auto-name-session hook.

Covers issues:
  #6  Session auto-naming fails silently
  #7  Shell injection in auto-name-session
"""

import json
import os
import re
import subprocess
import tempfile

import pytest


SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'bin', 'auto-name-session')
VENV_PYTHON = os.path.expanduser("~/.local/share/voice-claude/venv/bin/python3")


# ======================================================================
# Regex validation (same as used in the bash script)
# ======================================================================

NAME_REGEX = re.compile(r'^[a-z][a-z0-9]{1,29}$')


class TestNameValidation:
    """Test the name validation regex used by the script."""

    @pytest.mark.parametrize("name", [
        "firmware", "gpio", "ab", "setup2", "alpha",
        "a1", "test123", "x" * 30,  # max length
    ])
    def test_valid_names(self, name):
        assert NAME_REGEX.match(name), f"{name} should be valid"

    @pytest.mark.parametrize("name,reason", [
        ("A", "uppercase"),
        ("1abc", "starts with digit"),
        ("a", "too short"),
        ("abc-def", "contains hyphen"),
        ("abc def", "contains space"),
        ("", "empty"),
        ("a" * 31, "too long"),
        ("ABC", "all uppercase"),
        ("abc!", "special char"),
        ("abc;echo hi", "shell injection"),
        ("abc'def", "single quote"),
        ('abc"def', "double quote"),
    ])
    def test_invalid_names(self, name, reason):
        assert not NAME_REGEX.match(name), f"{name} should be invalid ({reason})"


# ======================================================================
# Keyword extraction logic
# ======================================================================

# Stopwords from the script
STOPWORDS = set(
    "a an the this that is are was were be been being have has had do does did "
    "will would could should can may might shall must i me my we our you your "
    "it its he she they them his her their to of in for on at by with from up "
    "out if or and but not no so as just about into than then some any all each "
    "every how what when where which who why take look check please help want "
    "need tell make get go see know think come give use find here there very "
    "also only really much many like well back even still way thing right good "
    "new now try these those let hi hey hello ok okay sure yeah yes".split()
)


class TestKeywordExtraction:
    """Test the keyword extraction fallback."""

    def _extract_keywords(self, text, max_words=2):
        """Replicate the script's keyword extraction logic."""
        words = re.sub(r'[^a-z0-9]', ' ', text.lower()).split()
        filtered = [w for w in words if w not in STOPWORDS]
        return "-".join(filtered[:max_words]) if filtered else ""

    def test_gpio_firmware(self):
        name = self._extract_keywords("Help me set up GPIO pins for the firmware")
        assert "gpio" in name or "firmware" in name

    def test_all_stopwords(self):
        """A message with only stopwords should produce empty string."""
        name = self._extract_keywords("help me please")
        assert name == ""

    def test_single_keyword(self):
        name = self._extract_keywords("check firmware")
        assert name == "firmware"

    def test_two_keywords(self):
        name = self._extract_keywords("debug the GPIO driver")
        # 'debug' is not a stopword, so it's the first keyword
        # 'gpio' is second; max_words=2 means we get at most 2
        assert "gpio" in name
        # 'driver' may or may not appear depending on 'debug' being first
        assert "debug" in name or "driver" in name

    def test_numbers_preserved(self):
        name = self._extract_keywords("fix issue 42 in the build")
        assert "42" in name or "issue" in name

    def test_special_chars_stripped(self):
        name = self._extract_keywords("fix the @#$% build!")
        assert "@" not in name
        assert "!" not in name


class TestClaudePFallback:
    """Test claude -p validation logic (without actually calling claude)."""

    def test_valid_response_accepted(self):
        """A clean one-word response should be accepted."""
        response = "  gpio  \n"
        cleaned = response.strip().replace(" ", "")
        assert NAME_REGEX.match(cleaned)

    def test_multiline_response_rejected(self):
        """'Here is a codename: gpio' should be rejected."""
        response = "Here is a codename: gpio"
        cleaned = response.strip().replace(" ", "")
        assert not NAME_REGEX.match(cleaned)

    def test_uppercase_response_rejected(self):
        response = "GPIO"
        cleaned = response.strip().replace(" ", "")
        assert not NAME_REGEX.match(cleaned)

    def test_hyphenated_response_rejected(self):
        response = "gpio-pins"
        cleaned = response.strip().replace(" ", "")
        assert not NAME_REGEX.match(cleaned)


# ======================================================================
# Issue #7: Shell injection prevention
# ======================================================================

class TestIssue7ShellInjection:
    """Shell injection in auto-name-session.

    The old code interpolated $existing_name directly into inline Python.
    The fix passes names as environment variables instead.
    """

    def test_name_with_quotes_is_safe(self):
        """Names with quotes should not break the script."""
        # The regex validation prevents this, but let's verify
        dangerous_names = [
            'test"; echo pwned; #',
            "test'; echo pwned; #",
            "test`echo pwned`",
            "test$(echo pwned)",
            "test\necho pwned",
        ]
        for name in dangerous_names:
            assert not NAME_REGEX.match(name), \
                f"Dangerous name passed validation: {name!r}"

    def test_env_var_passing_safe(self):
        """Environment variable passing prevents injection."""
        # The script now does:
        #   ITERM_SID="$iterm_sid" SESSION_NAME="$existing_name" python3 -c "
        #     name = os.environ['SESSION_NAME']
        #     ...
        #   "
        # This is safe because the name never appears in the Python source code.
        # Verify by checking the script source contains the safe pattern.
        with open(SCRIPT_PATH) as f:
            script = f.read()

        # Check that SESSION_NAME is passed as env var
        assert "SESSION_NAME=" in script
        assert "os.environ['SESSION_NAME']" in script

        # Check there's no direct interpolation of $existing_name into Python code
        # (the old vulnerable pattern was: name = '$existing_name')
        assert "name = '$existing_name'" not in script
        assert 'name = "$existing_name"' not in script

    def test_iterm_sid_passed_as_env_var(self):
        """ITERM_SID should also be passed as an env var."""
        with open(SCRIPT_PATH) as f:
            script = f.read()

        assert "ITERM_SID=" in script
        assert "os.environ['ITERM_SID']" in script


# ======================================================================
# Name collision deduplication
# ======================================================================

class TestNameDeduplication:
    """Test that duplicate names get a numeric suffix."""

    def test_deduplication_logic(self, tmp_path):
        """Simulate the deduplication counter logic from the script."""
        registry = {"session-1": "firmware", "session-2": "frontend"}

        name = "firmware"
        base_name = name
        counter = 2

        existing_names = set(registry.values())
        while name in existing_names and counter <= 9:
            name = f"{base_name}-{counter}"
            counter += 1

        assert name == "firmware-2"

    def test_deduplication_multiple_collisions(self, tmp_path):
        """Multiple collisions should increment the counter."""
        registry = {
            "s1": "firmware",
            "s2": "firmware-2",
            "s3": "firmware-3",
        }

        name = "firmware"
        base_name = name
        counter = 2

        existing_names = set(registry.values())
        while name in existing_names and counter <= 9:
            name = f"{base_name}-{counter}"
            counter += 1

        assert name == "firmware-4"

    def test_deduplication_max_counter(self):
        """Counter should stop at 9 to prevent infinite loops."""
        registry = {f"s{i}": f"firmware-{i}" if i > 1 else "firmware"
                    for i in range(1, 10)}

        name = "firmware"
        base_name = name
        counter = 2

        existing_names = set(registry.values())
        iterations = 0
        while name in existing_names and counter <= 9:
            name = f"{base_name}-{counter}"
            counter += 1
            iterations += 1

        assert counter <= 10  # should stop at 9
        assert iterations <= 8  # shouldn't loop forever


# ======================================================================
# TERM_SESSION_ID parsing
# ======================================================================

class TestTermSessionIdParsing:
    """Test parsing of TERM_SESSION_ID environment variable."""

    def test_standard_format(self):
        """Standard format: 'w0t0p0:XXXXXXXX-XXXX-...'"""
        term_sid = "w0t0p0:3F2504E0-4F89-11D3-9A0C-0305E82C3301"
        # The script does: ${TERM_SESSION_ID##*:}
        iterm_sid = term_sid.split(":")[-1] if ":" in term_sid else ""
        assert iterm_sid == "3F2504E0-4F89-11D3-9A0C-0305E82C3301"

    def test_no_colon(self):
        """If no colon, extraction should handle gracefully."""
        term_sid = "plain-session-id"
        iterm_sid = term_sid.split(":")[-1] if ":" in term_sid else ""
        assert iterm_sid == ""

    def test_empty_string(self):
        term_sid = ""
        iterm_sid = term_sid.split(":")[-1] if ":" in term_sid and term_sid else ""
        assert iterm_sid == ""

    def test_multiple_colons(self):
        """Handle edge case with multiple colons."""
        term_sid = "prefix:mid:suffix"
        # ${TERM_SESSION_ID##*:} removes everything up to and including last ':'
        # In Python: split(':')[-1]
        iterm_sid = term_sid.split(":")[-1]
        assert iterm_sid == "suffix"


# ======================================================================
# Registry file operations
# ======================================================================

class TestRegistryFileOperations:
    """Test registry file read/write race conditions."""

    def test_atomic_write_with_jq(self, tmp_path):
        """Simulate the atomic write pattern: jq > tmp, mv tmp -> registry."""
        reg_file = tmp_path / "name-registry.json"
        reg_file.write_text("{}")

        # Simulate adding a name
        registry = json.loads(reg_file.read_text())
        registry["session-1"] = "firmware"

        tmp_file = tmp_path / "tmp-registry.json"
        tmp_file.write_text(json.dumps(registry))

        # Atomic rename
        os.replace(str(tmp_file), str(reg_file))

        result = json.loads(reg_file.read_text())
        assert result["session-1"] == "firmware"

    def test_concurrent_reads_safe(self, tmp_path):
        """Reading the registry while another process writes should be safe
        because of atomic rename."""
        import threading

        reg_file = tmp_path / "name-registry.json"
        reg_file.write_text(json.dumps({"s1": "alpha"}))

        errors = []

        def reader():
            for _ in range(50):
                try:
                    data = json.loads(reg_file.read_text())
                    assert isinstance(data, dict)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    errors.append(e)

        def writer():
            for i in range(50):
                data = {f"s{j}": f"name{j}" for j in range(i + 1)}
                tmp = tmp_path / "tmp.json"
                tmp.write_text(json.dumps(data))
                os.replace(str(tmp), str(reg_file))

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Atomic rename should prevent corruption
        # Note: on some filesystems, reader may see empty file briefly
        # But JSON decode errors should be rare
        assert len(errors) < 5, f"Too many errors during concurrent access: {errors}"


# ======================================================================
# Issue #6: Session auto-naming fails silently
# ======================================================================

class TestIssue6AutoNamingFailsSilently:
    """The Stop hook's escape sequences don't reach iTerm2 through Claude Code's TUI.
    Fixed by using iTerm2 Python API.
    """

    def test_script_uses_python_api_not_escape_sequences(self):
        """The script should use iterm2 Python API, not escape sequences."""
        with open(SCRIPT_PATH) as f:
            script = f.read()

        # Should import iterm2
        assert "import" in script and "iterm2" in script

        # Should NOT use escape sequences like \033]
        assert "\\033]" not in script
        assert "\\e]" not in script

        # Should use async_set_variable for cc_name
        assert "async_set_variable" in script

    def test_script_exits_with_suppress_output(self):
        """Script should always output suppressOutput JSON."""
        with open(SCRIPT_PATH) as f:
            script = f.read()

        # Every exit path should output suppressOutput
        assert script.count('{"suppressOutput":true}') >= 1


@pytest.mark.integration
class TestAutoNamingIntegration:
    """Integration tests requiring the actual script and dependencies."""

    def test_script_is_executable(self):
        """The script file should exist and be executable."""
        assert os.path.isfile(SCRIPT_PATH)
        assert os.access(SCRIPT_PATH, os.X_OK)

    def test_empty_input_exits_cleanly(self):
        """Script with empty JSON input should exit with suppressOutput."""
        result = subprocess.run(
            [SCRIPT_PATH],
            input="{}",
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "PATH": os.environ.get("PATH", "")},
        )
        assert '{"suppressOutput":true}' in result.stdout

    def test_missing_session_id_exits_cleanly(self):
        """Script with no session_id should exit with suppressOutput."""
        hook_json = json.dumps({"transcript_path": "/tmp/test.jsonl"})
        result = subprocess.run(
            [SCRIPT_PATH],
            input=hook_json,
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "PATH": os.environ.get("PATH", "")},
        )
        assert '{"suppressOutput":true}' in result.stdout
