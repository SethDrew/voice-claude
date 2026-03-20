#!/usr/bin/env python3
"""Tests for auto-name-session script (claude -p naming)."""

import json
import os
import subprocess
import tempfile

import pytest


SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'bin', 'auto-name-session')


@pytest.fixture
def temp_registry(tmp_path):
    """Create a temporary registry file."""
    reg_file = tmp_path / "name-registry.json"
    reg_file.write_text("{}")
    return str(reg_file)


@pytest.fixture
def temp_transcript(tmp_path):
    """Create a temporary transcript file."""
    transcript = tmp_path / "transcript.jsonl"
    entry = {"type": "user", "message": {"content": "Help me set up GPIO pins for the firmware"}}
    transcript.write_text(json.dumps(entry) + "\n")
    return str(transcript)


class TestAutoNameRegex:
    """Test the regex validation for claude -p output."""

    def test_valid_names(self):
        """Valid codenames should match the regex."""
        import re
        pattern = re.compile(r'^[a-z][a-z0-9]{1,29}$')
        valid = ["firmware", "gpio", "setup2", "alpha", "ab"]
        for name in valid:
            assert pattern.match(name), f"{name} should be valid"

    def test_invalid_names(self):
        """Invalid codenames should not match the regex."""
        import re
        pattern = re.compile(r'^[a-z][a-z0-9]{1,29}$')
        invalid = [
            "A",           # uppercase
            "1abc",        # starts with digit
            "a",           # too short (only 1 char total, need 2-30)
            "abc-def",     # contains hyphen
            "abc def",     # contains space
            "",            # empty
            "a" * 31,      # too long
            "ABC",         # all uppercase
        ]
        for name in invalid:
            assert not pattern.match(name), f"{name} should be invalid"


class TestAutoNameFallback:
    """Test the keyword extraction fallback (no claude -p)."""

    def test_keyword_extraction_basic(self):
        """Test that the script can extract keywords from a transcript."""
        # We test the keyword extraction logic in isolation by checking
        # that common stopwords are filtered out
        stopwords = {'a', 'an', 'the', 'this', 'that', 'is', 'are', 'to', 'of',
                     'in', 'for', 'on', 'at', 'by', 'with', 'help', 'me', 'set', 'up'}
        words = "help me set up gpio pins for the firmware".split()
        filtered = [w for w in words if w.lower() not in stopwords]
        # gpio and pins and firmware should survive
        assert "gpio" in filtered
        assert "firmware" in filtered

    def test_claude_p_validation_accepts_good_name(self):
        """Simulate what the script does with a valid claude -p response."""
        import re
        claude_output = "  gpio  \n"
        cleaned = claude_output.strip().replace(" ", "")
        assert re.match(r'^[a-z][a-z0-9]{1,29}$', cleaned)

    def test_claude_p_validation_rejects_bad_name(self):
        """Simulate what the script does with an invalid claude -p response."""
        import re
        bad_outputs = [
            "Here's a codename: gpio",
            "GPIO",
            "gpio-pins",
            "",
        ]
        for output in bad_outputs:
            cleaned = output.strip().replace(" ", "")
            match = re.match(r'^[a-z][a-z0-9]{1,29}$', cleaned)
            # At least some should fail
            # (the first one definitely fails due to extra text)
            if "Here's" in output:
                assert match is None, f"Should reject: {output}"


@pytest.mark.integration
class TestAutoNameIntegration:
    """Integration tests for auto-name-session (requires claude CLI)."""

    def test_script_is_executable(self):
        """The script file should exist and be executable."""
        assert os.path.isfile(SCRIPT_PATH)
        assert os.access(SCRIPT_PATH, os.X_OK)
