#!/usr/bin/env python3
"""Tests for anywhere-in-message target routing.

The parser currently only detects targets at the START of the message
("tell firmware check GPIO"). These tests verify that targets can be
detected ANYWHERE in the message using routing preposition patterns:
    - "send to <session>" / "send this to <session>"
    - "route to <session>" / "switch to <session>" / "go to <session>"
    - "talk to <session>"
    - "for <session>" at the end of the message
    - "to the <session> session" mid-sentence

The key distinction: routing patterns use a PREPOSITION + session name,
while mere mentions use the session name as subject/object of a verb.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from parser import classify


class TestAnywhereRouting:
    """Test that routing targets are detected anywhere in the message."""

    def test_target_at_end(self):
        """'check GPIO and send to firmware' should target firmware."""
        result = classify("check GPIO pins and send this to firmware", "built-app", ["firmware", "built-app"])
        assert result["action"] == "switch"
        assert result["target"] == "firmware"

    def test_target_mid_sentence(self):
        """'I want to talk to built-app about this' should target built-app."""
        result = classify("I want to talk to built-app about this bug", None, ["firmware", "built-app"])
        assert result["action"] == "switch"
        assert result["target"] == "built-app"

    def test_switch_to_at_end(self):
        """'do this then switch to firmware' should target firmware."""
        result = classify("check the logs then switch to firmware", "built-app", ["firmware", "built-app"])
        assert result["action"] == "switch"
        assert result["target"] == "firmware"

    def test_no_target_uses_sticky(self):
        """'check the error logs' with sticky target should be content."""
        result = classify("check the error logs", "built-app", ["firmware", "built-app"])
        assert result["action"] == "content"

    def test_no_target_no_sticky(self):
        """'check the error logs' with no sticky should be content (last-active)."""
        result = classify("check the error logs", None, ["firmware", "built-app"])
        assert result["action"] == "content"

    def test_target_at_start_still_works(self):
        """'tell firmware check GPIO' should still work (backward compat)."""
        result = classify("tell firmware check GPIO", "built-app", ["firmware", "built-app"])
        assert result["action"] == "switch"
        assert result["target"] == "firmware"

    def test_mention_not_route(self):
        """'I think firmware has a bug' should NOT route to firmware."""
        result = classify("I think firmware has a bug we need to fix", "built-app", ["firmware", "built-app"])
        assert result["action"] == "content"

    def test_send_to_pattern(self):
        """'send to X' or 'route to X' anywhere."""
        result = classify("please send this to firmware", "built-app", ["firmware", "built-app"])
        assert result["action"] == "switch"
        assert result["target"] == "firmware"

    def test_for_pattern_at_end(self):
        """'this is for firmware' at end."""
        result = classify("check the GPIO pins, this is for firmware", "built-app", ["firmware", "built-app"])
        assert result["action"] == "switch"
        assert result["target"] == "firmware"

    def test_route_to_pattern(self):
        """'route to X' anywhere in text."""
        result = classify("please route this to built-app", "firmware", ["firmware", "built-app"])
        assert result["action"] == "switch"
        assert result["target"] == "built-app"

    def test_go_to_mid_sentence(self):
        """'then go to firmware' mid-sentence."""
        result = classify("finish up and go to firmware", "built-app", ["firmware", "built-app"])
        assert result["action"] == "switch"
        assert result["target"] == "firmware"

    def test_fuzzy_match_in_anywhere_scan(self):
        """Fuzzy matching should work in the anywhere scan too."""
        result = classify("send this to firmwear", "built-app", ["firmware", "built-app"])
        assert result["action"] == "switch"
        assert result["target"] == "firmware"

    def test_self_route_anywhere(self):
        """'send to built-app' when sticky is built-app → self."""
        result = classify("send this to built-app", "built-app", ["firmware", "built-app"])
        assert result["action"] == "self"

    def test_existing_start_patterns_unaffected(self):
        """Existing parse() start patterns should continue working identically."""
        # "ask backend run tests" — start-of-message verb pattern
        result = classify("ask backend run tests", "firmware", ["firmware", "backend"])
        assert result["action"] == "switch"
        assert result["target"] == "backend"
        assert result["text"] == "run tests"

    def test_for_pattern_not_mid_sentence(self):
        """'for firmware' in the middle of a sentence should NOT route.

        Only 'for <session>' at the END of the message is a routing pattern.
        Mid-sentence 'for firmware' could be: 'the fix for firmware is in the PR'.
        """
        result = classify("the fix for firmware is in the pull request", "built-app", ["firmware", "built-app"])
        assert result["action"] == "content"
