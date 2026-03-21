#!/usr/bin/env python3
"""Tests for routing intelligence (length gate) in the Lua shouldSwitchTarget logic.

Since the routing logic lives in Lua (Hammerspoon init.lua), we test the
equivalent Python-side behavior through the parser, and also test the
shouldSwitchTarget decision function via a Python reimplementation that
mirrors the Lua logic exactly.

Test cases:
  1. Long message (7+ words) with routing verb -> goes to sticky target as content
  2. Short message (<=6 words) with routing verb -> switches target
  3. "go to X" always switches regardless of length
  4. Self-routing stripped: "tell built-app check X" while targeting built-app -> content only
  5. Just session name (1 word) -> switches target
  6. Long message with embedded "tell" -> content to sticky target
"""

import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import patch
from parser import parse, ParsedCommand


# ---------------------------------------------------------------------------
# Python reimplementation of Lua shouldSwitchTarget for testing
# ---------------------------------------------------------------------------

def should_switch_target(text, current_target):
    """Python port of the Lua shouldSwitchTarget function.

    Mirrors the logic exactly so we can unit test the routing decision
    without running Hammerspoon.

    Returns True if the target should be switched (re-resolved),
    False if the text should be sent as content to the current sticky target.
    """
    import re
    lower = text.lower()
    word_count = len(lower.split())

    # Focus verbs ALWAYS switch (regardless of length)
    if (re.match(r'^go\s+to\s', lower) or
        re.match(r'^switch\s+to\s', lower) or
        re.match(r'^focus\s', lower)):
        return True

    # Long messages (7+ words) are ALWAYS content when we have a sticky target
    if word_count >= 7 and current_target:
        return False

    # Short messages: check for routing verb + session name
    if (re.match(r'^tell\s', lower) or
        re.match(r'^ask\s', lower) or
        re.match(r'^send\s', lower) or
        re.match(r'^hey\s', lower) or
        re.match(r'^yo\s', lower) or
        re.match(r'^ping\s', lower) or
        re.match(r'^for\s', lower) or
        re.match(r'^message\s', lower)):
        return True

    # Very short (1-3 words): could be just a session name
    if word_count <= 3 and not current_target:
        return True  # let the resolver figure it out

    return False


def _isolated_parse(text):
    """Parse with LLM router and session registry disabled for unit test isolation."""
    with patch('parser._load_known_sessions', return_value=[]):
        with patch.dict('sys.modules', {'llm_router': None}):
            return parse(text)


def _parse_with_sessions(text, sessions):
    """Parse with specific session names available."""
    with patch('parser._load_known_sessions', return_value=sessions):
        with patch.dict('sys.modules', {'llm_router': None}):
            return parse(text)


# ---------------------------------------------------------------------------
# Tests for shouldSwitchTarget (length gate)
# ---------------------------------------------------------------------------


class TestLongMessageAsContent(unittest.TestCase):
    """Long messages (7+ words) should ALWAYS be sent as content to sticky target."""

    def test_long_with_tell_verb(self):
        """'tell firmware check the GPIO pins and verify the LED driver' (11 words)"""
        text = "tell firmware check the GPIO pins and verify the LED driver"
        result = should_switch_target(text, "firmware")
        self.assertFalse(result, "Long message with routing verb should be content")

    def test_long_with_ask_verb(self):
        """'ask frontend please update the navigation component styles' (8 words)"""
        text = "ask frontend please update the navigation component styles"
        result = should_switch_target(text, "frontend")
        self.assertFalse(result, "Long 'ask' message should be content")

    def test_embedded_tell_in_content(self):
        """'I think we should tell firmware to handle GPIO differently' (10 words)"""
        text = "I think we should tell firmware to handle GPIO differently"
        result = should_switch_target(text, "built-app")
        self.assertFalse(result, "Long message starting with 'I think' should be content")

    def test_exactly_seven_words_with_target(self):
        """Exactly 7 words with a sticky target should be content."""
        text = "tell firmware check the GPIO pins now"
        result = should_switch_target(text, "firmware")
        self.assertFalse(result, "7-word message with sticky target is content")

    def test_long_without_routing_verb(self):
        """Long bare text should be content when sticky target exists."""
        text = "check the GPIO pins and also verify the LED driver"
        result = should_switch_target(text, "firmware")
        self.assertFalse(result, "Long bare text with sticky target is content")


class TestShortMessageSwitchesTarget(unittest.TestCase):
    """Short messages (<=6 words) with routing verb should switch target."""

    def test_short_tell(self):
        """'tell firmware check GPIO' (4 words) should switch."""
        text = "tell firmware check GPIO"
        result = should_switch_target(text, "backend")
        self.assertTrue(result, "Short 'tell' should switch target")

    def test_short_ask(self):
        """'ask frontend add button' (4 words) should switch."""
        text = "ask frontend add button"
        result = should_switch_target(text, "backend")
        self.assertTrue(result, "Short 'ask' should switch target")

    def test_six_words_with_tell(self):
        """'tell firmware check the GPIO pins' (6 words) should switch."""
        text = "tell firmware check the GPIO pins"
        result = should_switch_target(text, "backend")
        self.assertTrue(result, "6-word 'tell' should switch target")

    def test_short_hey(self):
        """'hey firmware check status' should switch."""
        text = "hey firmware check status"
        result = should_switch_target(text, "backend")
        self.assertTrue(result, "Short 'hey' should switch target")

    def test_short_yo(self):
        """'yo backend run tests' should switch."""
        text = "yo backend run tests"
        result = should_switch_target(text, "frontend")
        self.assertTrue(result, "Short 'yo' should switch target")

    def test_short_for(self):
        """'for firmware check GPIO' should switch."""
        text = "for firmware check GPIO"
        result = should_switch_target(text, "backend")
        self.assertTrue(result, "Short 'for' should switch target")


class TestFocusVerbsAlwaysSwitch(unittest.TestCase):
    """Focus verbs (go to, switch to, focus) should ALWAYS switch regardless of length."""

    def test_go_to_short(self):
        """'go to firmware' should switch."""
        text = "go to firmware"
        result = should_switch_target(text, "backend")
        self.assertTrue(result, "'go to' should always switch")

    def test_switch_to_short(self):
        """'switch to frontend' should switch."""
        text = "switch to frontend"
        result = should_switch_target(text, "backend")
        self.assertTrue(result, "'switch to' should always switch")

    def test_focus_short(self):
        """'focus firmware' should switch."""
        text = "focus firmware"
        result = should_switch_target(text, "backend")
        self.assertTrue(result, "'focus' should always switch")

    def test_go_to_long(self):
        """'go to firmware and do something complex with the pins' should still switch."""
        text = "go to firmware and do something complex with the pins"
        result = should_switch_target(text, "backend")
        self.assertTrue(result, "'go to' should switch even with many words")

    def test_switch_to_long(self):
        """Long 'switch to' should still switch."""
        text = "switch to the frontend session for the new project"
        result = should_switch_target(text, "backend")
        self.assertTrue(result, "'switch to' should switch even with many words")


class TestSelfRouting(unittest.TestCase):
    """Self-routing: 'tell X ...' while targeting X should send as content."""

    def test_self_route_short_switches(self):
        """Short self-route 'tell firmware check GPIO' still switches.

        Note: the shouldSwitchTarget function itself doesn't handle
        self-routing stripping. That happens in the parser/router side.
        But a short 'tell X' should switch target, then the parser
        recognizes the self-route.
        """
        text = "tell firmware check GPIO"
        result = should_switch_target(text, "firmware")
        self.assertTrue(result, "Short 'tell' always switches")

    def test_self_route_long_is_content(self):
        """Long 'tell firmware ...' while targeting firmware is content."""
        text = "tell firmware to check the GPIO pins and verify the LED driver status"
        result = should_switch_target(text, "firmware")
        self.assertFalse(result, "Long message to same target is content")


class TestJustSessionName(unittest.TestCase):
    """Just a session name (1-3 words) should switch target when no target set."""

    def test_single_word_no_target(self):
        """'firmware' with no target should try to resolve."""
        text = "firmware"
        result = should_switch_target(text, None)
        self.assertTrue(result, "Single word without target should resolve")

    def test_two_words_no_target(self):
        """'built app' with no target should try to resolve."""
        text = "built app"
        result = should_switch_target(text, None)
        self.assertTrue(result, "Two words without target should resolve")

    def test_single_word_with_target(self):
        """'firmware' with existing target should NOT switch."""
        text = "firmware"
        result = should_switch_target(text, "backend")
        self.assertFalse(result, "Single word with sticky target is content")

    def test_three_words_no_target(self):
        """'run the tests' with no target."""
        text = "run the tests"
        result = should_switch_target(text, None)
        self.assertTrue(result, "Three words without target should try resolver")


class TestBareContentToStickyTarget(unittest.TestCase):
    """Bare text (no routing verb) should go to sticky target."""

    def test_medium_bare_text_with_target(self):
        """'check the logs' (3 words) with a target: is this content or switch?

        With 3 words and a sticky target, there's no routing verb,
        so it stays as content.
        """
        text = "check the logs"
        result = should_switch_target(text, "firmware")
        self.assertFalse(result, "Bare text with sticky target is content")

    def test_five_word_bare_text_with_target(self):
        """'check the logs for errors' (5 words) with target is content."""
        text = "check the logs for errors"
        result = should_switch_target(text, "firmware")
        self.assertFalse(result, "5-word bare text with sticky target is content")


# ---------------------------------------------------------------------------
# Tests for parser integration with routing intelligence
# ---------------------------------------------------------------------------


class TestParserWithStickyTarget(unittest.TestCase):
    """Test that the parser correctly handles commands in routing context."""

    def test_parser_extracts_target_from_tell(self):
        """Parser should extract target from 'tell firmware check GPIO'."""
        cmd = _isolated_parse("tell firmware check GPIO")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")

    def test_parser_focus_verb(self):
        """Parser should handle 'go to firmware' as focus-only."""
        cmd = _isolated_parse("go to firmware")
        self.assertEqual(cmd.target, "firmware")
        self.assertIsNone(cmd.text)

    def test_parser_bare_text_no_target(self):
        """Bare text should have no target."""
        cmd = _isolated_parse("check the GPIO pins and verify the LED driver")
        self.assertIsNone(cmd.target)
        self.assertIsNotNone(cmd.text)

    def test_parser_self_route_with_known_sessions(self):
        """When target matches a known session, parser should extract correctly."""
        cmd = _parse_with_sessions("tell firmware check GPIO", ["firmware", "frontend"])
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")


class TestWordCountEdgeCases(unittest.TestCase):
    """Test edge cases in the word count logic."""

    def test_empty_string(self):
        """Empty string should not switch."""
        result = should_switch_target("", "firmware")
        self.assertFalse(result)

    def test_single_routing_verb(self):
        """Just 'tell' should switch (1 word)."""
        result = should_switch_target("tell", "firmware")
        # 'tell' alone doesn't match '^tell\s' since there's no space after
        self.assertFalse(result)

    def test_long_without_target(self):
        """Long message without a current target should not trigger the 7-word gate."""
        text = "I think we should check the firmware and also verify things"
        result = should_switch_target(text, None)
        # No current_target, so the 7-word gate doesn't apply
        # Also no routing verb match, and not <= 3 words
        self.assertFalse(result)


class TestProcessQueueLogic(unittest.TestCase):
    """Test the processQueue decision tree that uses shouldSwitchTarget.

    This tests the logic flow:
    - With pendingTarget: shouldSwitchTarget decides route vs content
    - Without pendingTarget: resolve from speech
    """

    def test_with_target_long_message_routes_as_content(self):
        """With a pending target, long message should be content."""
        text = "I need you to check all the GPIO pins and update the configuration"
        # The processQueue would call shouldSwitchTarget(text, pendingTarget)
        self.assertFalse(should_switch_target(text, "firmware"))

    def test_with_target_short_tell_switches(self):
        """With a pending target, 'tell backend run tests' should switch."""
        text = "tell backend run tests"
        self.assertTrue(should_switch_target(text, "firmware"))

    def test_without_target_short_resolves(self):
        """Without a pending target, short text should resolve."""
        text = "firmware"
        self.assertTrue(should_switch_target(text, None))

    def test_without_target_long_bare_text(self):
        """Without a pending target, long bare text is ambiguous."""
        text = "check all the pins and verify the LED driver status"
        result = should_switch_target(text, None)
        # No routing verb, no target, > 3 words — returns False
        # processQueue would still call resolveTarget for this case
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
