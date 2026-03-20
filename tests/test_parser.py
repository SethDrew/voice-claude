#!/usr/bin/env python3
"""Unit tests for the voice command parser."""

import sys
import os
import unittest

# Add src to path so we can import parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import patch

from parser import parse, strip_wake_phrase, replace_slash_commands, ParsedCommand, _load_known_sessions, _fuzzy_session_match


class TestStripWakePhrase(unittest.TestCase):
    def test_hey_skynet(self):
        self.assertEqual(strip_wake_phrase("hey skynet, check the logs"), "check the logs")

    def test_hey_destroyer(self):
        self.assertEqual(strip_wake_phrase("Hey Destroyer: do something"), "do something")

    def test_hey_code(self):
        self.assertEqual(strip_wake_phrase("hey code tell firmware check GPIO"), "tell firmware check GPIO")

    def test_no_wake_phrase(self):
        self.assertEqual(strip_wake_phrase("tell firmware check GPIO"), "tell firmware check GPIO")

    def test_just_wake_phrase(self):
        self.assertEqual(strip_wake_phrase("hey skynet"), "")

    def test_wake_phrase_case_insensitive(self):
        self.assertEqual(strip_wake_phrase("HEY SKYNET, hello"), "hello")

    def test_empty_string(self):
        self.assertEqual(strip_wake_phrase(""), "")

    def test_whitespace_only(self):
        self.assertEqual(strip_wake_phrase("   "), "")


class TestSlashCommands(unittest.TestCase):
    def test_slash_commit(self):
        self.assertEqual(replace_slash_commands("slash commit"), "/commit")

    def test_slash_help(self):
        self.assertEqual(replace_slash_commands("slash help"), "/help")

    def test_slash_review(self):
        self.assertEqual(replace_slash_commands("slash review"), "/review-pr")

    def test_no_slash(self):
        self.assertEqual(replace_slash_commands("check the logs"), "check the logs")

    def test_slash_with_args(self):
        self.assertEqual(replace_slash_commands("slash commit with message"), "/commit with message")


class TestParse(unittest.TestCase):
    def test_tell_with_colon(self):
        cmd = parse("tell firmware: check GPIO pins")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO pins")

    def test_tell_without_colon(self):
        cmd = parse("tell frontend add a login button")
        self.assertEqual(cmd.target, "frontend")
        self.assertEqual(cmd.text, "add a login button")

    def test_go_to(self):
        cmd = parse("go to firmware")
        self.assertEqual(cmd.target, "firmware")
        self.assertIsNone(cmd.text)

    def test_bare_text(self):
        cmd = parse("check the error logs")
        self.assertIsNone(cmd.target)
        self.assertEqual(cmd.text, "check the error logs")

    def test_wake_phrase_stripped(self):
        cmd = parse("hey skynet, tell firmware: check GPIO")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")

    def test_slash_command_in_text(self):
        cmd = parse("tell firmware: slash commit")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "/commit")

    def test_slash_command_bare(self):
        cmd = parse("slash commit")
        self.assertIsNone(cmd.target)
        self.assertEqual(cmd.text, "/commit")

    def test_empty_string(self):
        cmd = parse("")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_just_wake_phrase(self):
        cmd = parse("hey skynet")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_target_is_lowercased(self):
        cmd = parse("tell Firmware: check GPIO")
        self.assertEqual(cmd.target, "firmware")

    def test_go_to_case_insensitive(self):
        cmd = parse("Go To Frontend")
        self.assertEqual(cmd.target, "frontend")
        self.assertIsNone(cmd.text)

    def test_tell_case_insensitive(self):
        cmd = parse("TELL firmware: check GPIO")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")

    def test_whitespace_handling(self):
        cmd = parse("  tell firmware :  check GPIO  ")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")

    def test_for_pattern(self):
        cmd = parse("for firmware, check GPIO pins")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO pins")

    def test_for_pattern_no_comma(self):
        cmd = parse("for firmware check GPIO pins")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO pins")

    def test_for_pattern_colon(self):
        cmd = parse("for firmware: check GPIO pins")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO pins")


class TestHallucinationFilter(unittest.TestCase):
    def test_thank_you(self):
        cmd = parse("Thank you.")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_thank_you_for_watching(self):
        cmd = parse("Thank you for watching!")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_bye(self):
        cmd = parse("Bye.")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_subscribe(self):
        cmd = parse("Subscribe")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_you(self):
        cmd = parse("You")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_real_text_not_filtered(self):
        cmd = parse("check the error logs")
        self.assertIsNone(cmd.target)
        self.assertEqual(cmd.text, "check the error logs")


class TestFillerWordStripping(unittest.TestCase):
    def test_uh_stripped(self):
        cmd = parse("uh check the logs")
        self.assertIsNone(cmd.target)
        self.assertEqual(cmd.text, "check the logs")

    def test_um_stripped(self):
        cmd = parse("um tell firmware check GPIO")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")

    def test_so_stripped(self):
        cmd = parse("so check the logs")
        self.assertIsNone(cmd.target)
        self.assertEqual(cmd.text, "check the logs")

    def test_okay_stripped(self):
        cmd = parse("okay tell firmware check GPIO")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")

    def test_basically_stripped(self):
        cmd = parse("basically check the logs")
        self.assertIsNone(cmd.target)
        self.assertEqual(cmd.text, "check the logs")


class TestInOnPatternRemoved(unittest.TestCase):
    """The in/on pattern was too greedy — 'in general the build is failing' would target 'general'."""
    def test_in_not_parsed_as_verb(self):
        """'in <word>' should not use in/on verb pattern to extract a target."""
        cmd = parse("in theory the build should pass")
        # Should NOT extract 'theory' as a target via in/on pattern
        # (may still match via session name lookup, but the verb pattern is gone)
        self.assertNotEqual(cmd.target, "theory")

    def test_on_not_parsed_as_verb(self):
        """'on <word>' should not use in/on verb pattern to extract a target."""
        cmd = parse("on the other hand we could refactor")
        self.assertNotEqual(cmd.target, "the")


class TestFuzzySessionMatch(unittest.TestCase):
    def test_short_word_rejected(self):
        self.assertIsNone(_fuzzy_session_match("a", ["alpha", "beta"]))

    def test_substring_match(self):
        self.assertEqual(_fuzzy_session_match("firm", ["firmware", "frontend"]), "firmware")

    def test_no_match(self):
        self.assertIsNone(_fuzzy_session_match("xyz", ["firmware", "frontend"]))

    def test_exact_match(self):
        self.assertEqual(_fuzzy_session_match("firmware", ["firmware", "frontend"]), "firmware")

    def test_fuzzy_misspelling(self):
        """Voice transcription 'firmwear' should match 'firmware'."""
        self.assertEqual(_fuzzy_session_match("firmwear", ["firmware", "frontend"]), "firmware")

    def test_joined_words(self):
        """'firm ware' should match 'firmware' via joined form."""
        self.assertEqual(_fuzzy_session_match("firm ware", ["firmware", "frontend"]), "firmware")


class TestTargetFirstJoined(unittest.TestCase):
    """Test that first-two-words joining works for target detection."""

    @patch('parser._load_known_sessions', return_value=["firmware", "frontend"])
    def test_firm_wear_check_gpio(self, mock_sessions):
        """'firm wear check GPIO' should route to firmware."""
        cmd = parse("firm wear check GPIO")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")

    @patch('parser._load_known_sessions', return_value=["firmware", "frontend"])
    def test_front_end_add_button(self, mock_sessions):
        """'front end add a button' should route to frontend."""
        cmd = parse("front end add a button")
        self.assertEqual(cmd.target, "frontend")
        self.assertEqual(cmd.text, "add a button")


if __name__ == "__main__":
    unittest.main()
