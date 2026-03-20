#!/usr/bin/env python3
"""Unit tests for the voice command parser."""

import sys
import os
import unittest

# Add src to path so we can import parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import patch, MagicMock

from parser import parse, strip_wake_phrase, replace_slash_commands, ParsedCommand, _load_known_sessions, _fuzzy_session_match


def _isolated_parse(text):
    """Parse with LLM router and session registry disabled for unit test isolation."""
    with patch('parser._load_known_sessions', return_value=[]):
        # Block LLM fallback by making the import fail
        with patch.dict('sys.modules', {'llm_router': None}):
            return parse(text)


class TestStripWakePhrase(unittest.TestCase):
    def test_hey_skynet(self):
        self.assertEqual(strip_wake_phrase("hey skynet, check the logs"), "check the logs")

    def test_hey_destroyer(self):
        self.assertEqual(strip_wake_phrase("hey destroyer check GPIO"), "check GPIO")

    def test_no_wake_phrase(self):
        self.assertEqual(strip_wake_phrase("check the logs"), "check the logs")

    def test_case_insensitive(self):
        self.assertEqual(strip_wake_phrase("Hey Skynet, run tests"), "run tests")


class TestSlashCommands(unittest.TestCase):
    def test_slash_commit(self):
        self.assertEqual(replace_slash_commands("slash commit"), "/commit")

    def test_slash_review(self):
        self.assertEqual(replace_slash_commands("slash review"), "/review-pr")

    def test_no_slash(self):
        self.assertEqual(replace_slash_commands("check the logs"), "check the logs")


class TestParse(unittest.TestCase):
    def test_tell_with_colon(self):
        cmd = _isolated_parse("tell firmware: check GPIO")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")

    def test_tell_without_colon(self):
        cmd = _isolated_parse("tell firmware check GPIO")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")

    def test_go_to(self):
        cmd = _isolated_parse("go to frontend")
        self.assertEqual(cmd.target, "frontend")
        self.assertIsNone(cmd.text)

    def test_bare_text(self):
        cmd = _isolated_parse("check the logs")
        self.assertIsNone(cmd.target)
        self.assertEqual(cmd.text, "check the logs")

    def test_slash_command_bare(self):
        cmd = _isolated_parse("slash commit")
        self.assertIsNone(cmd.target)
        self.assertEqual(cmd.text, "/commit")

    def test_slash_command_in_tell(self):
        cmd = _isolated_parse("tell firmware: slash commit")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "/commit")

    def test_case_insensitive(self):
        cmd = _isolated_parse("Tell Firmware: Check GPIO")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "Check GPIO")

    def test_whitespace_handling(self):
        cmd = _isolated_parse("  tell firmware:   check GPIO  ")
        self.assertEqual(cmd.target, "firmware")
        self.assertEqual(cmd.text, "check GPIO")


class TestHallucinationFilter(unittest.TestCase):
    def test_thank_you_for_watching(self):
        cmd = _isolated_parse("Thank you for watching.")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_subscribe(self):
        cmd = _isolated_parse("Subscribe")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_thank_you(self):
        cmd = _isolated_parse("Thank you")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_you(self):
        cmd = _isolated_parse("you")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_bye(self):
        cmd = _isolated_parse("Bye")
        self.assertIsNone(cmd.target)
        self.assertIsNone(cmd.text)

    def test_real_text_not_filtered(self):
        cmd = _isolated_parse("check the error logs")
        self.assertIsNotNone(cmd.text)


class TestFillerWordStripping(unittest.TestCase):
    def test_uh_stripped(self):
        cmd = _isolated_parse("uh tell firmware check GPIO")
        self.assertEqual(cmd.target, "firmware")

    def test_um_stripped(self):
        cmd = _isolated_parse("um tell firmware check GPIO")
        self.assertEqual(cmd.target, "firmware")

    def test_so_stripped(self):
        cmd = _isolated_parse("so tell firmware check GPIO")
        self.assertEqual(cmd.target, "firmware")

    def test_okay_stripped(self):
        cmd = _isolated_parse("okay tell firmware check GPIO")
        self.assertEqual(cmd.target, "firmware")

    def test_basically_stripped(self):
        cmd = _isolated_parse("basically tell firmware check GPIO")
        self.assertEqual(cmd.target, "firmware")


class TestInOnPatternRemoved(unittest.TestCase):
    def test_in_not_parsed_as_verb(self):
        cmd = _isolated_parse("in general the build is failing")
        self.assertIsNone(cmd.target)

    def test_on_not_parsed_as_verb(self):
        cmd = _isolated_parse("on the other hand try this")
        self.assertIsNone(cmd.target)


class TestFuzzySessionMatch(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(_fuzzy_session_match("firmware", ["firmware", "frontend"]), "firmware")

    def test_substring_match(self):
        self.assertEqual(_fuzzy_session_match("firm", ["firmware", "frontend"]), "firmware")

    def test_no_match(self):
        self.assertIsNone(_fuzzy_session_match("banana", ["firmware", "frontend"]))

    def test_short_word_rejected(self):
        self.assertIsNone(_fuzzy_session_match("a", ["firmware", "frontend"]))

    def test_fuzzy_misspelling(self):
        result = _fuzzy_session_match("firmwear", ["firmware", "frontend"])
        self.assertEqual(result, "firmware")

    def test_joined_words(self):
        result = _fuzzy_session_match("firmware", ["firmware", "frontend"])
        self.assertEqual(result, "firmware")


class TestTargetFirstJoined(unittest.TestCase):
    def test_firm_wear_check_gpio(self):
        with patch('parser._load_known_sessions', return_value=["firmware", "frontend"]):
            with patch.dict('sys.modules', {'llm_router': None}):
                cmd = parse("firm wear check GPIO")
                self.assertEqual(cmd.target, "firmware")
                self.assertEqual(cmd.text, "check GPIO")

    def test_front_end_add_button(self):
        with patch('parser._load_known_sessions', return_value=["firmware", "frontend"]):
            with patch.dict('sys.modules', {'llm_router': None}):
                cmd = parse("front end add a button")
                self.assertEqual(cmd.target, "frontend")
                self.assertEqual(cmd.text, "add a button")


if __name__ == "__main__":
    unittest.main()
