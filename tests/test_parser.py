#!/usr/bin/env python3
"""Unit tests for the voice command parser."""

import sys
import os
import unittest

# Add src to path so we can import parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from parser import parse, strip_wake_phrase, replace_slash_commands, ParsedCommand


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


if __name__ == "__main__":
    unittest.main()
