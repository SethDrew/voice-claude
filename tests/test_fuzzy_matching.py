#!/usr/bin/env python3
"""Tests for rapidfuzz-powered fuzzy matching in router and parser."""

import sys
import os

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from router import _fuzzy_match
from parser import _fuzzy_session_match


class TestRouterFuzzyMatch:
    """Test _fuzzy_match in router.py (rapidfuzz version)."""

    def test_exact_match(self):
        assert _fuzzy_match("firmware", "firmware")

    def test_substring_match(self):
        assert _fuzzy_match("firm", "firmware")

    def test_substring_in_hyphenated(self):
        assert _fuzzy_match("firmware", "dock-firmware")

    def test_short_query_exact(self):
        assert _fuzzy_match("a", "a")

    def test_short_query_rejects_partial(self):
        assert not _fuzzy_match("a", "alpha")

    def test_case_insensitive(self):
        assert _fuzzy_match("Firmware", "firmware")

    def test_joined_match_spaces(self):
        """'firm ware' should match 'firmware' via joined form."""
        assert _fuzzy_match("firm ware", "firmware")

    def test_joined_match_hyphens(self):
        """'dock-firmware' should match 'dockfirmware' joined."""
        assert _fuzzy_match("dock-firmware", "dockfirmware")

    def test_fuzzy_ratio_match(self):
        """Misspellings should match via fuzz.ratio."""
        assert _fuzzy_match("firmwear", "firmware")

    def test_fuzzy_partial_ratio_match(self):
        """Partial misspellings should match via fuzz.partial_ratio."""
        assert _fuzzy_match("firmw", "firmware")

    def test_no_match(self):
        """Completely unrelated strings should not match."""
        assert not _fuzzy_match("xyzzyx", "firmware")

    def test_empty_query(self):
        assert not _fuzzy_match("", "firmware")

    def test_single_char_no_match(self):
        assert not _fuzzy_match("f", "firmware")


class TestParserFuzzySessionMatch:
    """Test _fuzzy_session_match in parser.py (rapidfuzz version)."""

    def test_exact_match(self):
        assert _fuzzy_session_match("firmware", ["firmware", "frontend"]) == "firmware"

    def test_substring_match(self):
        assert _fuzzy_session_match("firm", ["firmware", "frontend"]) == "firmware"

    def test_reverse_substring(self):
        """Session name contained in the word."""
        assert _fuzzy_session_match("firmwareproject", ["firmware", "frontend"]) == "firmware"

    def test_short_word_rejected(self):
        assert _fuzzy_session_match("a", ["alpha", "beta"]) is None

    def test_no_match(self):
        assert _fuzzy_session_match("xyzzyx", ["firmware", "frontend"]) is None

    def test_fuzzy_misspelling(self):
        """Voice transcription error: 'firmwear' should match 'firmware'."""
        result = _fuzzy_session_match("firmwear", ["firmware", "frontend"])
        assert result == "firmware"

    def test_joined_words(self):
        """'firm ware' joined should match 'firmware'."""
        result = _fuzzy_session_match("firm ware", ["firmware", "frontend"])
        assert result == "firmware"

    def test_best_match_selected(self):
        """Should pick the best fuzzy match."""
        result = _fuzzy_session_match("front", ["firmware", "frontend"])
        assert result == "frontend"

    def test_empty_sessions(self):
        assert _fuzzy_session_match("firmware", []) is None
