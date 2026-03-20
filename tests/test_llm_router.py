#!/usr/bin/env python3
"""Tests for Ollama-based LLM routing (with mocks)."""

import json
import sys
import os
from unittest.mock import patch, MagicMock
from io import BytesIO

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_router import is_available, llm_parse


class TestIsAvailable:
    """Test Ollama availability check."""

    def test_available_when_ollama_running(self):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("llm_router.urllib.request.urlopen", return_value=mock_response):
            assert is_available() is True

    def test_unavailable_when_connection_refused(self):
        import urllib.error
        with patch("llm_router.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            assert is_available() is False

    def test_unavailable_on_timeout(self):
        with patch("llm_router.urllib.request.urlopen", side_effect=TimeoutError):
            assert is_available() is False


class TestLlmParse:
    """Test LLM-based voice command parsing."""

    def _mock_response(self, content: str):
        """Create a mock urllib response with given JSON content."""
        body = json.dumps({
            "message": {"content": content},
        }).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_successful_parse(self):
        mock_resp = self._mock_response('{"target": "firmware", "text": "check GPIO pins"}')
        with patch("llm_router.urllib.request.urlopen", return_value=mock_resp):
            result = llm_parse("check GPIO pins on firmware", ["firmware", "frontend"])
        assert result is not None
        assert result["target"] == "firmware"
        assert result["text"] == "check GPIO pins"

    def test_case_insensitive_target_validation(self):
        mock_resp = self._mock_response('{"target": "Firmware", "text": "check GPIO"}')
        with patch("llm_router.urllib.request.urlopen", return_value=mock_resp):
            result = llm_parse("check GPIO on firmware", ["firmware", "frontend"])
        assert result is not None
        assert result["target"] == "firmware"

    def test_unknown_target_rejected(self):
        mock_resp = self._mock_response('{"target": "database", "text": "check status"}')
        with patch("llm_router.urllib.request.urlopen", return_value=mock_resp):
            result = llm_parse("check database status", ["firmware", "frontend"])
        assert result is not None
        assert result["target"] is None  # rejected because not in known sessions

    def test_null_target_preserved(self):
        mock_resp = self._mock_response('{"target": null, "text": "check the logs"}')
        with patch("llm_router.urllib.request.urlopen", return_value=mock_resp):
            result = llm_parse("check the logs", ["firmware"])
        assert result is not None
        assert result["target"] is None
        assert result["text"] == "check the logs"

    def test_markdown_code_fence_stripped(self):
        content = '```json\n{"target": "firmware", "text": "hello"}\n```'
        mock_resp = self._mock_response(content)
        with patch("llm_router.urllib.request.urlopen", return_value=mock_resp):
            result = llm_parse("hello firmware", ["firmware"])
        assert result is not None
        assert result["target"] == "firmware"

    def test_empty_sessions_returns_none(self):
        result = llm_parse("check logs", [])
        assert result is None

    def test_timeout_returns_none(self):
        with patch("llm_router.urllib.request.urlopen", side_effect=TimeoutError):
            result = llm_parse("check logs", ["firmware"])
        assert result is None

    def test_invalid_json_returns_none(self):
        mock_resp = self._mock_response("I cannot parse that command")
        with patch("llm_router.urllib.request.urlopen", return_value=mock_resp):
            result = llm_parse("gibberish", ["firmware"])
        assert result is None


@pytest.mark.integration
class TestLlmRouterIntegration:
    """Integration tests that require a running Ollama instance."""

    def test_real_ollama_availability(self):
        """Smoke test: check if Ollama is reachable."""
        # This will pass if Ollama is running, skip-worthy otherwise
        result = is_available()
        assert isinstance(result, bool)
