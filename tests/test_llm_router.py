#!/usr/bin/env python3
"""Tests for MLX-LM-based LLM routing (with mocks)."""

import json
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import llm_router
from llm_router import is_available, llm_parse


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Reset the lazy-loaded model singleton between tests."""
    llm_router._model = None
    llm_router._tokenizer = None
    yield
    llm_router._model = None
    llm_router._tokenizer = None


def _setup_mlx_mock(response_text):
    """Create a mock mlx_lm module that returns the given text from generate().

    Returns a context-manager-compatible mock setup for use with `patch.dict`.
    The mock module has load() and generate() functions.
    """
    mock_model = MagicMock(name="mock_model")
    mock_tokenizer = MagicMock(name="mock_tokenizer")
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    mock_mlx_lm = MagicMock()
    mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
    mock_mlx_lm.generate.return_value = response_text

    return mock_mlx_lm


class TestIsAvailable:
    """Test MLX-LM availability check."""

    def test_available_when_mlx_lm_installed(self):
        mock_mlx = MagicMock()
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
            assert is_available() is True

    def test_unavailable_when_mlx_lm_not_installed(self):
        # Setting module to None in sys.modules causes ImportError
        with patch.dict("sys.modules", {"mlx_lm": None}):
            assert is_available() is False


class TestLlmParse:
    """Test LLM-based voice command parsing."""

    def _run_parse(self, response_text, raw_text, sessions):
        """Helper: run llm_parse with a mocked mlx_lm returning response_text."""
        mock_mlx = _setup_mlx_mock(response_text)
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
            return llm_parse(raw_text, sessions)

    def test_successful_parse(self):
        result = self._run_parse(
            '{"target": "firmware", "text": "check GPIO pins"}',
            "check GPIO pins on firmware",
            ["firmware", "frontend"],
        )
        assert result is not None
        assert result["target"] == "firmware"
        assert result["text"] == "check GPIO pins"

    def test_case_insensitive_target_validation(self):
        result = self._run_parse(
            '{"target": "Firmware", "text": "check GPIO"}',
            "check GPIO on firmware",
            ["firmware", "frontend"],
        )
        assert result is not None
        assert result["target"] == "firmware"

    def test_unknown_target_rejected(self):
        result = self._run_parse(
            '{"target": "database", "text": "check status"}',
            "check database status",
            ["firmware", "frontend"],
        )
        assert result is not None
        assert result["target"] is None  # rejected because not in known sessions

    def test_null_target_preserved(self):
        result = self._run_parse(
            '{"target": null, "text": "check the logs"}',
            "check the logs",
            ["firmware"],
        )
        assert result is not None
        assert result["target"] is None
        assert result["text"] == "check the logs"

    def test_markdown_code_fence_stripped(self):
        content = '```json\n{"target": "firmware", "text": "hello"}\n```'
        result = self._run_parse(content, "hello firmware", ["firmware"])
        assert result is not None
        assert result["target"] == "firmware"

    def test_empty_sessions_returns_none(self):
        result = llm_parse("check logs", [])
        assert result is None

    def test_import_error_returns_none(self):
        # Setting module to None causes ImportError on `from mlx_lm import generate`
        with patch.dict("sys.modules", {"mlx_lm": None}):
            result = llm_parse("check logs", ["firmware"])
        assert result is None

    def test_invalid_json_returns_none(self):
        result = self._run_parse(
            "I cannot parse that command",
            "gibberish",
            ["firmware"],
        )
        assert result is None


class TestModelCaching:
    """Test that the model is loaded once and cached."""

    def test_model_loaded_once(self):
        mock_mlx = _setup_mlx_mock('{"target": "a", "text": "b"}')
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
            llm_router._load_model()
            llm_router._load_model()
        # load() should only be called once due to caching
        mock_mlx.load.assert_called_once()


@pytest.mark.integration
class TestLlmRouterIntegration:
    """Integration tests that require mlx-lm and a downloaded model."""

    def test_real_mlx_availability(self):
        """Smoke test: check if mlx-lm is importable."""
        result = is_available()
        assert isinstance(result, bool)
