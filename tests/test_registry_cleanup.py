#!/usr/bin/env python3
"""Tests for automatic registry cleanup.

The name registry (name-registry.json) accumulates dead sessions.
These tests verify that do_list() prunes stale entries from the registry.
"""

import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def tmp_registry(tmp_path):
    """Create a temporary name registry file."""
    registry_file = tmp_path / "name-registry.json"
    return registry_file


class TestRegistryCleanup:
    """Test that do_list() cleans up dead sessions from the registry."""

    @pytest.mark.asyncio
    async def test_registry_cleaned_on_list(self, tmp_registry):
        """voice-route --list should remove dead sessions from registry."""
        # Registry has 3 entries, but only 2 are live
        tmp_registry.write_text(json.dumps({
            "session-1": "firmware",
            "session-2": "built-app",
            "session-3": "dead-session",
        }))

        # Mock list_sessions to return only 2 live sessions
        mock_sessions = [
            {"name": "firmware", "title": "firmware session", "session_id": "s1"},
            {"name": "built-app", "title": "built-app session", "session_id": "s2"},
        ]

        from main import do_list
        from router import NAME_REGISTRY_FILE

        with patch('main.iterm2.Connection.async_create', new_callable=AsyncMock):
            with patch('main.list_sessions', new_callable=AsyncMock, return_value=mock_sessions):
                with patch('router.NAME_REGISTRY_FILE', tmp_registry):
                    with patch('main.NAME_REGISTRY_FILE', tmp_registry):
                        await do_list()

        # Registry should now only have the 2 live sessions
        cleaned = json.loads(tmp_registry.read_text())
        assert "dead-session" not in cleaned.values()
        assert "firmware" in cleaned.values()
        assert "built-app" in cleaned.values()
        assert len(cleaned) == 2

    @pytest.mark.asyncio
    async def test_registry_keeps_live_sessions(self, tmp_registry):
        """Live sessions should survive cleanup."""
        tmp_registry.write_text(json.dumps({
            "session-1": "firmware",
            "session-2": "built-app",
        }))

        mock_sessions = [
            {"name": "firmware", "title": "firmware session", "session_id": "s1"},
            {"name": "built-app", "title": "built-app session", "session_id": "s2"},
        ]

        from main import do_list

        with patch('main.iterm2.Connection.async_create', new_callable=AsyncMock):
            with patch('main.list_sessions', new_callable=AsyncMock, return_value=mock_sessions):
                with patch('main.NAME_REGISTRY_FILE', tmp_registry):
                    await do_list()

        cleaned = json.loads(tmp_registry.read_text())
        assert len(cleaned) == 2
        assert "firmware" in cleaned.values()
        assert "built-app" in cleaned.values()

    @pytest.mark.asyncio
    async def test_registry_cleanup_handles_empty(self, tmp_registry):
        """Empty registry doesn't crash."""
        tmp_registry.write_text(json.dumps({}))

        mock_sessions = [
            {"name": "firmware", "title": "firmware session", "session_id": "s1"},
        ]

        from main import do_list

        with patch('main.iterm2.Connection.async_create', new_callable=AsyncMock):
            with patch('main.list_sessions', new_callable=AsyncMock, return_value=mock_sessions):
                with patch('main.NAME_REGISTRY_FILE', tmp_registry):
                    await do_list()

        cleaned = json.loads(tmp_registry.read_text())
        assert len(cleaned) == 0

    @pytest.mark.asyncio
    async def test_registry_cleanup_handles_missing_file(self, tmp_registry):
        """Missing registry file doesn't crash."""
        # Don't create the file — it shouldn't exist
        assert not tmp_registry.exists()

        mock_sessions = [
            {"name": "firmware", "title": "firmware session", "session_id": "s1"},
        ]

        from main import do_list

        with patch('main.iterm2.Connection.async_create', new_callable=AsyncMock):
            with patch('main.list_sessions', new_callable=AsyncMock, return_value=mock_sessions):
                with patch('main.NAME_REGISTRY_FILE', tmp_registry):
                    await do_list()

        # Should not crash, and file should not be created if it didn't exist
        # (nothing to clean up)

    @pytest.mark.asyncio
    async def test_registry_no_write_when_no_changes(self, tmp_registry):
        """Registry should not be rewritten if nothing was pruned."""
        original = {"session-1": "firmware", "session-2": "built-app"}
        tmp_registry.write_text(json.dumps(original))
        original_mtime = tmp_registry.stat().st_mtime

        mock_sessions = [
            {"name": "firmware", "title": "firmware session", "session_id": "s1"},
            {"name": "built-app", "title": "built-app session", "session_id": "s2"},
        ]

        from main import do_list

        import time
        time.sleep(0.01)  # Ensure mtime would differ if rewritten

        with patch('main.iterm2.Connection.async_create', new_callable=AsyncMock):
            with patch('main.list_sessions', new_callable=AsyncMock, return_value=mock_sessions):
                with patch('main.NAME_REGISTRY_FILE', tmp_registry):
                    await do_list()

        # File should not have been rewritten (same mtime)
        assert tmp_registry.stat().st_mtime == original_mtime
