#!/usr/bin/env python3
"""iTerm2 session router — find and send text to named Claude Code sessions."""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import iterm2

STATE_DIR = Path.home() / ".local" / "share" / "voice-router"
STATE_FILE = STATE_DIR / "state.json"


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


def get_last_active() -> Optional[str]:
    return _load_state().get("last_active")


def set_last_active(name: str) -> None:
    state = _load_state()
    state["last_active"] = name
    _save_state(state)


async def find_session(connection: iterm2.Connection, name: str) -> Optional[iterm2.Session]:
    """Find a session by cc_name user variable, falling back to tab title."""
    app = await iterm2.async_get_app(connection)
    name_lower = name.lower()

    # First pass: match on cc_name user variable
    for window in app.windows:
        for tab in window.tabs:
            for session in tab.sessions:
                try:
                    cc_name = await session.async_get_variable("user.cc_name")
                    if cc_name and cc_name.lower() == name_lower:
                        return session
                except Exception:
                    continue

    # Second pass: match on tab title (partial match)
    for window in app.windows:
        for tab in window.tabs:
            for session in tab.sessions:
                try:
                    title = await session.async_get_variable("name")
                    if title and name_lower in title.lower():
                        return session
                except Exception:
                    continue

    return None


async def list_sessions(connection: iterm2.Connection) -> list[dict]:
    """Return all sessions with cc_name set."""
    app = await iterm2.async_get_app(connection)
    sessions = []

    for window in app.windows:
        for tab in window.tabs:
            for session in tab.sessions:
                try:
                    cc_name = await session.async_get_variable("user.cc_name")
                    if cc_name:
                        title = await session.async_get_variable("name") or ""
                        sessions.append({
                            "name": cc_name,
                            "title": title,
                            "session_id": session.session_id,
                        })
                except Exception:
                    continue

    return sessions


async def activate_session(connection: iterm2.Connection, session: iterm2.Session) -> None:
    """Bring a session's tab to the front."""
    app = await iterm2.async_get_app(connection)
    for window in app.windows:
        for tab in window.tabs:
            if session in tab.sessions:
                await tab.async_select()
                await session.async_activate()
                return


async def route_command(connection: iterm2.Connection, target: Optional[str], text: Optional[str]) -> bool:
    """Route text to a named session. Returns True on success."""
    # Resolve target
    session_name = target or get_last_active()
    if not session_name:
        print("Error: no target session and no last-active session")
        return False

    session = await find_session(connection, session_name)
    if not session:
        print(f"Error: session '{session_name}' not found")
        available = await list_sessions(connection)
        if available:
            names = ", ".join(s["name"] for s in available)
            print(f"Available sessions: {names}")
        return False

    # Focus the session
    await activate_session(connection, session)

    # Send text if provided (not focus-only)
    if text is not None:
        await session.async_send_text(text + "\n")

    set_last_active(session_name)
    return True
