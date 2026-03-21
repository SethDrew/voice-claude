#!/usr/bin/env python3
"""iTerm2 session router — find and send text to named Claude Code sessions."""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

import iterm2
from rapidfuzz import fuzz

STATE_DIR = Path.home() / ".local" / "share" / "voice-claude"
STATE_FILE = STATE_DIR / "state.json"
NAME_REGISTRY_FILE = STATE_DIR / "name-registry.json"


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


def _load_name_registry() -> dict:
    """Load the session name registry (session_id -> name)."""
    try:
        return json.loads(NAME_REGISTRY_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _fuzzy_match(query: str, candidate: str) -> bool:
    """Check if query is a fuzzy/partial match for candidate.

    Uses rapidfuzz for robust matching of voice-transcribed session names.
    Very short queries (< 2 chars) require an exact match to avoid
    false positives (e.g. session named "a" matching everything).
    Examples:
        - "firmware" matches "dock-firmware"
        - "dock" matches "dock-firmware"
        - "firm wear" matches "firmware" (via joined form)
    """
    q = query.lower()
    c = candidate.lower()
    if len(q) < 2:
        return q == c  # exact match only for very short queries

    # Fast path: direct substring match
    if q in c:
        return True

    # Join words (remove spaces/hyphens) and retry substring
    q_joined = re.sub(r'[\s\-]+', '', q)
    c_joined = re.sub(r'[\s\-]+', '', c)
    if q_joined in c_joined:
        return True

    # Fuzzy ratio check
    if fuzz.ratio(q_joined, c_joined) >= 65:
        return True

    # Partial ratio check
    if fuzz.partial_ratio(q_joined, c_joined) >= 80:
        return True

    return False


def get_last_active() -> Optional[str]:
    return _load_state().get("last_active")


def set_last_active(name: str) -> None:
    state = _load_state()
    state["last_active"] = name
    _save_state(state)


async def find_session(connection: iterm2.Connection, name: str) -> Optional[iterm2.Session]:
    """Find a session by cc_name user variable, tab title, or name registry.

    Priority:
    1. Exact match on cc_name user variable
    2. Fuzzy match on cc_name user variable
    3. Fuzzy match on session/tab title
    4. Fuzzy match via name registry (session_id -> name mapping)
    """
    app = await iterm2.async_get_app(connection)
    name_lower = name.lower()

    # Collect all sessions for multi-pass matching
    all_sessions = []
    for window in app.windows:
        for tab in window.tabs:
            for session in tab.sessions:
                all_sessions.append(session)

    # Pass 1: exact match on cc_name user variable (highest priority)
    for session in all_sessions:
        try:
            cc_name = await session.async_get_variable("user.cc_name")
            if cc_name and cc_name.lower() == name_lower:
                return session
        except Exception:
            continue

    # Pass 2: fuzzy/partial match on cc_name user variable
    for session in all_sessions:
        try:
            cc_name = await session.async_get_variable("user.cc_name")
            if cc_name and _fuzzy_match(name, cc_name):
                return session
        except Exception:
            continue

    # Pass 3: fuzzy/partial match on session name / tab title
    for session in all_sessions:
        try:
            title = await session.async_get_variable("name")
            if title and _fuzzy_match(name, title):
                return session
        except Exception:
            continue

    # Pass 4: match via name registry as fallback
    registry = _load_name_registry()
    if registry:
        # Build a map of session_id -> iterm2.Session for lookup
        session_map = {}
        for session in all_sessions:
            session_map[session.session_id] = session

        # Check registry entries for fuzzy match on the registered name
        for reg_session_id, reg_name in registry.items():
            if _fuzzy_match(name, reg_name):
                # Find the iTerm2 session that corresponds to this registry entry.
                # The registry stores Claude Code session IDs, not iTerm2 session IDs,
                # so we check if any session's cc_session_id or title matches.
                # As a practical fallback, try to match by checking terminal titles
                # that may have been set by the hook.
                for session in all_sessions:
                    try:
                        title = await session.async_get_variable("name")
                        if title and reg_name.lower() in title.lower():
                            return session
                    except Exception:
                        continue

    return None


async def list_sessions(connection: iterm2.Connection) -> list[dict]:
    """Return all sessions with cc_name set, supplemented by name registry."""
    app = await iterm2.async_get_app(connection)
    sessions = []
    seen_session_ids = set()

    # First: gather sessions that have cc_name set
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
                        seen_session_ids.add(session.session_id)
                except Exception:
                    continue

    # Second: check name registry for sessions that may not have cc_name set
    # but do have a terminal title matching a registry name
    registry = _load_name_registry()
    if registry:
        for window in app.windows:
            for tab in window.tabs:
                for session in tab.sessions:
                    if session.session_id in seen_session_ids:
                        continue
                    try:
                        title = await session.async_get_variable("name") or ""
                        for reg_name in registry.values():
                            if title and reg_name.lower() in title.lower():
                                sessions.append({
                                    "name": reg_name,
                                    "title": title,
                                    "session_id": session.session_id,
                                })
                                seen_session_ids.add(session.session_id)
                                break
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
