#!/usr/bin/env python3
"""Voice command parser for routing speech to Claude Code sessions."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

WAKE_PHRASES = [
    "hey skynet",
    "hey destroyer",
    "hey code",
]

# Voice-friendly slash commands
SLASH_COMMANDS = {
    "slash commit": "/commit",
    "slash help": "/help",
    "slash clear": "/clear",
    "slash review": "/review-pr",
    "slash status": "/status",
    "slash diff": "/diff",
    "slash compact": "/compact",
    "slash init": "/init",
}

@dataclass
class ParsedCommand:
    target: Optional[str]  # Session name, or None for last-active
    text: Optional[str]     # Text to send, or None for focus-only

def strip_wake_phrase(raw: str) -> str:
    """Remove wake phrase prefix if present."""
    lower = raw.lower().strip()
    for phrase in WAKE_PHRASES:
        if lower.startswith(phrase):
            rest = raw[len(phrase):].lstrip(" ,:.!").strip()
            return rest if rest else ""
    return raw.strip()

def replace_slash_commands(text: str) -> str:
    """Convert voice-friendly slash commands to actual slash commands."""
    lower = text.lower()
    for voice_form, slash_form in SLASH_COMMANDS.items():
        if lower.startswith(voice_form):
            return slash_form + text[len(voice_form):]
    return text

REGISTRY_FILE = Path.home() / ".local" / "share" / "voice-claude" / "name-registry.json"

def _load_known_sessions() -> list[str]:
    """Load known session names from the registry."""
    try:
        registry = json.loads(REGISTRY_FILE.read_text())
        return list(registry.values())
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def _fuzzy_session_match(word: str, sessions: list[str]) -> str | None:
    """Check if a word fuzzy-matches any known session name.

    Uses rapidfuzz to handle voice transcription errors like
    'firm wear' -> 'firmware'.
    """
    word_lower = word.lower()
    if len(word_lower) < 2:
        return None

    word_joined = re.sub(r'[\s\-]+', '', word_lower)

    best_match = None
    best_score = 0
    best_ratio = 0
    best_partial = 0

    for name in sessions:
        name_lower = name.lower()
        name_joined = re.sub(r'[\s\-]+', '', name_lower)

        # Direct substring match (fast path)
        if word_lower in name_lower or name_lower in word_lower:
            return name

        # Joined substring match
        if word_joined in name_joined or name_joined in word_joined:
            return name

        # Fuzzy scoring on joined forms — track ratio and partial_ratio
        # separately so we can apply different thresholds (matching the
        # router's _fuzzy_match logic)
        ratio = fuzz.ratio(word_joined, name_joined)
        partial = fuzz.partial_ratio(word_joined, name_joined)
        score = max(ratio, partial)
        if score > best_score:
            best_score = score
            best_ratio = ratio
            best_partial = partial
            best_match = name

    # Use separate thresholds consistent with the router's _fuzzy_match:
    # ratio >= 65 catches genuine misspellings (e.g. "firmwear" vs "firmware")
    # partial_ratio >= 80 catches partial matches without false positives
    # (a single threshold of 65 on max() was too aggressive — e.g.
    # partial_ratio("built","ipaddrthunderbolt")=66.7 falsely matched)
    if best_match and (best_ratio >= 65 or best_partial >= 80):
        return best_match

    return None

def parse(raw: str) -> ParsedCommand:
    """Parse a voice command into target and text.

    Patterns:
        "tell <target>: <text>"  -> target + text
        "tell <target> <text>"   -> target + text
        "go to <target>"         -> target, focus only
        bare text                -> last-active session
    """
    text = strip_wake_phrase(raw)

    # Filter common Whisper hallucinations
    HALLUCINATIONS = {
        "thank you for watching", "thanks for watching", "subscribe",
        "please subscribe", "like and subscribe", "thank you",
        "you", "bye", "goodbye",
    }
    if text.lower().strip().rstrip('.!') in HALLUCINATIONS:
        return ParsedCommand(target=None, text=None)

    # Strip leading filler words
    text = re.sub(r'^(?:uh|um|so|like|okay|well|alright|anyway|basically)\s+', '', text, flags=re.IGNORECASE).strip()

    if not text:
        return ParsedCommand(target=None, text=None)

    # Apply slash command substitution
    text = replace_slash_commands(text)

    # "go to <target>" / "switch to <target>" / "focus <target>"
    m = re.match(r'^(?:go\s+to|switch\s+to|focus)\s+(\S+)\s*$', text, re.IGNORECASE)
    if m:
        return ParsedCommand(target=m.group(1).lower(), text=None)

    # "<verb> <target>: <text>" with colon
    verbs = r'(?:tell|ask|send|message|ping|talk\s+to|hey|yo)'
    m = re.match(rf'^{verbs}\s+(\S+)\s*:\s*(.+)$', text, re.IGNORECASE)
    if m:
        return ParsedCommand(
            target=m.group(1).lower(),
            text=replace_slash_commands(m.group(2).strip()),
        )

    # "<verb> <target> [to] <text>" without colon — strip leading "to" from text
    m = re.match(rf'^{verbs}\s+(\S+)\s+(.+)$', text, re.IGNORECASE)
    if m:
        cmd_text = re.sub(r'^to\s+', '', m.group(2).strip(), flags=re.IGNORECASE)
        return ParsedCommand(
            target=m.group(1).lower(),
            text=replace_slash_commands(cmd_text),
        )

    # "for <target>, <text>" / "for <target> <text>"
    m = re.match(r'^for\s+([^\s,:]+)\s*[,:]?\s+(.+)$', text, re.IGNORECASE)
    if m:
        return ParsedCommand(
            target=m.group(1).lower(),
            text=replace_slash_commands(m.group(2).strip()),
        )

    # Target-first: if first word(s) match a known session name, treat as <target> <text>
    sessions = _load_known_sessions()
    all_words = text.split()

    # Try joining first two words first (handles "firm wear" -> "firmware")
    # This must come before single-word match so "firm wear X" doesn't get
    # matched as target="firmware" text="wear X" via the single-word "firm".
    if len(all_words) >= 3:
        joined_first_two = all_words[0] + all_words[1]
        matched = _fuzzy_session_match(joined_first_two, sessions)
        if matched:
            remainder = " ".join(all_words[2:])
            return ParsedCommand(
                target=matched,
                text=replace_slash_commands(remainder.strip()),
            )

    # Single first word match
    words = text.split(None, 1)
    if len(words) >= 2:
        matched = _fuzzy_session_match(words[0], sessions)
        if matched:
            return ParsedCommand(
                target=matched,
                text=replace_slash_commands(words[1].strip()),
            )

    # LLM fallback — try Ollama-based routing if available
    try:
        from llm_router import is_available, llm_parse
        if is_available():
            result = llm_parse(text, sessions if sessions else _load_known_sessions())
            if result and (result.get("target") or result.get("text")):
                return ParsedCommand(
                    target=result.get("target"),
                    text=replace_slash_commands(result["text"]) if result.get("text") else None,
                )
    except ImportError:
        pass

    # Bare text — route to last-active
    return ParsedCommand(target=None, text=text)


def _scan_anywhere_target(text: str, sessions: list[str]) -> str | None:
    """Scan for routing patterns ANYWHERE in the text.

    This is a fallback for when parse() (which checks start-of-message
    patterns) finds no target. It looks for preposition-based routing
    patterns that indicate the user wants to direct the message to a
    specific session.

    Routing patterns (preposition + session name):
        - "send [this] to <session>"
        - "route [this] to <session>"
        - "switch to <session>"
        - "go to <session>"
        - "talk to <session>"
        - "for <session>" at the END of the message only

    NOT routing (session as subject/object):
        - "firmware has a bug"
        - "check if firmware works"
        - "like firmware does"

    Returns the matched session name or None.
    """
    if not sessions:
        return None

    # Patterns where a session name follows a routing preposition.
    # Each pattern captures what comes after the preposition.
    # We try to match the captured word(s) against known sessions.
    routing_patterns = [
        # "send [this/it] to <session>"
        r'(?:send|route|forward|redirect)(?:\s+(?:this|it|that))?\s+to\s+(\S+)',
        # "switch to <session>", "go to <session>"
        r'(?:switch|go|move)\s+to\s+(\S+)',
        # "talk to <session>" / "speak to <session>"
        r'(?:talk|speak)\s+to\s+(?:the\s+)?(\S+)',
        # "to the <session> session" (mid-sentence with "session" after)
        r'to\s+(?:the\s+)?(\S+)\s+session',
    ]

    for pattern in routing_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            candidate = m.group(1).rstrip('.,;:!?')
            matched = _fuzzy_session_match(candidate, sessions)
            if matched:
                return matched

    # "for <session>" at the END of the message only
    # This avoids false positives like "the fix for firmware is in the PR"
    end_for = re.search(r'(?:^|,\s*|\.\s*)\s*(?:this\s+is\s+)?for\s+(\S+)\s*[.!?]?\s*$', text, re.IGNORECASE)
    if end_for:
        candidate = end_for.group(1).rstrip('.,;:!?')
        matched = _fuzzy_session_match(candidate, sessions)
        if matched:
            return matched

    return None


def classify(text: str, sticky_target: str | None, sessions: list[str] | None = None) -> dict:
    """Classify whether text is a routing command or content for the sticky target.

    Uses parse() to detect routing intent, then compares the parsed target
    against the current sticky_target to decide the action.

    Fallback order:
        1. parse() — start-of-message patterns (tell X, go to X, etc.)
        2. _scan_anywhere_target() — routing prepositions anywhere in text
        3. LLM router (if available)
        4. Return as content for sticky target or last-active

    Returns one of:
        {"action": "switch", "target": "<name>", "text": "<command>"}
        {"action": "content", "text": "<full original text>"}
        {"action": "self", "text": "<stripped text>"}
    """
    # Resolve session list
    effective_sessions = sessions if sessions is not None else _load_known_sessions()

    # Provide session list to parser via _load_known_sessions patch
    if sessions is not None:
        import unittest.mock
        with unittest.mock.patch('parser._load_known_sessions', return_value=sessions):
            cmd = parse(text)
    else:
        cmd = parse(text)

    # If parse() found a target, validate it against known sessions.
    # If the target matches a known session (or the sticky target), use it.
    # Otherwise, fall through to the anywhere-scan — parse() may have
    # misidentified a non-session word as the target (e.g., "send this to
    # firmware" → parse() extracts target="this", but the real target is
    # "firmware" detected by the anywhere-scan).
    if cmd.target is not None:
        target_is_known = (
            _fuzzy_session_match(cmd.target, effective_sessions) is not None
            if effective_sessions else True  # no sessions to validate against
        )
        target_matches_sticky = (
            sticky_target is not None and _is_same_target(cmd.target, sticky_target)
        )

        if target_is_known or target_matches_sticky:
            # No sticky target — any detected target is a switch
            if sticky_target is None:
                return {"action": "switch", "target": cmd.target, "text": cmd.text}

            # Compare parsed target to sticky target using fuzzy matching
            if _is_same_target(cmd.target, sticky_target):
                # Self-routing: send FULL original text (the routing phrase
                # is part of the user's thought, not a command to strip)
                return {"action": "self", "text": text}

            # Different target — switch
            return {"action": "switch", "target": cmd.target, "text": cmd.text}

    # parse() found no target, or its target didn't match any known session.
    # Try anywhere-scan as fallback.
    anywhere_target = _scan_anywhere_target(text, effective_sessions)
    if anywhere_target is not None:
        # Anywhere-scan found a target — treat full text as the command
        if sticky_target is not None and _is_same_target(anywhere_target, sticky_target):
            return {"action": "self", "text": text}

        return {"action": "switch", "target": anywhere_target, "text": text}

    # No target found anywhere — it's bare content
    return {"action": "content", "text": text}


def _is_same_target(parsed: str, sticky: str) -> bool:
    """Check if parsed target matches the sticky target (fuzzy).

    Uses the same thresholds as router._fuzzy_match:
      - Direct substring match
      - Joined-form substring match
      - ratio >= 65 or partial_ratio >= 80
    """
    p = parsed.lower()
    s = sticky.lower()

    if p == s:
        return True

    # Direct substring match
    if p in s or s in p:
        return True

    # Joined forms (remove spaces/hyphens)
    p_joined = re.sub(r'[\s\-]+', '', p)
    s_joined = re.sub(r'[\s\-]+', '', s)
    if p_joined in s_joined or s_joined in p_joined:
        return True

    # Fuzzy matching with same thresholds as router._fuzzy_match
    if fuzz.ratio(p_joined, s_joined) >= 65:
        return True
    if fuzz.partial_ratio(p_joined, s_joined) >= 80:
        return True

    return False
