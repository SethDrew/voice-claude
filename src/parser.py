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

REGISTRY_FILE = Path.home() / ".local" / "share" / "voice-router" / "name-registry.json"

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

    for name in sessions:
        name_lower = name.lower()
        name_joined = re.sub(r'[\s\-]+', '', name_lower)

        # Direct substring match (fast path)
        if word_lower in name_lower or name_lower in word_lower:
            return name

        # Joined substring match
        if word_joined in name_joined or name_joined in word_joined:
            return name

        # Fuzzy scoring on joined forms
        score = max(
            fuzz.ratio(word_joined, name_joined),
            fuzz.partial_ratio(word_joined, name_joined),
        )
        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= 65:
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
