#!/usr/bin/env python3
"""Voice command parser for routing speech to Claude Code sessions."""

import re
from dataclasses import dataclass
from typing import Optional

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

def parse(raw: str) -> ParsedCommand:
    """Parse a voice command into target and text.

    Patterns:
        "tell <target>: <text>"  -> target + text
        "tell <target> <text>"   -> target + text
        "go to <target>"         -> target, focus only
        bare text                -> last-active session
    """
    text = strip_wake_phrase(raw)

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

    # "<verb> <target> <text>" without colon
    m = re.match(rf'^{verbs}\s+(\S+)\s+(.+)$', text, re.IGNORECASE)
    if m:
        return ParsedCommand(
            target=m.group(1).lower(),
            text=replace_slash_commands(m.group(2).strip()),
        )

    # "in <target>, <text>" / "on <target>, <text>"
    m = re.match(r'^(?:in|on)\s+(\S+)\s*[,:]?\s+(.+)$', text, re.IGNORECASE)
    if m:
        return ParsedCommand(
            target=m.group(1).lower(),
            text=replace_slash_commands(m.group(2).strip()),
        )

    # Bare text — route to last-active
    return ParsedCommand(target=None, text=text)
