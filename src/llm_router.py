#!/usr/bin/env python3
"""Ollama-based LLM routing for voice commands.

Uses a local Ollama instance to parse ambiguous voice commands into
structured {target, text} pairs when fuzzy matching fails.
"""

import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
TIMEOUT = 2  # seconds


def is_available() -> bool:
    """Check if Ollama is running on localhost:11434."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT):
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def llm_parse(raw_text: str, sessions: list[str]) -> dict | None:
    """Use Ollama to parse a voice command into target and text.

    Args:
        raw_text: The transcribed voice text.
        sessions: List of known session names.

    Returns:
        Dict with "target" and "text" keys, or None on failure.
    """
    if not sessions:
        return None

    session_list = ", ".join(sessions)
    system_prompt = (
        "You are a voice command router. Given a voice transcription and a list of "
        "session names, extract the target session and the command text.\n\n"
        f"Known sessions: {session_list}\n\n"
        "Respond with ONLY a JSON object: {\"target\": \"session_name\", \"text\": \"command text\"}\n"
        "If you cannot determine a target, set target to null.\n"
        "If there is no command text, set text to null.\n"
        "Do not include any other text in your response."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 100,
        },
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        content = body.get("message", {}).get("content", "")

        # Try to parse JSON from the response
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        result = json.loads(content)

        # Validate target against known sessions
        target = result.get("target")
        if target and target not in sessions:
            # Try case-insensitive match
            target_lower = target.lower()
            matched = None
            for s in sessions:
                if s.lower() == target_lower:
                    matched = s
                    break
            target = matched

        return {
            "target": target,
            "text": result.get("text"),
        }

    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError,
            KeyError, ValueError):
        return None
