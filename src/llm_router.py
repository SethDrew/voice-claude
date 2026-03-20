#!/usr/bin/env python3
"""MLX-LM-based LLM routing for voice commands.

Uses a local MLX language model to parse ambiguous voice commands into
structured {target, text} pairs when fuzzy matching fails.
"""

import json
import logging

MLX_MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

logger = logging.getLogger(__name__)

# Lazy-loaded singleton for model and tokenizer
_model = None
_tokenizer = None


def _load_model():
    """Load and cache the MLX model and tokenizer (lazy singleton)."""
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load
        logger.info("Loading MLX model %s (first call, may take a few seconds)...", MLX_MODEL_NAME)
        _model, _tokenizer = load(MLX_MODEL_NAME)
        logger.info("MLX model loaded successfully.")
    return _model, _tokenizer


def is_available() -> bool:
    """Check if mlx_lm is importable and functional."""
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def llm_parse(raw_text: str, sessions: list[str]) -> dict | None:
    """Use a local MLX model to parse a voice command into target and text.

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

    try:
        from mlx_lm import generate

        model, tokenizer = _load_model()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        content = generate(model, tokenizer, prompt=prompt, max_tokens=100)

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

    except (ImportError, json.JSONDecodeError, KeyError, ValueError, Exception) as e:
        logger.debug("llm_parse failed: %s", e)
        return None
