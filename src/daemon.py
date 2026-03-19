#!/usr/bin/env python3
"""Wake word detection daemon — listens continuously, triggers STT on wake word."""

import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

MODEL_DIR = Path.home() / ".local" / "share" / "voice-router" / "models"
COOLDOWN = 2.0  # seconds between activations
CHUNK_SIZE = 1280  # 80ms at 16kHz — required by OpenWakeWord
SAMPLE_RATE = 16000

running = True


def shutdown_handler(signum, frame):
    global running
    print("\nShutting down wake word daemon...")
    running = False


def find_models() -> list[Path]:
    """Find all .onnx wake word models."""
    if not MODEL_DIR.exists():
        return []
    return sorted(MODEL_DIR.glob("*.onnx"))


def main():
    global running

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        from openwakeword.model import Model
    except ImportError:
        print("Error: openwakeword not installed. Run setup.sh first.", file=sys.stderr)
        sys.exit(1)

    models = find_models()
    if not models:
        print(f"No .onnx models found in {MODEL_DIR}", file=sys.stderr)
        print("Train wake word models and place them in the models/ directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(models)} wake word model(s):")
    for m in models:
        print(f"  - {m.stem}")

    oww = Model(
        wakeword_models=[str(m) for m in models],
        inference_framework="onnx",
    )

    print(f"Listening for wake words... (Ctrl+C to stop)")
    last_activation = 0

    audio_buffer = []

    def audio_callback(indata, frames, time_info, status):
        audio_buffer.append(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    )

    with stream:
        while running:
            if not audio_buffer:
                time.sleep(0.01)
                continue

            # Process all buffered audio
            chunk = audio_buffer.pop(0)

            # Feed to OpenWakeWord
            oww.predict(chunk)

            # Check detections
            for model_name, score in oww.get_prediction().items():
                if score > 0.5 and (time.time() - last_activation) > COOLDOWN:
                    print(f"\n Wake word detected: {model_name} (score: {score:.2f})")
                    last_activation = time.time()

                    # Stop listening, run STT
                    stream.stop()
                    try:
                        result = subprocess.run(
                            ["voice-route", "--hotkey"],
                            capture_output=True,
                            text=True,
                        )
                        if result.stdout:
                            print(result.stdout, end="")
                        if result.stderr:
                            print(result.stderr, end="", file=sys.stderr)
                    except Exception as e:
                        print(f"Error routing: {e}", file=sys.stderr)

                    # Reset and restart
                    oww.reset()
                    audio_buffer.clear()
                    stream.start()
                    print("Listening for wake words...")
                    break

    print("Daemon stopped.")


if __name__ == "__main__":
    main()
