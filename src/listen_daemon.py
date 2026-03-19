#!/usr/bin/env python3
"""Warm listen daemon — stays running with model pre-loaded.

Protocol (signal-based):
  SIGUSR1 = start recording
  SIGUSR2 = stop recording → transcribe → write result
  SIGTERM = shutdown

State written to ~/.local/share/voice-router/daemon-state.json:
  {"state": "idle"}
  {"state": "recording"}
  {"state": "transcribing"}
  {"state": "done", "text": "..."}
  {"state": "error", "error": "..."}
"""

import json
import os
import signal
import sys
import tempfile
import time
import wave

import numpy as np
import sounddevice as sd

# Add listen dir to path for config
listen_dir = os.path.expanduser("~/.local/share/listen")
sys.path.insert(0, listen_dir)
import config

# Try faster-whisper first, fall back to openai-whisper
try:
    from faster_whisper import WhisperModel
    FASTER = True
except ImportError:
    import whisper
    FASTER = False

# Paths
STATE_DIR = os.path.expanduser("~/.local/share/voice-router")
STATE_FILE = os.path.join(STATE_DIR, "daemon-state.json")
PID_FILE = os.path.join(STATE_DIR, "listen-daemon.pid")

# Audio config
SAMPLE_RATE = config.SAMPLE_RATE
CHANNELS = config.CHANNELS
MODEL_NAME = "base"

# Recording state
recording = False
rec_frames = []
stream = None
model = None


def write_state(state, **extra):
    data = {"state": state, "pid": os.getpid(), "ts": time.time()}
    data.update(extra)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, STATE_FILE)


def audio_callback(data, frames, t, status):
    if recording:
        rec_frames.append(data.copy())


def start_recording(signum, frame):
    global recording, rec_frames, stream
    if recording:
        return

    rec_frames = []
    recording = True

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=audio_callback,
        )
        stream.start()
        write_state("recording")
        print(f"[daemon] recording started", flush=True)
    except Exception as e:
        recording = False
        write_state("error", error=str(e))
        print(f"[daemon] failed to start recording: {e}", flush=True)


def stop_recording(signum, frame):
    global recording, stream
    if not recording:
        return

    recording = False

    # Stop mic immediately
    if stream:
        try:
            stream.abort()
            stream.close()
        except Exception:
            pass
        stream = None

    print(f"[daemon] recording stopped, {len(rec_frames)} frames", flush=True)

    if not rec_frames:
        write_state("done", text="")
        return

    write_state("transcribing")

    try:
        # Concatenate audio
        data = np.concatenate(rec_frames)
        if data.ndim > 1:
            data = data.flatten()

        # Save to temp wav
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(CHANNELS)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((data * 32767).astype(np.int16).tobytes())

        # Transcribe
        t0 = time.time()
        if FASTER:
            segments, info = model.transcribe(tmp.name, language="en", beam_size=5)
            text = "".join(s.text for s in segments).strip()
        else:
            r = model.transcribe(tmp.name, language="en", fp16=False, verbose=False)
            text = r["text"].strip()

        elapsed = time.time() - t0
        print(f"[daemon] transcribed in {elapsed:.2f}s: {text!r}", flush=True)

        os.unlink(tmp.name)
        write_state("done", text=text)

    except Exception as e:
        write_state("error", error=str(e))
        print(f"[daemon] transcription error: {e}", flush=True)


def shutdown(signum, frame):
    global recording, stream
    print("[daemon] shutting down", flush=True)
    if stream:
        try:
            stream.abort()
            stream.close()
        except Exception:
            pass
    try:
        os.unlink(PID_FILE)
    except OSError:
        pass
    write_state("stopped")
    sys.exit(0)


def main():
    global model

    os.makedirs(STATE_DIR, exist_ok=True)

    # Write PID
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Load model (the slow part — only happens once)
    print(f"[daemon] loading whisper model '{MODEL_NAME}'...", flush=True)
    t0 = time.time()
    if FASTER:
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        print(f"[daemon] faster-whisper loaded in {time.time()-t0:.2f}s", flush=True)
    else:
        model = whisper.load_model(MODEL_NAME)
        print(f"[daemon] whisper loaded in {time.time()-t0:.2f}s", flush=True)

    # Register signals
    signal.signal(signal.SIGUSR1, start_recording)
    signal.signal(signal.SIGUSR2, stop_recording)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    write_state("idle")
    print(f"[daemon] ready (pid={os.getpid()})", flush=True)

    # Sleep forever, signals do the work
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
