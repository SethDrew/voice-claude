#!/usr/bin/env python3
"""Warm listen daemon — stays running with model pre-loaded.

Protocol (signal-based, using self-pipe for safety):
  SIGUSR1 = start recording
  SIGUSR2 = stop recording → transcribe → write result
  SIGTERM = shutdown

Signal handlers are minimal (write a byte to a pipe). All real work
runs in the main loop, avoiding signal-handler reentrancy issues with
MLX's Metal GPU threads.

State written to ~/.local/share/voice-router/daemon-state.json:
  {"state": "idle"}
  {"state": "recording"}
  {"state": "transcribing"}
  {"state": "done", "text": "..."}
  {"state": "error", "error": "..."}
"""

import json
import os
import select
import signal
import sys
import tempfile
import threading
import time
import wave

import numpy as np
import sounddevice as sd

# Add listen dir to path for config
listen_dir = os.path.expanduser("~/.local/share/listen")
sys.path.insert(0, listen_dir)
import config

# Three-tier import: mlx_whisper > faster_whisper > openai-whisper
BACKEND = None
try:
    import mlx_whisper
    BACKEND = "mlx"
except ImportError:
    try:
        from faster_whisper import WhisperModel
        BACKEND = "faster"
    except ImportError:
        import whisper
        BACKEND = "openai"

# Paths
STATE_DIR = os.path.expanduser("~/.local/share/voice-router")
STATE_FILE = os.path.join(STATE_DIR, "daemon-state.json")
PID_FILE = os.path.join(STATE_DIR, "listen-daemon.pid")
RESULTS_FILE = os.path.join(STATE_DIR, "daemon-results.jsonl")

# Audio config
SAMPLE_RATE = config.SAMPLE_RATE
CHANNELS = config.CHANNELS

# Model constants
MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
FALLBACK_MODEL = "base"

# Initial prompt with coding vocabulary to guide transcription
INITIAL_PROMPT = (
    "Claude Code, firmware, frontend, backend, GPIO, API, SDK, CLI, "
    "slash commit, slash review, slash compact, slash clear, slash diff, "
    "hey skynet, hey destroyer, hey code, "
    "refactor, deploy, debug, repository, pull request, merge, rebase"
)

# Recording state
recording = False
rec_frames = []
stream = None
model = None
seq = 0

# Lock to serialize MLX transcription calls — prevents concurrent Metal GPU access
_transcribe_lock = threading.Lock()


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


def transcribe_frames(frames_copy, seq_num):
    """Transcribe audio — serialized by _transcribe_lock to prevent concurrent MLX access."""
    with _transcribe_lock:
        write_state("transcribing", seq=seq_num)

        try:
            data = np.concatenate(frames_copy)
            if data.ndim > 1:
                data = data.flatten()

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp.name, "wb") as w:
                w.setnchannels(CHANNELS)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes((data * 32767).astype(np.int16).tobytes())

            t0 = time.time()
            if BACKEND == "mlx":
                result = mlx_whisper.transcribe(
                    tmp.name,
                    path_or_hf_repo=MLX_MODEL,
                    language="en",
                    initial_prompt=INITIAL_PROMPT,
                )
                text = result["text"].strip()
            elif BACKEND == "faster":
                segments, info = model.transcribe(
                    tmp.name, language="en", beam_size=5,
                    initial_prompt=INITIAL_PROMPT,
                )
                text = "".join(s.text for s in segments).strip()
            else:
                r = model.transcribe(
                    tmp.name, language="en", fp16=False, verbose=False,
                    initial_prompt=INITIAL_PROMPT,
                )
                text = r["text"].strip()

            elapsed = time.time() - t0
            print(f"[daemon] [{seq_num}] transcribed in {elapsed:.2f}s: {text!r}", flush=True)

            os.unlink(tmp.name)

            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps({"seq": seq_num, "text": text}) + "\n")

            write_state("done", text=text, seq=seq_num)

        except Exception as e:
            write_state("error", error=str(e), seq=seq_num)
            print(f"[daemon] [{seq_num}] transcription error: {e}", flush=True)


def do_start_recording():
    """Start recording — called from main loop, NOT from a signal handler."""
    global recording, rec_frames, stream, seq
    if recording:
        return

    seq += 1
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


def do_stop_recording():
    """Stop recording and spawn transcription — called from main loop."""
    global recording, stream
    if not recording:
        return

    recording = False

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

    frames_copy = list(rec_frames)
    current_seq = seq
    rec_frames.clear()
    threading.Thread(target=transcribe_frames, args=(frames_copy, current_seq), daemon=True).start()


def do_shutdown():
    """Clean shutdown — called from main loop."""
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

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    if BACKEND == "mlx":
        model_name = MLX_MODEL
    else:
        model_name = FALLBACK_MODEL

    print(f"[daemon] loading whisper model '{model_name}' (backend={BACKEND})...", flush=True)
    t0 = time.time()
    if BACKEND == "mlx":
        warmup_file = os.path.join(tempfile.gettempdir(), "voice-router-warmup.wav")
        with wave.open(warmup_file, "wb") as w:
            w.setnchannels(CHANNELS)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(b'\x00' * (SAMPLE_RATE * CHANNELS * 2 // 2))
        try:
            mlx_whisper.transcribe(warmup_file, path_or_hf_repo=MLX_MODEL, language="en")
        except Exception as e:
            print(f"[daemon] mlx warmup note: {e}", flush=True)
        try:
            os.unlink(warmup_file)
        except OSError:
            pass
        model = None
        print(f"[daemon] mlx-whisper warmed up in {time.time()-t0:.2f}s", flush=True)
    elif BACKEND == "faster":
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f"[daemon] faster-whisper loaded in {time.time()-t0:.2f}s", flush=True)
    else:
        model = whisper.load_model(model_name)
        print(f"[daemon] whisper loaded in {time.time()-t0:.2f}s", flush=True)

    # Self-pipe pattern: signal handlers write a byte to the pipe,
    # main loop reads from it. This keeps signal handlers trivial
    # and avoids interrupting MLX Metal GPU operations.
    sig_read, sig_write = os.pipe()
    os.set_blocking(sig_write, False)

    def _sig_handler(signum, frame):
        try:
            os.write(sig_write, bytes([signum & 0xFF]))
        except OSError:
            pass

    signal.signal(signal.SIGUSR1, _sig_handler)
    signal.signal(signal.SIGUSR2, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    write_state("idle")
    print(f"[daemon] ready (pid={os.getpid()})", flush=True)

    # Main event loop — all real work happens here, not in signal handlers
    os.set_blocking(sig_read, False)
    while True:
        try:
            select.select([sig_read], [], [], 1.0)  # 1s timeout as safety
        except (InterruptedError, OSError):
            continue

        try:
            data = os.read(sig_read, 64)
        except (BlockingIOError, OSError):
            continue

        if not data:
            continue

        for b in data:
            if b == (signal.SIGUSR1 & 0xFF):
                do_start_recording()
            elif b == (signal.SIGUSR2 & 0xFF):
                do_stop_recording()
            elif b in ((signal.SIGTERM & 0xFF), (signal.SIGINT & 0xFF)):
                do_shutdown()


if __name__ == "__main__":
    main()
