#!/usr/bin/env python3
"""Warm listen daemon — stays running with model pre-loaded.

Protocol (signal-based, using self-pipe for safety):
  SIGUSR1 = start recording
  SIGUSR2 = stop recording → transcribe → write result
  SIGTERM = shutdown

Signal handlers are minimal (write a byte to a pipe). All real work
runs in the main loop, avoiding signal-handler reentrancy issues with
MLX's Metal GPU threads.

Chunked streaming mode:
  During recording, a background thread monitors audio levels for pauses
  (silence > CHUNK_SILENCE_DURATION after speech). When a pause is detected,
  the chunk is transcribed and written to daemon-chunks.jsonl. On key release,
  the remaining audio is transcribed as the final chunk. The full concatenated
  text is also written to daemon-results.jsonl for backward compatibility.

State written to ~/.local/share/voice-claude/daemon-state.json:
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

# Config is in the same install dir (PYTHONPATH includes it)
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
STATE_DIR = os.path.expanduser("~/.local/share/voice-claude")
STATE_FILE = os.path.join(STATE_DIR, "daemon-state.json")
PID_FILE = os.path.join(STATE_DIR, "listen-daemon.pid")
RESULTS_FILE = os.path.join(STATE_DIR, "daemon-results.jsonl")
CHUNKS_FILE = os.path.join(STATE_DIR, "daemon-chunks.jsonl")

# Audio config
SAMPLE_RATE = config.SAMPLE_RATE
CHANNELS = config.CHANNELS

# Chunked streaming config
VAD_THRESHOLD = getattr(config, 'VAD_THRESHOLD', 0.015)
CHUNK_SILENCE_DURATION = 0.4  # seconds of silence to trigger chunk boundary

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

# Chunk streaming state
_chunk_num = 0                # chunk counter within current recording
_chunk_frames = []            # frames for the current (uncommitted) chunk
_chunk_lock = threading.Lock()  # protects _chunk_frames and _chunk_num
_chunk_texts = []             # accumulated chunk texts for final concatenation
_chunk_monitor_thread = None  # background thread for pause detection

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
        frame_copy = data.copy()
        rec_frames.append(frame_copy)
        with _chunk_lock:
            _chunk_frames.append(frame_copy)


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


def _filter_hallucination(text: str) -> str:
    """Filter Whisper hallucinations: repeated tokens, silence artifacts.

    Returns cleaned text, or empty string if the entire text is a hallucination.
    """
    if not text:
        return ""

    # Check for repeated token pattern (e.g., "Factor Factor Factor...")
    words = text.split()
    if len(words) >= 4:
        # If any single word makes up 60%+ of the text, it's a hallucination
        from collections import Counter
        counts = Counter(w.lower() for w in words)
        most_common_word, most_common_count = counts.most_common(1)[0]
        if most_common_count / len(words) >= 0.6:
            print(f"[daemon] filtered hallucination: repeated '{most_common_word}' ({most_common_count}/{len(words)})", flush=True)
            return ""

    # Known hallucination phrases
    lower = text.lower().strip().rstrip('.!,')
    hallucinations = {
        "thank you", "thanks", "thank you for watching", "thanks for watching",
        "subscribe", "please subscribe", "like and subscribe",
        "you", "bye", "goodbye", "the end",
    }
    if lower in hallucinations:
        return ""

    return text


def _is_silent(frames_list) -> bool:
    """Check if audio frames are mostly silence."""
    if not frames_list:
        return True
    data = np.concatenate(frames_list)
    if data.ndim > 1:
        data = data.flatten()
    rms = float(np.sqrt(np.mean(data**2)))
    return rms < VAD_THRESHOLD


def _transcribe_audio(frames_list):
    """Transcribe a list of audio frames, returning text. Uses _transcribe_lock.

    Returns the transcribed text string, or empty string on error.
    """
    # Skip transcription for silent audio
    if _is_silent(frames_list):
        print("[daemon] skipping silent chunk", flush=True)
        return ""

    with _transcribe_lock:
        try:
            data = np.concatenate(frames_list)
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

            # Filter hallucinations: repeated tokens, silence artifacts
            text = _filter_hallucination(text)

            print(f"[daemon] chunk transcribed in {elapsed:.2f}s: {text!r}", flush=True)

            os.unlink(tmp.name)
            return text

        except Exception as e:
            print(f"[daemon] chunk transcription error: {e}", flush=True)
            return ""


def _write_chunk(seq_num, chunk_num, text, final=False):
    """Append a chunk record to daemon-chunks.jsonl."""
    record = {
        "seq": seq_num,
        "chunk": chunk_num,
        "text": text,
        "final": final,
    }
    with open(CHUNKS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _chunk_monitor():
    """Background thread: detect speech pauses and transcribe chunks.

    Runs while `recording` is True. Monitors audio levels in _chunk_frames
    for silence gaps longer than CHUNK_SILENCE_DURATION. When a pause is
    detected (silence after speech), the accumulated chunk frames are
    transcribed and written to daemon-chunks.jsonl.
    """
    global _chunk_num

    # Number of frames constituting the silence duration
    # Each frame from sounddevice is CHUNK_SIZE samples (default ~1024 at 16kHz)
    chunk_size = getattr(config, 'CHUNK_SIZE', 1024)
    frames_per_second = SAMPLE_RATE / chunk_size
    silence_frame_count = int(CHUNK_SILENCE_DURATION * frames_per_second)

    had_speech = False
    silence_count = 0

    while recording:
        time.sleep(0.05)  # poll at 20Hz

        with _chunk_lock:
            if not _chunk_frames:
                continue

            # Check the most recent frame for speech/silence
            latest = _chunk_frames[-1]
            level = np.abs(latest).mean()

        if level >= VAD_THRESHOLD:
            had_speech = True
            silence_count = 0
        else:
            silence_count += 1

        # If we had speech and now have enough silence, emit a chunk
        if had_speech and silence_count >= silence_frame_count:
            with _chunk_lock:
                if not _chunk_frames:
                    continue
                chunk_snapshot = list(_chunk_frames)
                _chunk_frames.clear()

            if chunk_snapshot:
                text = _transcribe_audio(chunk_snapshot)
                if text:
                    _chunk_num += 1
                    _write_chunk(seq, _chunk_num, text, final=False)
                    _chunk_texts.append(text)

            had_speech = False
            silence_count = 0


def do_start_recording():
    """Start recording — called from main loop, NOT from a signal handler."""
    global recording, rec_frames, stream, seq
    global _chunk_num, _chunk_frames, _chunk_texts, _chunk_monitor_thread
    if recording:
        return

    seq += 1
    rec_frames = []
    recording = True

    # Initialize chunk streaming state
    _chunk_num = 0
    with _chunk_lock:
        _chunk_frames.clear()
    _chunk_texts.clear()

    # Truncate chunks file for new recording
    try:
        with open(CHUNKS_FILE, "w") as f:
            pass
    except OSError:
        pass

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=audio_callback,
        )
        stream.start()

        # Start chunk monitor thread
        _chunk_monitor_thread = threading.Thread(target=_chunk_monitor, daemon=True)
        _chunk_monitor_thread.start()

        write_state("recording")
        print(f"[daemon] recording started", flush=True)
    except Exception as e:
        recording = False
        write_state("error", error=str(e))
        print(f"[daemon] failed to start recording: {e}", flush=True)


def do_stop_recording():
    """Stop recording and spawn transcription — called from main loop.

    On stop, any remaining chunk frames are transcribed as the final chunk.
    The full concatenated text (all chunks) is also written to the results
    file for backward compatibility.
    """
    global recording, stream, _chunk_monitor_thread
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

    # Wait for chunk monitor to stop (it checks `recording` flag)
    if _chunk_monitor_thread and _chunk_monitor_thread.is_alive():
        _chunk_monitor_thread.join(timeout=2.0)
    _chunk_monitor_thread = None

    print(f"[daemon] recording stopped, {len(rec_frames)} frames", flush=True)

    if not rec_frames:
        write_state("done", text="")
        return

    # Transcribe any remaining chunk frames as the final chunk
    frames_copy = list(rec_frames)
    current_seq = seq
    rec_frames.clear()

    def _finalize():
        global _chunk_num
        # Grab any remaining chunk frames not yet transcribed
        with _chunk_lock:
            remaining = list(_chunk_frames)
            _chunk_frames.clear()

        if remaining:
            text = _transcribe_audio(remaining)
            if text:
                _chunk_num += 1
                _write_chunk(current_seq, _chunk_num, text, final=True)
                _chunk_texts.append(text)
            else:
                # Write empty final marker
                _chunk_num += 1
                _write_chunk(current_seq, _chunk_num, "", final=True)
        elif _chunk_texts:
            # No remaining frames but we had earlier chunks — mark last as final
            # Rewrite the final flag by appending a final marker
            _chunk_num += 1
            _write_chunk(current_seq, _chunk_num, "", final=True)
        else:
            # No chunks at all — do a full transcription of all frames
            text = _transcribe_audio(frames_copy)
            _chunk_num += 1
            _write_chunk(current_seq, _chunk_num, text, final=True)
            _chunk_texts.append(text)

        # Write full concatenated text to results file (backward compat)
        full_text = " ".join(t for t in _chunk_texts if t).strip()
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps({"seq": current_seq, "text": full_text}) + "\n")

        write_state("done", text=full_text, seq=current_seq)
        print(f"[daemon] [{current_seq}] final text: {full_text!r}", flush=True)

    threading.Thread(target=_finalize, daemon=True).start()


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

    # Singleton guard: exit if another daemon is already running
    if os.path.exists(PID_FILE):
        try:
            old_pid = int(open(PID_FILE).read().strip())
            if old_pid != os.getpid():
                os.kill(old_pid, 0)  # check if alive
                print(f"[daemon] another instance running (pid={old_pid}), exiting", flush=True)
                sys.exit(0)
        except (ProcessLookupError, ValueError, OSError):
            pass  # stale PID file, process already dead

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    if BACKEND == "mlx":
        model_name = MLX_MODEL
    else:
        model_name = FALLBACK_MODEL

    print(f"[daemon] loading whisper model '{model_name}' (backend={BACKEND})...", flush=True)
    t0 = time.time()
    if BACKEND == "mlx":
        warmup_file = os.path.join(tempfile.gettempdir(), "voice-claude-warmup.wav")
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
