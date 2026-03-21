#!/usr/bin/env python3
"""V2 Listen daemon — Moonshine Voice streaming STT.

True streaming transcription: text appears as you speak, not after you stop.
Uses Moonshine Voice (ONNX-based, Apple Silicon optimized) instead of
batch Whisper.

Protocol (signal-based, self-pipe):
  SIGUSR1 = start recording + streaming transcription
  SIGUSR2 = stop recording, finalize
  SIGTERM = shutdown

Output files:
  daemon-state.json    — {"state": "idle|recording|done", ...}
  daemon-partial.json  — {"text": "live text...", "final": false}  (updated live)
  daemon-results.jsonl — {"seq": N, "text": "final text"}  (appended on finalize)
"""

import json
import os
import select
import signal
import sys
import threading
import time

# Paths
STATE_DIR = os.path.expanduser("~/.local/share/voice-claude")
STATE_FILE = os.path.join(STATE_DIR, "daemon-state.json")
PID_FILE = os.path.join(STATE_DIR, "listen-daemon.pid")
RESULTS_FILE = os.path.join(STATE_DIR, "daemon-results.jsonl")
PARTIAL_FILE = os.path.join(STATE_DIR, "daemon-partial.json")

# State
recording = False
seq = 0
_transcriber = None
_mic = None
_live_text = ""
_finalized_lines = []
_lock = threading.Lock()


def write_state(state, **extra):
    data = {"state": state, "pid": os.getpid(), "ts": time.time()}
    data.update(extra)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, STATE_FILE)


def write_partial(text, final=False):
    """Write live transcription to partial file for Hammerspoon to display."""
    data = {"text": text, "final": final, "ts": time.time()}
    tmp = PARTIAL_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, PARTIAL_FILE)


class StreamListener:
    """Receives live transcription events from Moonshine."""

    def on_line_started(self, event):
        pass

    def on_line_text_changed(self, event):
        global _live_text
        with _lock:
            # Build full text: finalized lines + current live line
            parts = list(_finalized_lines)
            if event.line.text:
                parts.append(event.line.text)
            _live_text = " ".join(parts)
        write_partial(_live_text, final=False)
        print(f"[v2] live: {event.line.text}", flush=True)

    def on_line_completed(self, event):
        global _live_text
        with _lock:
            if event.line.text:
                _finalized_lines.append(event.line.text)
            _live_text = " ".join(_finalized_lines)
        write_partial(_live_text, final=False)
        print(f"[v2] line done: {event.line.text}", flush=True)


def do_start_recording():
    global recording, seq, _live_text, _finalized_lines
    if recording:
        return

    seq += 1
    recording = True
    _live_text = ""
    _finalized_lines = []

    # Clear partial file
    write_partial("", final=False)

    _mic.start()
    write_state("recording")
    print(f"[v2] recording started (seq={seq})", flush=True)


def do_stop_recording():
    global recording, _live_text
    if not recording:
        return

    recording = False
    _mic.stop()

    # Give a moment for final transcription
    time.sleep(0.3)

    with _lock:
        final_text = _live_text.strip()

    print(f"[v2] recording stopped, final: {final_text!r}", flush=True)

    # Filter hallucinations
    final_text = _filter_hallucination(final_text)

    # Write to results file (backward compat)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps({"seq": seq, "text": final_text}) + "\n")

    # Write final partial
    write_partial(final_text, final=True)
    write_state("done", text=final_text, seq=seq)


def _filter_hallucination(text):
    if not text:
        return ""
    words = text.split()
    if len(words) >= 4:
        from collections import Counter
        counts = Counter(w.lower() for w in words)
        most_common_word, most_common_count = counts.most_common(1)[0]
        if most_common_count / len(words) >= 0.6:
            print(f"[v2] filtered hallucination: '{most_common_word}' repeated", flush=True)
            return ""
    lower = text.lower().strip().rstrip('.!,')
    if lower in {"thank you", "thanks", "thank you for watching", "thanks for watching",
                 "subscribe", "you", "bye", "goodbye", "the end"}:
        return ""
    return text


def do_shutdown():
    global recording
    print("[v2] shutting down", flush=True)
    if recording:
        _mic.stop()
    if _mic:
        _mic.close()
    if _transcriber:
        pass  # transcriber cleanup if needed
    try:
        os.unlink(PID_FILE)
    except OSError:
        pass
    write_state("stopped")
    sys.exit(0)


def main():
    global _transcriber, _mic

    os.makedirs(STATE_DIR, exist_ok=True)

    # Singleton guard
    if os.path.exists(PID_FILE):
        try:
            old_pid = int(open(PID_FILE).read().strip())
            if old_pid != os.getpid():
                os.kill(old_pid, 0)
                print(f"[v2] another instance running (pid={old_pid}), exiting", flush=True)
                sys.exit(0)
        except (ProcessLookupError, ValueError, OSError):
            pass

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Load Moonshine
    from moonshine_voice import MicTranscriber, get_model_for_language
    from moonshine_voice.transcriber import TranscriptEventListener

    print("[v2] loading Moonshine model...", flush=True)
    t0 = time.time()
    model_path, model_arch = get_model_for_language("en")
    _mic = MicTranscriber(
        model_path=model_path,
        model_arch=model_arch,
        update_interval=0.5,
    )
    _mic.add_listener(StreamListener())
    print(f"[v2] Moonshine loaded in {time.time()-t0:.2f}s", flush=True)

    # Self-pipe for signals
    sig_read, sig_write = os.pipe()
    os.set_blocking(sig_write, False)
    os.set_blocking(sig_read, False)

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
    print(f"[v2] ready (pid={os.getpid()})", flush=True)

    while True:
        try:
            select.select([sig_read], [], [], 1.0)
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
