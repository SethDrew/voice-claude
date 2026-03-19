#!/usr/bin/env python3
import sys, os, tempfile, wave, time, threading, queue, fcntl, termios, re, signal, argparse
import numpy as np
import sounddevice as sd
import whisper
import config

# Try to import faster-whisper for performance optimization
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

__version__ = '2.1.0'

# ANSI escape codes
CLR = '\033[K'
HOME = '\033[2J\033[H'
RED = '\033[91m'
YEL = '\033[93m'
MAG = '\033[95m'  # Magenta/Purple
RST = '\033[0m'

# Audio configuration (from config.py)
SAMPLE_RATE = config.SAMPLE_RATE
CHANNELS = config.CHANNELS
CHUNK_SIZE = config.CHUNK_SIZE
DTYPE = 'float32'

# Recording state
rec = []
lvl = [0.0]
pct = [0.0]
verbose = False
is_tty = sys.stdout.isatty()
stdin_is_tty = sys.stdin.isatty()
signal_stop = [False]
signal_mode = False
vad_enabled = False
vad_silence_duration = config.VAD_DEFAULT_DURATION
vad_threshold = config.VAD_THRESHOLD

# Scripting mode flags
quiet_mode = False
json_mode = False
output_file = None
codevoice_mode = False
status_file = None
fast_mode = False
preloaded_model = None

# Help text
HELP = """usage: listen [MODE] [OPTIONS]

Modes:
  (default)           Record audio from microphone and transcribe
  -f, --file FILE     Transcribe audio from file (mp3, wav, m4a, etc.)

Options:
  -l, --language LANG     Language code (default: {lang})
  -m, --model MODEL       Whisper model (default: {model})
  --signal-mode           Use SIGUSR1 signal to stop recording
  --vad SECONDS           Auto-stop after N seconds of silence
  --codevoice             Full-width visual mode for code voice input
  --fast-mode             Use faster-whisper for 3-4x speed (requires faster-whisper)
  --version               Show version and exit
  -v, --verbose           Verbose output

Scripting options:
  -q, --quiet             Suppress UI, output only transcription
  -j, --json              Output in JSON format
  -o, --output FILE       Write transcription to file
  --status-file FILE      Write real-time status to JSON file

Recording controls:
  (default) Press SPACE to stop recording
  (signal)  Send SIGUSR1 to process (kill -SIGUSR1 <pid>)
  (vad)     Auto-stop after silence duration

Configuration:
  Edit config.py and reinstall to change defaults
  CLI arguments override config.py values"""

def log(msg):
    if verbose:
        print(f'\n[DEBUG] {msg}', file=sys.stderr)


def write_status(data):
    """Write status to JSON file atomically"""
    if not status_file:
        return

    try:
        import json
        # Atomic write: write to temp file then rename
        temp_file = f"{status_file}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(temp_file, status_file)
        log(f'Status written: {data.get("status")}')
    except Exception as e:
        log(f'Error writing status file: {e}')


def output_transcription(text, lang, model_name, duration=None):
    """Handle transcription output based on flags"""
    import json as json_lib

    # Prepare data
    if json_mode:
        data = {
            "transcription": text,
            "language": lang,
            "model": model_name
        }
        if duration:
            data["duration"] = duration
        output_text = json_lib.dumps(data, ensure_ascii=False)
    else:
        output_text = text

    # Output to stdout
    print(output_text, flush=True)

    # Write to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
            if not quiet_mode and not json_mode:
                print(f'Written to {output_file}', file=sys.stderr)
        except Exception as e:
            print(f'Error writing to file: {e}', file=sys.stderr)


def signal_handler(signum, frame):
    """Handle SIGUSR1 to stop recording gracefully"""
    global signal_stop
    signal_stop[0] = True
    log(f'Received signal {signum}, stopping recording')


def audio_cb(data, frames, t, status):
    global np
    rec.append(data.copy())
    lvl[0] = float(np.sqrt(np.mean(data**2)))


def kbd_listen(q):
    if not stdin_is_tty:
        return

    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        flg = fcntl.fcntl(fd, fcntl.F_GETFL)
    except (termios.error, OSError):
        # Termux or environments where terminal control is not available
        log('Terminal control not available, keyboard listener disabled')
        return

    try:
        new = termios.tcgetattr(fd)
        new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, new)
        fcntl.fcntl(fd, fcntl.F_SETFL, flg | os.O_NONBLOCK)

        while True:
            try:
                if sys.stdin.read(1) == ' ':
                    q.put(1)
                    break
            except (IOError, OSError, BlockingIOError):
                pass
            time.sleep(0.005)

        termios.tcsetattr(fd, termios.TCSAFLUSH, old)
        fcntl.fcntl(fd, fcntl.F_SETFL, flg)
    except Exception as e:
        log(f'Keyboard listener error: {e}')
        pass


def draw(level, txt='Listening', hint='', fullwidth=False):
    """Draw progress bar - simple or full-width based on mode"""
    # Skip UI in quiet/json mode
    if quiet_mode or json_mode:
        return

    out = sys.stderr if not is_tty else sys.stdout
    hint_str = f'  {MAG}{hint}{RST}' if hint else ''

    if fullwidth:
        # Full-width codevoice mode
        try:
            import shutil
            terminal_width = shutil.get_terminal_size().columns
        except:
            terminal_width = 80

        prefix_len = len('● ' + txt + '  ')
        hint_len = len(hint) + 2 if hint else 0
        available_width = max(10, terminal_width - prefix_len - hint_len - 4)

        filled = max(0, min(int(level * available_width), available_width))
        bar = '█' * filled + '░' * (available_width - filled)
        out.write(f'\r{RED}●{RST} {txt}  [{YEL}{bar}{RST}]{hint_str}')
    else:
        # Simple mode - fixed width
        filled = min(int(level * 200), 10)
        bar = '=' * filled + ' ' * (10 - filled)
        out.write(f'\r{RED}●{RST} {txt}  [{bar}]{hint_str}')

    out.flush()


def record(start_proc, lang, mdl):
    global rec, signal_stop

    rec = []
    q = queue.Queue()

    # Only start keyboard listener if not in signal mode
    if not signal_mode:
        log('Starting keyboard listener thread')
        threading.Thread(target=kbd_listen, args=(q,), daemon=True).start()

    # Determine hint message based on mode
    hint = ''
    mode = 'space'
    if signal_mode:
        pid = os.getpid()
        hint = f'Signal mode: kill -SIGUSR1 {pid}'
        mode = 'signal'
    elif vad_enabled:
        hint = f'VAD mode: auto-stop after {vad_silence_duration}s silence'
        mode = 'vad'
    elif stdin_is_tty:
        hint = 'Press SPACE to stop'

    draw(0.0, hint=hint, fullwidth=codevoice_mode)

    # Write initial status
    write_status({
        'status': 'recording',
        'audio_level': 0.0,
        'pid': os.getpid(),
        'language': lang,
        'model': mdl,
        'timestamp': int(time.time()),
        'progress': 0.0,
        'duration': 0.0,
        'mode': mode,
        'transcription': None
    })

    log(f'Starting audio stream ({SAMPLE_RATE//1000}kHz, {"stereo" if CHANNELS == 2 else "mono"})')
    # In signal mode or VAD mode, no timeout - wait for signal/silence
    # Otherwise, use timeout when piped (not TTY)
    timeout = None if (signal_mode or vad_enabled) else (10 if not stdin_is_tty else None)

    # VAD tracking variables
    silence_start = None
    has_speech = False

    # Create stream without context manager for explicit control
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_cb)

    try:
        stream.start()
        log('Audio stream started')
        t0 = time.time()

        while True:
            draw(lvl[0], hint=hint, fullwidth=codevoice_mode)

            # Update status with current audio level
            write_status({
                'status': 'recording',
                'audio_level': float(lvl[0]),
                'pid': os.getpid(),
                'language': lang,
                'model': mdl,
                'timestamp': int(time.time()),
                'progress': 0.0,
                'duration': time.time() - t0,
                'mode': mode,
                'transcription': None
            })

            # Check signal stop (for signal mode)
            if signal_mode and signal_stop[0]:
                dur = time.time() - t0
                log(f'Recording stopped by signal after {dur:.2f}s')
                break

            # VAD: Check for silence
            if vad_enabled:
                current_level = lvl[0]

                # Detect if currently silent
                is_silent = current_level < vad_threshold

                if not is_silent:
                    # Speech detected
                    has_speech = True
                    silence_start = None
                    log(f'Speech detected (level: {current_level:.4f})')
                elif has_speech:
                    # Silence after speech
                    if silence_start is None:
                        silence_start = time.time()
                        log(f'Silence started (level: {current_level:.4f})')
                    else:
                        silence_duration = time.time() - silence_start
                        if silence_duration >= vad_silence_duration:
                            dur = time.time() - t0
                            log(f'Recording stopped after {silence_duration:.2f}s of silence (total: {dur:.2f}s)')
                            break

            # Check timeout when piped
            if timeout and (time.time() - t0) >= timeout:
                log(f'Recording stopped after timeout ({timeout}s)')
                break

            # Check keyboard input (only if not in signal mode)
            if not signal_mode:
                try:
                    if q.get_nowait():
                        dur = time.time() - t0
                        log(f'Recording stopped after {dur:.2f}s')
                        break
                except queue.Empty:
                    pass

            time.sleep(0.01)  # Reduced from 0.05 for faster signal detection
    except Exception as e:
        log(f'Error during recording: {e}')
        print(f'\n{e}', file=sys.stderr)
        write_status({
            'status': 'error',
            'error_message': str(e),
            'pid': os.getpid(),
            'timestamp': int(time.time())
        })
        return None
    finally:
        # Explicitly stop and close stream
        stream.stop()
        stream.close()
        log('Audio stream stopped and closed')

    for _ in range(3):
        draw(1.0, fullwidth=codevoice_mode)
        time.sleep(0.15)
        draw(0.0, fullwidth=codevoice_mode)
        time.sleep(0.15)

    start_proc()

    if rec:
        frames = len(rec)
        log(f'Concatenating {frames} audio frames')
        data = np.concatenate(rec)
        if data.ndim > 1:
            data = data.flatten()
        return data
    else:
        log('No audio recorded')
        return None


def transcribe(path, model, lang, run=None, blink_state=None):
    global pct, preloaded_model

    # Write processing status
    write_status({
        'status': 'processing',
        'audio_level': 0.0,
        'pid': os.getpid(),
        'language': lang,
        'model': model,
        'timestamp': int(time.time()),
        'progress': 0.0,
        'duration': 0.0,
        'mode': 'processing',
        'transcription': None
    })

    try:
        # Use faster-whisper if fast_mode enabled and available
        if fast_mode and FASTER_WHISPER_AVAILABLE:
            log(f'Using faster-whisper with model: {model}')

            # Use preloaded model if available, otherwise load now
            if preloaded_model is not None:
                m = preloaded_model
                log('Using preloaded model')
            else:
                t0 = time.time()
                m = WhisperModel(model, device="cpu", compute_type="int8")
                log(f'Model loaded in {time.time()-t0:.2f}s')

            if blink_state:
                while (blink_state[0] // 6) % 2 != 0:
                    time.sleep(0.01)
                pct[0] = 0.2

            log(f'Starting transcription (language={lang})')
            t0 = time.time()

            # faster-whisper returns (segments, info)
            segments, info = m.transcribe(path, language=lang, beam_size=5)

            # Collect segments into text
            text = ""
            for segment in segments:
                text += segment.text
                if blink_state:
                    pct[0] = min(0.2 + (segment.end / 30.0) * 0.8, 1.0)
                log(f'Segment: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}')

            log(f'Transcription completed in {time.time()-t0:.2f}s')
            log(f'Detected language: {info.language}')
            log(f'Text length: {len(text)} chars')

            if blink_state:
                pct[0] = 1.0
                time.sleep(0.1)

            # Return dict compatible with openai-whisper format
            return {'text': text, 'language': info.language}

        else:
            # Use standard openai-whisper
            log(f'Loading Whisper model: {model}')

            # Use preloaded model if available
            if preloaded_model is not None:
                m = preloaded_model
                log('Using preloaded model')
            else:
                t0 = time.time()
                m = whisper.load_model(model)
                log(f'Model loaded in {time.time()-t0:.2f}s')

            if blink_state:
                while (blink_state[0] // 6) % 2 != 0:
                    time.sleep(0.01)
                pct[0] = 0.2

            class P:
                def write(self, txt):
                    if verbose and txt.strip():
                        print(f'[WHISPER] {txt.strip()}', file=sys.__stderr__)
                    if '%' in txt and (x := re.search(r'(\d+)%', txt)):
                        progress = int(x.group(1)) / 100.0
                        if blink_state:
                            pct[0] = 0.2 + progress * 0.8
                        # Update status with progress
                        write_status({
                            'status': 'processing',
                            'audio_level': 0.0,
                            'pid': os.getpid(),
                            'language': lang,
                            'model': model,
                            'timestamp': int(time.time()),
                            'progress': progress,
                            'duration': 0.0,
                            'mode': 'processing',
                            'transcription': None
                        })
                def flush(self): pass

            old = sys.stderr
            sys.stderr = P()

            log(f'Starting transcription (language={lang})')
            t0 = time.time()
            r = m.transcribe(path, language=lang, fp16=False, verbose=False)
            log(f'Transcription completed in {time.time()-t0:.2f}s')
            log(f'Detected language: {r.get("language", "unknown")}')
            log(f'Text length: {len(r["text"])} chars')

            sys.stderr = old
            if blink_state:
                pct[0] = 1.0
                time.sleep(0.1)

            return r

    except Exception as e:
        log(f'Transcription error in transcribe(): {e}')
        write_status({
            'status': 'error',
            'error_message': str(e),
            'pid': os.getpid(),
            'timestamp': int(time.time())
        })
        raise
    finally:
        time.sleep(0.05)


def preload_model(model_name, lang):
    """Preload Whisper model for faster transcription"""
    global preloaded_model

    if fast_mode and FASTER_WHISPER_AVAILABLE:
        log(f'Preloading faster-whisper model: {model_name}')
        t0 = time.time()
        preloaded_model = WhisperModel(model_name, device="cpu", compute_type="int8")
        log(f'Model preloaded in {time.time()-t0:.2f}s')
    else:
        log(f'Preloading whisper model: {model_name}')
        t0 = time.time()
        preloaded_model = whisper.load_model(model_name)
        log(f'Model preloaded in {time.time()-t0:.2f}s')


def show_processing_animation(run, pct, blink_state, fullwidth):
    """Show processing animation in background thread"""
    def prog():
        while run[0]:
            # Simple pulsing animation when starting (<15%)
            if pct[0] < 0.15:
                level = 0.1 if (blink_state[0] // 6) % 2 == 0 else 0.0
            else:
                level = pct[0]
            blink_state[0] += 1
            draw(level, txt='Processing', fullwidth=fullwidth)
            time.sleep(0.05)
    threading.Thread(target=prog, daemon=True).start()
    pct[0] = 0.0


def process_file(file_path, lang, mdl, codevoice):
    """Transcribe audio from file"""
    # Validate file exists
    if not os.path.exists(file_path):
        print(f'Error: File not found: {file_path}', file=sys.stderr)
        sys.exit(1)

    # Validate file is readable
    if not os.path.isfile(file_path):
        print(f'Error: Not a file: {file_path}', file=sys.stderr)
        sys.exit(1)

    # Check file size (warn if > 100MB)
    file_size = os.path.getsize(file_path)
    if file_size > 100 * 1024 * 1024:
        print(f'Warning: Large file ({file_size / (1024*1024):.1f}MB), transcription may take a while', file=sys.stderr)

    log(f'Processing file: {file_path} ({file_size} bytes)')

    # Preload model if fast_mode enabled
    if fast_mode:
        preload_model(mdl, lang)

    # Show processing UI
    global pct
    run = [True]
    blink_state = [0]
    show_processing_animation(run, pct, blink_state, codevoice)

    try:
        r = transcribe(file_path, mdl, lang, run, blink_state)
        # Clear the UI line (unless in quiet/json mode)
        if not quiet_mode and not json_mode:
            out = sys.stderr if not is_tty else sys.stdout
            out.write('\r' + CLR)
            out.flush()
        log(f'Final transcription: "{r["text"].strip()}"')

        text = r['text'].strip()
        output_transcription(text, lang, mdl)

        # Write final status
        write_status({
            'status': 'done',
            'audio_level': 0.0,
            'pid': os.getpid(),
            'language': lang,
            'model': mdl,
            'timestamp': int(time.time()),
            'progress': 1.0,
            'duration': 0.0,
            'mode': 'done',
            'transcription': text
        })
    except Exception as e:
        log(f'Transcription error: {e}')
        if verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
    finally:
        run[0] = False


def process_recording(lang, mdl, sig_mode, codevoice):
    """Record and transcribe audio from microphone"""
    # Configure signal handler if in signal mode
    if sig_mode:
        signal.signal(signal.SIGUSR1, signal_handler)
        log('Signal mode enabled: listening for SIGUSR1')
        # Show PID in stderr even in quiet/json mode (needed for signal)
        if quiet_mode or json_mode:
            pid = os.getpid()
            print(f'PID: {pid}  # Send SIGUSR1 to stop: kill -SIGUSR1 {pid}', file=sys.stderr)

    log(f'Starting listen (language={lang}, model={mdl})')

    # Preload model if fast_mode enabled
    if fast_mode:
        preload_model(mdl, lang)

    global pct
    run = [True]
    blink_state = [0]

    def start_proc():
        show_processing_animation(run, pct, blink_state, codevoice)

    data = record(start_proc, lang, mdl)
    if data is None or len(data) == 0:
        log('No audio data to process')
        sys.exit(1)

    log(f'Audio data shape: {data.shape}, duration: {len(data)/SAMPLE_RATE:.2f}s')

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    log(f'Saving audio to temp file: {tmp.name}')
    with wave.open(tmp.name, 'wb') as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((data * 32767).astype(np.int16).tobytes())
    log(f'Audio file size: {os.path.getsize(tmp.name)} bytes')

    try:
        r = transcribe(tmp.name, mdl, lang, run, blink_state)
        # Clear the UI line on the appropriate stream (unless in quiet/json mode)
        if not quiet_mode and not json_mode:
            out = sys.stderr if not is_tty else sys.stdout
            out.write('\r' + CLR)
            out.flush()
        log(f'Final transcription: "{r["text"].strip()}"')

        text = r['text'].strip()
        output_transcription(text, lang, mdl)

        # Write final status
        write_status({
            'status': 'done',
            'audio_level': 0.0,
            'pid': os.getpid(),
            'language': lang,
            'model': mdl,
            'timestamp': int(time.time()),
            'progress': 1.0,
            'duration': 0.0,
            'mode': 'done',
            'transcription': text
        })
    except Exception as e:
        log(f'Transcription error: {e}')
        if verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
    finally:
        run[0] = False
        try:
            os.unlink(tmp.name)
            log(f'Deleted temp file: {tmp.name}')
        except:
            pass


def main():
    global verbose, signal_mode, vad_enabled, vad_silence_duration, codevoice_mode

    # Handle version
    if len(sys.argv) > 1 and sys.argv[1] == '--version':
        print(f"listen {__version__}")
        return

    # Handle help
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(HELP.format(lang=config.LANGUAGE, model=config.MODEL))
        return

    # Parse CLI arguments with argparse
    global quiet_mode, json_mode, output_file, status_file, fast_mode
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-l', '--language', default=config.LANGUAGE)
    parser.add_argument('-m', '--model', default=config.MODEL)
    parser.add_argument('-f', '--file', dest='file_path')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-j', '--json', action='store_true', dest='json_output')
    parser.add_argument('-o', '--output')
    parser.add_argument('--status-file', dest='status_file')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--signal-mode', action='store_true')
    parser.add_argument('--vad', type=float, metavar='SECONDS')
    parser.add_argument('--codevoice', action='store_true')
    parser.add_argument('--fast-mode', action='store_true')

    args = parser.parse_args()

    # Apply parsed arguments
    lang = args.language
    mdl = args.model
    file_path = args.file_path
    quiet_mode = args.quiet
    json_mode = args.json_output
    output_file = args.output
    status_file = args.status_file
    verbose = args.verbose if args.verbose else config.SHOW_VERBOSE
    signal_mode = args.signal_mode
    vad_enabled = args.vad is not None
    vad_silence_duration = args.vad if args.vad else config.VAD_DEFAULT_DURATION
    codevoice_mode = args.codevoice
    fast_mode = args.fast_mode

    # Check if fast_mode is enabled but faster-whisper is not available
    if fast_mode and not FASTER_WHISPER_AVAILABLE:
        print('Error: --fast-mode requires faster-whisper. Install with: pip install faster-whisper', file=sys.stderr)
        sys.exit(1)

    # Route to appropriate mode
    if file_path:
        process_file(file_path, lang, mdl, codevoice_mode)
    else:
        process_recording(lang, mdl, signal_mode, codevoice_mode)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
