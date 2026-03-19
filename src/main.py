#!/usr/bin/env python3
"""Voice router CLI — hotkey mode, text mode, and session listing."""

import argparse
import asyncio
import os
import subprocess
import sys

import iterm2

from parser import parse
from router import route_command, list_sessions

PID_FILE = os.path.expanduser("~/.local/share/voice-router/listen.pid")


def transcribe_hotkey() -> str:
    """Run listen in signal+VAD mode and return transcription.

    Listen writes its own PID to a file so external tools (e.g. Hammerspoon)
    can send SIGUSR1 to stop recording on key release.
    VAD (10s silence) acts as a fallback if no signal arrives.
    """
    proc = subprocess.Popen(
        ["listen", "--signal-mode", "--vad", "10", "-m", "base"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    finally:
        # Clean up PID file
        try:
            os.unlink(PID_FILE)
        except OSError:
            pass

    if proc.returncode != 0:
        print(f"listen failed (exit {proc.returncode})", file=sys.stderr)
        if stderr:
            # Print last few lines of stderr for debugging
            lines = stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"  {line}", file=sys.stderr)
        sys.exit(1)
    return stdout.strip()


async def do_route(text: str) -> None:
    """Parse and route a text command."""
    cmd = parse(text)
    if cmd.target is None and cmd.text is None:
        print("Empty command, nothing to do")
        return

    connection = await iterm2.Connection.async_create()
    success = await route_command(connection, cmd.target, cmd.text)
    if success:
        desc = f"→ {cmd.target or 'last-active'}"
        if cmd.text:
            preview = cmd.text[:60] + ("..." if len(cmd.text) > 60 else "")
            desc += f": {preview}"
        else:
            desc += " (focus)"
        print(desc)


async def do_list() -> None:
    """List available cc sessions."""
    connection = await iterm2.Connection.async_create()
    sessions = await list_sessions(connection)
    if not sessions:
        print("No cc sessions found")
        return
    for s in sessions:
        print(f"  {s['name']:<20} {s['title']}")


def main():
    ap = argparse.ArgumentParser(description="Voice router for Claude Code sessions")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--hotkey", action="store_true", help="Record + transcribe + route")
    group.add_argument("--text", type=str, help="Route pre-transcribed text")
    group.add_argument("--list", action="store_true", help="List available cc sessions")
    args = ap.parse_args()

    if args.hotkey:
        raw = transcribe_hotkey()
        if not raw:
            print("No speech detected")
            sys.exit(0)
        print(f"Heard: {raw}")
        asyncio.run(do_route(raw))

    elif args.text:
        asyncio.run(do_route(args.text))

    elif args.list:
        asyncio.run(do_list())


if __name__ == "__main__":
    main()
