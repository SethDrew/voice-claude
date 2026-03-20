#!/usr/bin/env python3
"""Voice router CLI — hotkey mode, text mode, resolve mode, and session listing."""

import argparse
import asyncio
import os
import subprocess
import sys

import iterm2

from parser import parse
from router import route_command, find_session, list_sessions, get_last_active

PID_FILE = os.path.expanduser("~/.local/share/voice-router/listen.pid")


def transcribe_hotkey() -> str:
    """Run listen in signal+VAD mode and return transcription."""
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
        try:
            os.unlink(PID_FILE)
        except OSError:
            pass

    if proc.returncode != 0:
        print(f"listen failed (exit {proc.returncode})", file=sys.stderr)
        if stderr:
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


async def do_route_to_target(target: str, text: str) -> None:
    """Route text directly to a named target (skip parsing)."""
    connection = await iterm2.Connection.async_create()
    success = await route_command(connection, target, text)
    if success:
        print(f"→ {target}: {text[:60]}")


async def do_resolve(text: str) -> None:
    """Resolve which session a command would target, without routing.

    Prints the resolved session name to stdout. Used by Hammerspoon
    for the two-press flow (first press = resolve target, second = route).
    Always goes through the router's fuzzy matching to get the actual
    session name (e.g., "ledger" → "ledger-skill").
    """
    cmd = parse(text)
    search = cmd.target or text.strip().lower()

    connection = await iterm2.Connection.async_create()
    session = await find_session(connection, search)
    if session:
        cc_name = await session.async_get_variable("user.cc_name")
        if cc_name:
            print(cc_name)
            return

    # Fall back to last-active
    last = get_last_active()
    if last:
        print(last)
    else:
        sys.exit(1)


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
    group.add_argument("--resolve", type=str, help="Resolve target session without routing")
    group.add_argument("--list", action="store_true", help="List available cc sessions")

    ap.add_argument("--target", type=str, help="Route directly to this target (with --text)")

    args = ap.parse_args()

    if args.hotkey:
        raw = transcribe_hotkey()
        if not raw:
            print("No speech detected")
            sys.exit(0)
        print(f"Heard: {raw}")
        asyncio.run(do_route(raw))

    elif args.resolve:
        asyncio.run(do_resolve(args.resolve))

    elif args.text and args.target:
        asyncio.run(do_route_to_target(args.target, args.text))

    elif args.text:
        asyncio.run(do_route(args.text))

    elif args.list:
        asyncio.run(do_list())


if __name__ == "__main__":
    main()
