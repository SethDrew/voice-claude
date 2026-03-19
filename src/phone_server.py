#!/usr/bin/env python3
"""HTTP endpoint for phone-based voice routing."""

import asyncio
import sys

from flask import Flask, jsonify, request

import iterm2

from parser import parse
from router import list_sessions, route_command

app = Flask(__name__)


def _run_async(coro):
    """Run an async iterm2 coroutine from sync Flask context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _do_route(text: str) -> dict:
    cmd = parse(text)
    if cmd.target is None and cmd.text is None:
        return {"ok": False, "error": "empty command"}

    connection = await iterm2.Connection.async_create()
    success = await route_command(connection, cmd.target, cmd.text)
    return {
        "ok": success,
        "target": cmd.target,
        "text": cmd.text,
    }


async def _do_list() -> list:
    connection = await iterm2.Connection.async_create()
    return await list_sessions(connection)


@app.route("/voice", methods=["POST"])
def voice():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"ok": False, "error": "no text provided"}), 400

    result = _run_async(_do_route(text))
    status = 200 if result["ok"] else 404
    return jsonify(result), status


@app.route("/sessions", methods=["GET"])
def sessions():
    result = _run_async(_do_list())
    return jsonify({"sessions": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def main():
    print("Starting voice router phone server on 0.0.0.0:7890")
    app.run(host="0.0.0.0", port=7890)


if __name__ == "__main__":
    main()
