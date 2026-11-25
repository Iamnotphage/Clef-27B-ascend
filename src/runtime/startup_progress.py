#!/usr/bin/env python3
"""Render vLLM model-load progress on the terminal during service startup.

``inference.py`` redirects stdout/stderr to ``<output-dir>/service.log``
before exec'ing vLLM, so the terminal would otherwise go silent during the
1-3 minute model load.  This helper is spawned just before that redirect; it
inherits the original terminal, tails the service log for the tqdm shard
progress vLLM already writes, and exits when ``/health`` is ready (or the
parent process disappears).  It never writes to the service log.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import re
import signal
import sys
import time
import urllib.request


SHARD_RE = re.compile(
    r"Loading safetensors checkpoint shards:\s+(\d+)% Completed \| (\d+)/(\d+)")
STARTED_RE = re.compile(
    r"Application startup complete|Uvicorn running on|Started server process")
CLEAR_LINE = "\r\033[K"
LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
}
RESET = "\033[0m"


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _elapsed(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def _color_enabled():
    if os.environ.get("NO_COLOR") is not None:
        return False
    override = os.environ.get("CLEF_LOG_COLOR")
    if override is not None:
        return override.strip().lower() not in ("0", "false", "no", "off")
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def _log_line(level, message):
    stamp = _stamp()
    if not _color_enabled():
        print(f"{stamp} {level:<5} clef.launcher: {message}", flush=True)
        return
    color = LEVEL_COLORS.get(level.upper(), "")
    print(
        f"\033[2m{stamp}{RESET} {color}{level:<5}{RESET} "
        f"\033[36mclef.launcher:{RESET} {message}",
        flush=True,
    )


def _pid_alive(pid):
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _health(host, port):
    try:
        with urllib.request.urlopen(
                f"http://{host}:{port}/health", timeout=1.0) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _read_new_lines(path, position, buffer):
    """Return (new position, buffer, complete lines) for an append-only log."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(position)
            chunk = handle.read()
            position = handle.tell()
    except OSError:
        return position, buffer, []
    data = buffer + chunk
    lines = data.splitlines()
    if data and data[-1] not in "\r\n":
        buffer = lines.pop() if lines else data
    else:
        buffer = ""
    return position, buffer, lines


def _render_line(elapsed, state):
    sys.stdout.write(f"{CLEAR_LINE}[startup {_elapsed(elapsed)}] {state}")
    sys.stdout.flush()


def _clear():
    sys.stdout.write(CLEAR_LINE)
    sys.stdout.flush()


def watch(log, host, port, parent_pid, timeout, force=False):
    if not force and not sys.stdout.isatty():
        return 0
    started = time.monotonic()
    position = 0
    try:
        position = os.path.getsize(log)
    except OSError:
        pass
    buffer = ""
    state = "waiting for vLLM model-load logs"
    last_state = None

    def stop(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while True:
            elapsed = time.monotonic() - started
            if not _pid_alive(parent_pid):
                _clear()
                return 0
            position, buffer, lines = _read_new_lines(log, position, buffer)
            for line in lines:
                match = SHARD_RE.search(line)
                if match:
                    percent, done, total = match.groups()
                    state = f"model load {percent}% ({done}/{total} shards)"
                elif STARTED_RE.search(line):
                    state = "vLLM API starting"
            if _health(host, port):
                _clear()
                _log_line("INFO",
                          f"Service ready at http://{host}:{port} "
                          f"(startup {_elapsed(elapsed)})")
                return 0
            if state != last_state or int(elapsed * 2) % 2 == 0:
                _render_line(elapsed, state)
                last_state = state
            if elapsed >= timeout:
                _clear()
                _log_line("WARNING",
                          f"Service not ready after {int(elapsed)}s; "
                          f"see {log}")
                return 0
            time.sleep(0.5)
    finally:
        # Leave the cursor on a fresh line if the terminal is still ours.
        try:
            _clear()
        except Exception:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--force", action="store_true",
                        help="render even when stdout is not a TTY (tests)")
    args = parser.parse_args(argv)
    return watch(args.log, args.host, args.port, args.pid, args.timeout, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
