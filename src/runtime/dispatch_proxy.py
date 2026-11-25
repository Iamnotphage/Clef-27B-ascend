#!/usr/bin/env python3
"""Single-entry dispatch proxy in front of the two TP1 instances.

Least-inflight dispatch over ports 8000/8001, listening on 8002.
Per-instance request counters and timings are dumped periodically and at exit.
"""
import argparse
import json
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--listen-port", type=int, default=8002)
parser.add_argument("--targets", default="127.0.0.1:8000,127.0.0.1:8001")
parser.add_argument("--stats-file", required=True)
parser.add_argument("--timeout-seconds", type=float, default=300)
args = parser.parse_args()

TARGETS = [("http://" + t).rstrip("/") for t in args.targets.split(",")]
lock = threading.Lock()
stats = {
    t: {"dispatched": 0, "inflight": 0, "peak_inflight": 0, "errors": 0,
        "latency_ms_sum": 0.0, "latency_ms_max": 0.0} for t in TARGETS
}
rr_index = 0


def pick_target():
    global rr_index
    with lock:
        best = min(TARGETS, key=lambda t: (stats[t]["inflight"], TARGETS.index(t) == rr_index))
        # round-robin tie-break among equally loaded targets
        candidates = [t for t in TARGETS if stats[t]["inflight"] == stats[best]["inflight"]]
        if len(candidates) > 1:
            rr_index = (rr_index + 1) % len(candidates)
            best = candidates[rr_index]
        stats[best]["inflight"] += 1
        stats[best]["dispatched"] += 1
        stats[best]["peak_inflight"] = max(stats[best]["peak_inflight"], stats[best]["inflight"])
        return best


def release_target(target, latency_ms, error=False):
    with lock:
        s = stats[target]
        s["inflight"] -= 1
        if error:
            s["errors"] += 1
        else:
            s["latency_ms_sum"] += latency_ms
            s["latency_ms_max"] = max(s["latency_ms_max"], latency_ms)


def dump_stats(final=False):
    with lock:
        snapshot = {t: dict(v) for t, v in stats.items()}
    snapshot["_generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    snapshot["_final"] = final
    with open(args.stats_file, "w") as f:
        json.dump(snapshot, f, indent=1)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *log_args):
        pass

    def _forward(self, body=None):
        target = pick_target()
        started = time.monotonic()
        url = target + self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        try:
            req = Request(url, data=body, headers=headers, method=self.command)
            with urlopen(req, timeout=args.timeout_seconds) as resp:
                payload = resp.read()
                latency = (time.monotonic() - started) * 1000
                release_target(target, latency)
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Dispatched-To", target)
                self.end_headers()
                self.wfile.write(payload)
        except Exception as exc:
            latency = (time.monotonic() - started) * 1000
            release_target(target, latency, error=True)
            try:
                message = json.dumps({"dispatch_error": str(exc), "target": target}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            except Exception:
                pass

    def do_GET(self):
        if self.path == "/health":
            statuses = {}
            for t in TARGETS:
                try:
                    with urlopen(t + "/health", timeout=5) as resp:
                        statuses[t] = resp.status
                except Exception as exc:
                    statuses[t] = f"error: {exc}"
            ok = all(s == 200 for s in statuses.values())
            payload = json.dumps({
                "status": "ready" if ok else "degraded",
                "mode": "dual-tp1-dispatch",
                "targets": statuses,
            }).encode()
            self.send_response(200 if ok else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._forward()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._forward(body)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


def main():
    server = Server(("127.0.0.1", args.listen_port), Handler)

    def shutdown(signum, frame):
        dump_stats(final=True)
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    def dumper():
        while True:
            time.sleep(10)
            dump_stats()

    threading.Thread(target=dumper, daemon=True).start()
    server.serve_forever()
    dump_stats(final=True)


if __name__ == "__main__":
    main()
