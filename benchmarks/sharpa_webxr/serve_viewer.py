"""Serve the disagreement viewer and record human review decisions.

GET  /...            static files under data/webxr_episodes/_viewer (index.html, videos/)
GET  /reviews.json   all saved reviews
POST /review         JSON {key, decision, reason, ...}; merged into data/webxr_episodes/_viewer/reviews.json

decision: "verifier" (verifier verdict is right, human label wrong), "human" (human label right, verifier wrong),
"both_wrong", "unsure", or "" to clear. The reviews file is the record used to correct labels for calibration.

usage: python benchmarks/sharpa_webxr/serve_viewer.py [--port 8766]
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIEWER = ROOT / "data/webxr_episodes/_viewer"
REVIEWS = VIEWER / "reviews.json"
LOCK = threading.Lock()


def load_reviews() -> dict:
    try:
        return json.loads(REVIEWS.read_text())
    except Exception:
        return {}


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter log
        if "/review" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def do_GET(self):
        if self.path.split("?")[0] == "/reviews.json":
            body = json.dumps(load_reviews()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/review":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            rec = json.loads(self.rfile.read(n))
            key = rec["key"]
        except Exception as e:  # noqa: BLE001
            self.send_error(400, f"bad json: {e}")
            return
        rec["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with LOCK:
            reviews = load_reviews()
            if rec.get("decision"):
                reviews[key] = rec
            else:
                reviews.pop(key, None)
            tmp = REVIEWS.with_suffix(".tmp")
            tmp.write_text(json.dumps(reviews, indent=1, sort_keys=True))
            os.replace(tmp, REVIEWS)
        body = json.dumps({"ok": True, "n": len(reviews)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--host", default="0.0.0.0")
    a = p.parse_args()
    VIEWER.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(VIEWER)))
    print(f"serving {VIEWER} on http://{a.host}:{a.port}/  reviews -> {REVIEWS}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
