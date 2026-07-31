"""A fake Ollama server for demos and manual testing.

    python tools/mock_ollama.py 11434 11435

Streams NDJSON word by word so the UI behaves exactly as it would against a
real server, without needing a GPU.
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ["gemma2:9b", "llama3:8b", "qwen2.5:7b", "phi3:mini"]

REPLIES = [
    "I think the key consideration is maintainability. A simpler design that "
    "the team understands beats a clever one that nobody can change safely.",
    "Speed matters, but correctness matters more. I would measure first, then "
    "optimise only what the numbers say is slow.",
    "Both positions have merit. The honest answer is that it depends on the "
    "size of the team and how long the code has to live.",
    "Let me push back on that slightly: the trade-off is real, but the cost of "
    "getting it wrong is much higher on the reliability side.",
]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": name} for name in MODELS]})
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        model = request.get("model", "unknown")

        reply = random.choice(REPLIES)
        words = reply.split(" ")

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            for word in words:
                self._chunk(json.dumps({
                    "model": model,
                    "message": {"role": "assistant", "content": word + " "},
                    "done": False,
                }))
                time.sleep(0.04)
            self._chunk(json.dumps({
                "model": model,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "eval_count": len(words),
                "prompt_eval_count": 24,
                "eval_duration": int(len(words) * 0.04 * 1e9),
            }))
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass       # the client cancelled, which is the point of the test

    def _chunk(self, text: str) -> None:
        payload = (text + "\n").encode()
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()


def serve(port: int) -> None:
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    ports = [int(arg) for arg in sys.argv[1:]] or [11434]
    for port in ports[:-1]:
        threading.Thread(target=serve, args=(port,), daemon=True).start()
    serve(ports[-1])
