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

MODELS = ["gemma2:9b", "llama3:8b", "qwen2.5:7b", "phi3:mini", "qwen3:8b"]

# What a reasoning model sends in `message.thinking` before it answers.  It
# arrives first, in its own field, and is the whole reason the app needs to
# treat reasoning separately from the reply.
THOUGHTS = [
    "Let me work through what is actually being asked here. ",
    "There are two readings of the question and they lead to different "
    "answers, so I should pick the one that matches the context. ",
    "Checking my reasoning: the second reading is the one that fits. ",
]

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

        if self.path == "/api/pull":
            self._pull(model)
            return

        # Replies vary in length, so a low num_predict genuinely truncates some
        # of them — which is what makes the "stopped at the length cap" path
        # testable without a GPU.
        paragraphs = [random.choice(REPLIES)
                      for _ in range(random.randint(1, 12))]
        words = " ".join(paragraphs).split(" ")

        cap = (request.get("options") or {}).get("num_predict")
        truncated = isinstance(cap, int) and 0 < cap < len(words)
        if truncated:
            words = words[:cap]

        # Ollama refuses `think` on a model that cannot reason, and the app is
        # supposed to know the difference — so the mock refuses too.
        think = request.get("think")
        reasons = "qwen3" in model or "gpt-oss" in model
        if think is not None and not reasons:
            self._json({"error": f"{model} does not support thinking"}, 400)
            return
        thinking = bool(think) and reasons

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            if thinking:
                for thought in THOUGHTS:
                    for piece in thought.split(" "):
                        self._chunk(json.dumps({
                            "model": model,
                            "message": {"role": "assistant",
                                        "thinking": piece + " "},
                            "done": False,
                        }))
                        time.sleep(0.01)
            for word in words:
                self._chunk(json.dumps({
                    "model": model,
                    "message": {"role": "assistant", "content": word + " "},
                    "done": False,
                }))
                time.sleep(0.01)
            self._chunk(json.dumps({
                "model": model,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "length" if truncated else "stop",
                "eval_count": len(words),
                "prompt_eval_count": 24,
                "eval_duration": int(len(words) * 0.01 * 1e9),
            }))
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass       # the client cancelled, which is the point of the test

    def _pull(self, model: str) -> None:
        """Two layers of NDJSON progress, the way Ollama sends it."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self._chunk(json.dumps({"status": "pulling manifest"}))
            for digest, size in (("sha256:aaa", 40_000_000),
                                 ("sha256:bbb", 10_000_000)):
                done = 0
                while done < size:
                    done = min(size, done + size // 5)
                    self._chunk(json.dumps({
                        "status": f"pulling {digest[7:19]}",
                        "digest": digest, "total": size, "completed": done}))
                    time.sleep(0.05)
            self._chunk(json.dumps({"status": "verifying sha256 digest"}))
            self._chunk(json.dumps({"status": "success"}))
            self.wfile.write(b"0\r\n\r\n")
            if model not in MODELS:
                MODELS.append(model)
        except (BrokenPipeError, ConnectionResetError):
            pass

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
