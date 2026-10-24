#!/usr/bin/env python3
"""OpenAI-compatible stand-in for llama-server / vLLM, for bench tests without a GPU.

    python tools/mock_vlm_server.py --port 8080 --latency 1.2 --script "SAFE: driver attentive." \
        --script "UNSAFE: the driver is holding a phone while driving."

Answers cycle through --script lines. Serves GET /v1/models and POST /v1/chat/completions,
validates that the request carries a base64 JPEG, and records its size.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(args):
    answers = itertools.cycle(args.script or ["SAFE: no unsafe behavior visible."])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # quiet
            pass

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/v1/models"):
                self._json(200, {"object": "list", "data": [{"id": args.model, "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self.path.startswith("/v1/chat/completions"):
                self._json(404, {"error": "not found"})
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            content = body["messages"][0]["content"]
            image = next(c for c in content if c["type"] == "image_url")["image_url"]["url"]
            jpeg = base64.b64decode(image.split(",", 1)[1])
            assert jpeg[:2] == b"\xff\xd8", "expected JPEG"
            time.sleep(args.latency)
            text = next(answers)
            self._json(200, {"id": "mock", "object": "chat.completion", "model": args.model,
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": text}}],
                             "usage": {"prompt_tokens": 0, "completion_tokens": len(text.split()),
                                       "image_bytes": len(jpeg)}})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--latency", type=float, default=1.0)
    ap.add_argument("--model", default="mock-vlm-1.5b-q4")
    ap.add_argument("--script", action="append")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args))
    print(f"mock VLM on http://{args.host}:{args.port}/v1/chat/completions", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
