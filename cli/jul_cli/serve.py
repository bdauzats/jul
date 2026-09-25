"""A tiny local HTTP server around `TypeSafeClient`, for callers that are not Python.

    jul serve --model minicpm5-2b --port 8577

Binds to 127.0.0.1 by default (local only): the model runs on this machine and the server is a
bridge for a local process — Wispr's meeting classifier, in practice — not a public API. It can be
bound elsewhere with `--host`, in which case an API key (`--api-key` or `$JUL_API_KEY`) should be
set; requests then need a matching `x-api-key` header. Built on the standard library alone, so
`pip install jul` is enough; no extra dependency.

Privacy: the request `state` (for Wispr, private meeting speech) is never logged. Only
non-sensitive metadata (question count, latency) is logged at INFO.

The request and response shapes mirror JuL's hosted/Lambda deployment, so a client can talk to
either without change:

    POST /v1/classify            (requires x-api-key when a key is configured)
    {
      "state": {"text": "..."} | "a plain string",
      "questions": {
        "name": {"type": "choice|noul|score", "instructions": "...", "criteria": {...} | [...]}
      },
      "model": "minicpm5-2b"        # optional, overrides the server default for this call
    }

    -> {"model": ..., "latency_ms": ..., "choices": {...}, "nouls": {...}, "scores": {...}}

    GET /health   -> {"status": "ok", "model": ..., "ready": bool}   (also requires x-api-key when set)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger("jul.serve")

# One client for the whole process, loaded on the first call (or by --warmup at startup). MLX and the
# torch backends hold a single model in memory, so the client is a shared, guarded singleton.
_client = None
_client_lock = threading.Lock()
_default_model: str | None = None
_default_backend: str | None = None
#: When set, every classify request must carry a matching `x-api-key` header.
#: None means no authentication (safe only when bound to localhost).
_api_key: str | None = None


def get_client():
    """Get or build the shared client. Thread-safe: the HTTP server is threaded."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            from jul import TypeSafeClient
            logger.info("loading JuL client (model=%s, backend=%s)", _default_model, _default_backend)
            start = time.time()
            _client = TypeSafeClient(model=_default_model, backend=_default_backend)
            # A warmup call forces the weights to load now, so the first real request is not slow.
            from jul import Choice
            _client.system_one(state={"text": "warmup"},
                               questions={"q": Choice(instructions="test?",
                                                      criteria={"a": "a", "b": "b"})})
            logger.info("JuL client ready in %.1fs", time.time() - start)
    return _client


def build_questions(questions_data: dict) -> dict:
    """Build JuL question objects from the request. Same mapping as the Lambda handler."""
    from jul import Choice, Noul, Score

    questions: dict[str, Any] = {}
    for name, q in questions_data.items():
        q_type = q.get("type", "choice").lower()
        instructions = q.get("instructions", "")
        criteria = q.get("criteria", {})
        if q_type == "choice":
            questions[name] = Choice(
                instructions=instructions,
                criteria=criteria if isinstance(criteria, dict)
                else {str(i): c for i, c in enumerate(criteria)},
            )
        elif q_type == "noul":
            questions[name] = Noul(instructions=instructions)
        elif q_type == "score":
            questions[name] = Score(
                instructions=instructions,
                criteria=criteria if isinstance(criteria, list) else list(criteria.values()),
            )
        else:
            raise ValueError(f"Unknown question type: {q_type!r} for question {name!r}")
    return questions


def format_response(response, latency_ms: float) -> dict:
    """Format a `SystemOneResponse` as the API JSON. Same shape as the Lambda handler."""
    result: dict[str, Any] = {
        "model": response.model,
        "latency_ms": round(latency_ms, 2),
        "usage": {"input_tokens": response.usage.input_tokens if response.usage else None},
    }
    if response.choices:
        result["choices"] = {
            name: {"choice": c.choice,
                   "confidence": round(c.confidence, 4),
                   "probabilities": {k: round(v, 4) for k, v in c.probabilities.items()}}
            for name, c in response.choices.items()
        }
    if response.nouls:
        result["nouls"] = {name: {"noul": round(n.noul, 4)} for name, n in response.nouls.items()}
    if response.scores:
        result["scores"] = {
            name: {"score": round(s.score, 4),
                   "confidence": round(s.confidence, 4),
                   "probabilities": {k: round(v, 4) for k, v in s.probabilities.items()}}
            for name, s in response.scores.items()
        }
    return result


class Handler(BaseHTTPRequestHandler):
    # Quieter logs: one line per request through the module logger, not stderr prints.
    def log_message(self, fmt, *args):
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _authorized(self) -> bool:
        """True when no key is configured, or the request carries the right one.
        Uses a constant-time compare to avoid leaking the key via timing."""
        if _api_key is None:
            return True
        import hmac
        provided = self.headers.get("x-api-key", "")
        return hmac.compare_digest(provided, _api_key)

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            if not self._authorized():
                self._send(401, {"error": "unauthorized: missing or invalid x-api-key"})
                return
            self._send(200, {"status": "ok", "model": _default_model, "ready": _client is not None})
        else:
            self._send(404, {"error": "not found", "path": self.path})

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in ("/v1/classify", "/classify"):
            self._send(404, {"error": "not found", "path": self.path})
            return
        if not self._authorized():
            self._send(401, {"error": "unauthorized: missing or invalid x-api-key"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            # Cap the body to guard against a malformed/oversized request while
            # leaving ample room for a large state: a 32k-token context is well
            # under 1 MB even in multi-byte scripts, so 10 MB is generous headroom.
            if length > 10_000_000:
                self._send(413, {"error": "request body too large"})
                return
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"invalid JSON: {e}"})
            return

        questions_data = body.get("questions") or {}
        if not questions_data:
            self._send(400, {"error": "missing required field: questions"})
            return

        try:
            client = get_client()
            questions = build_questions(questions_data)
            state = body.get("state", {})
            model = body.get("model")  # optional per-call override
            start = time.time()
            response = client.system_one(state=state, questions=questions, model=model)
            result = format_response(response, (time.time() - start) * 1000)
            # Privacy: never log the state — for Wispr it is private meeting
            # speech. Log only non-sensitive metadata (question count, latency).
            # Enable DEBUG to see question names, still never the transcript.
            logger.info("classify: %d question(s), %.0f ms",
                        len(questions_data), (time.time() - start) * 1000)
            logger.debug("classify question names: %s", list(questions_data.keys()))
            self._send(200, result)
        except ValueError as e:
            self._send(400, {"error": str(e), "type": type(e).__name__})
        except Exception as e:  # noqa: BLE001 - surface any inference error as 500 with its type
            logger.exception("inference failed")
            self._send(500, {"error": str(e), "type": type(e).__name__})


def serve(model: str | None = None, backend: str | None = None,
          host: str = "127.0.0.1", port: int = 8577, warmup: bool = True,
          api_key: str | None = None) -> None:
    """Run the local classify server until interrupted.

    Binds to 127.0.0.1 by default (local only). When `api_key` is set, every
    request (including `/health`) must send a matching `x-api-key` header.
    """
    global _default_model, _default_backend, _api_key
    _default_model, _default_backend = model, backend
    _api_key = api_key or os.environ.get("JUL_API_KEY") or None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Security posture: warn loudly when exposing beyond loopback without a key.
    is_local = host in ("127.0.0.1", "::1", "localhost")
    if not is_local and _api_key is None:
        logger.warning("jul serve is bound to %s WITHOUT an API key — anyone on the "
                       "network can query it. Set --api-key or $JUL_API_KEY.", host)
    if _api_key is not None:
        logger.info("API key authentication is enabled (x-api-key required).")

    if warmup:
        get_client()  # load and warm the model before accepting requests

    server = ThreadingHTTPServer((host, port), Handler)
    logger.info("jul serve listening on http://%s:%d  (POST /v1/classify, GET /health)", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
