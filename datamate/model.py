"""Bounded local inference client. No remote endpoint, redirects or runtime downloads."""
from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelReply:
    text: str
    tokens: int


class BackendError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BackendError("local model redirects are disabled")


class LocalModel:
    def __init__(self, url: str, model: str, revision: str, parameter_count: int):
        parsed = urllib.parse.urlsplit(url)
        try:
            local = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            local = False
        if parsed.scheme != "http" or not local or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("model URL must use a literal loopback IP over HTTP")
        if not revision or not 0 < parameter_count <= 15_000_000_000:
            raise ValueError("pinned revision and <=15B declared inference parameters required")
        self.url = url.rstrip("/") + "/chat/completions"
        self.model, self.revision, self.parameter_count = model, revision, parameter_count
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def generate(self, messages: list[dict], max_tokens: int, deadline: float) -> ModelReply:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or max_tokens < 1:
            raise BackendError("inference budget exhausted")
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=remaining) as response:
                payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                raise BackendError("model response exceeds byte budget")
            result = json.loads(payload)
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise BackendError("model output truncated or incomplete")
            used = result.get("usage", {}).get("total_tokens")
            if not isinstance(used, int) or isinstance(used, bool) or used < 1:
                raise BackendError("model response lacks token accounting")
            text = choice["message"]["content"]
            if not isinstance(text, str):
                raise BackendError("model content is not text")
            return ModelReply(text, used)
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise BackendError(f"local inference failed: {type(exc).__name__}") from exc


class ScriptedModel:
    """CPU-only planner tests, explicitly not an inference benchmark."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def generate(self, messages, max_tokens, deadline):
        self.calls.append(messages)
        if not self.replies:
            raise BackendError("scripted replies exhausted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ModelReply(json.dumps(reply) if isinstance(reply, dict) else reply, 100)
