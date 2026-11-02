"""Client gọi server tương thích OpenAI (vLLM) bằng thư viện chuẩn — không phụ thuộc mạng ngoài."""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request

_THINK = re.compile(r"<think>.*?</think>", re.S)


def answer_logprob(content: list | None, text: str) -> float | None:
    """Logprob trung bình của các token nằm trong khối <answer>…</answer> cuối (độ tự tin của đáp án, EHRSQL)."""
    if not content or "<answer>" not in text:
        return None
    toks = [c.get("token", "") for c in content]
    lps = [c.get("logprob") for c in content]
    joined, starts = "", []
    for t in toks:
        starts.append(len(joined))
        joined += t
    a = joined.rfind("<answer>")
    b = joined.find("</answer>", a)
    if a < 0:
        return None
    b = len(joined) if b < 0 else b
    sel = [lp for s, lp in zip(starts, lps) if a <= s < b and lp is not None]
    return round(sum(sel) / len(sel), 4) if sel else None


class LLM:
    def __init__(self, base_url: str, model: str, temperature: float = 0.6, top_p: float = 0.95,
                 max_tokens: int = 8192, enable_thinking: bool = True, timeout: float = 600,
                 logprobs: bool = False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self.logprobs = logprobs
        self._local = threading.local()  # các mẫu chạy song song: logprob lần gọi cuối theo từng luồng

    @property
    def last_logprob(self) -> float | None:
        return getattr(self._local, "lp", None)

    @property
    def last_think(self) -> str | None:
        """Khối <think> của lần gọi cuối trong luồng này (chỉ ghi log: phân tích lỗi, dữ liệu SFT có suy luận)."""
        return getattr(self._local, "think", None)

    def chat(self, messages: list[dict], seed: int | None = None) -> str:
        body = {"model": self.model, "messages": messages, **self.params,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if self.logprobs:
            body["logprobs"] = True
        self._local.lp = self._local.think = None
        if seed is not None:
            body["seed"] = seed
        data = json.dumps(body).encode()
        last = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                choice = out["choices"][0]
                text = choice["message"].get("content") or ""
                if self.logprobs:
                    self._local.lp = answer_logprob((choice.get("logprobs") or {}).get("content"), text)
                think = _THINK.search(text)
                self._local.think = think.group(0)[7:-8].strip() if think else None
                return _THINK.sub("", text).strip()
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:  # vd vượt max_model_len: thử lại vô ích
                    raise RuntimeError(f"LLM từ chối ({e.code}): {e.read()[:300]!r}") from e
                last = e
                time.sleep(2 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM lỗi sau 4 lần thử: {last}")


class ScriptedLLM:
    """LLM giả cho test: trả lần lượt các câu trả lời định sẵn (theo seed nếu là dict)."""

    def __init__(self, replies):
        self.replies = replies
        self.calls: list[list[dict]] = []

    def chat(self, messages, seed=None):
        self.calls.append(messages)
        seq = self.replies[seed] if isinstance(self.replies, dict) else self.replies
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        return seq[min(n_assistant, len(seq) - 1)]
