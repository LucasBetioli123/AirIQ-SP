"""LLM client configured through environment variables (always with a free option).

  LLM_PROVIDER=llamacpp       (default) local GGUF via llama.cpp, CPU, zero cost
  LLM_PROVIDER=ollama         local Ollama server (OpenAI-compatible API)
  LLM_PROVIDER=openai_compat  any OpenAI-compatible API (Groq, Gemini, OpenRouter...)
  LLM_MODEL=...               GGUF path (llamacpp) or model name
  LLM_BASE_URL / LLM_API_KEY  for ollama/openai_compat

For robustness with small models, every call requests schema-validated JSON output. In
llama.cpp this becomes a grammar (constrained decoding); with OpenAI-compatible providers,
`response_format`.
"""
from __future__ import annotations

import functools
import json
import os
import re
import time
from dataclasses import dataclass

from src.common.config import ROOT

DEFAULT_GGUF = "models/llm/qwen3-4b-instruct-2507-q4_k_m.gguf"
# estimated price (US$ per 1M tokens: input, output) for the cost in the log. Local = 0.
PRICES = {"llamacpp": (0.0, 0.0), "ollama": (0.0, 0.0),
          "llama-3.1-8b-instant": (0.05, 0.08), "gemini-2.0-flash": (0.10, 0.40)}


@dataclass
class LLMResult:
    text: str
    data: dict | None
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    model: str
    cost_usd: float


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None, n_ctx: int = 8192):
        self.provider = provider or os.environ.get("LLM_PROVIDER", "llamacpp")
        self.model = model or os.environ.get("LLM_MODEL") or (DEFAULT_GGUF if self.provider == "llamacpp" else "")
        if self.provider == "llamacpp":
            from llama_cpp import Llama

            path = self.model if os.path.isabs(self.model) else str(ROOT / self.model)
            self.llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=os.cpu_count(), n_batch=512,
                             verbose=False, seed=42)
            self.name = os.path.basename(path)
        else:
            from openai import OpenAI

            base = os.environ.get("LLM_BASE_URL") or ("http://localhost:11434/v1" if self.provider == "ollama" else None)
            self.client = OpenAI(base_url=base, api_key=os.environ.get("LLM_API_KEY", "ollama"))
            self.name = self.model

    def _price(self, pt: int, ct: int) -> float:
        pin, pout = PRICES.get(self.provider if self.provider != "openai_compat" else self.model, (0.0, 0.0))
        return (pt * pin + ct * pout) / 1e6

    def chat_json(self, system: str, user: str, schema: dict, max_tokens: int = 700,
                  temperature: float = 0.0) -> LLMResult:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t0 = time.time()
        if self.provider == "llamacpp":
            r = self.llm.create_chat_completion(messages=msgs, temperature=temperature, max_tokens=max_tokens,
                                                response_format={"type": "json_object", "schema": schema})
        else:
            r = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=temperature,
                                                    max_tokens=max_tokens, response_format={"type": "json_object"})
            r = r.model_dump()
        text = r["choices"][0]["message"]["content"] or ""
        usage = r.get("usage") or {}
        pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        return LLMResult(text, parse_json(text), pt, ct, time.time() - t0, self.name, self._price(pt, ct))


def parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


@functools.lru_cache(maxsize=2)
def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    return LLM(provider, model)
