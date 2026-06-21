"""
Ollama-backed LLM provider. Implements cheap/expensive tiers via
sample count + context size, NOT via different models (per project
scoping: only ollama 3.1:8b is available).

Cheap tier:     n_samples=1, smaller context (top-k docs truncated)
Expensive tier: n_samples=3 (majority vote), larger context allowed

Requires: `pip install requests` and a running `ollama serve` with
the model pulled (`ollama pull llama3.1:8b`).
"""
import os
import re
import time
from collections import Counter
from typing import List, Optional

import requests

from llm.base import LLMProvider, LLMResponse
from llm.cache import DiskCache


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        use_cache: bool = True,
        cache_dir: str = ".llm_cache",
        request_timeout: float = 60.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.cache = DiskCache(cache_dir) if use_cache else None
        self.timeout = request_timeout

    def count_tokens(self, text: str) -> int:
        # Ollama doesn't expose a tokenizer endpoint cheaply; approximate
        # with a whitespace/punctuation heuristic (~4 chars/token is the
        # standard rough estimate for English; we use word count * 1.3
        # which tends to track llama tokenizers reasonably well).
        words = re.findall(r"\S+", text)
        return max(1, int(len(words) * 1.3))

    def _call_once(self, prompt: str, max_tokens: int, temperature: float) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def generate(
        self,
        prompt: str,
        n_samples: int = 1,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Force some temperature if we want multiple distinct samples
        eff_temp = temperature if (n_samples == 1) else max(temperature, 0.7)

        if self.cache is not None:
            cached = self.cache.get(self.model_name, prompt, n_samples, max_tokens, eff_temp)
            if cached is not None:
                return cached

        samples: List[str] = []
        for _ in range(n_samples):
            samples.append(self._call_once(prompt, max_tokens, eff_temp))

        majority_text, agreement = self._majority_vote(samples)
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = sum(self.count_tokens(s) for s in samples)

        response = LLMResponse(
            text=majority_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            n_samples=n_samples,
            raw_samples=samples,
            agreement=agreement,
        )

        if self.cache is not None:
            self.cache.set(self.model_name, prompt, n_samples, max_tokens, eff_temp, response)

        return response

    @staticmethod
    def _majority_vote(samples: List[str]) -> (str, float):
        if len(samples) == 1:
            return samples[0], 1.0
        # Normalize for voting purposes (strip whitespace/case) but
        # return the original-cased majority sample's text.
        norm_to_orig = {}
        normed = []
        for s in samples:
            n = s.strip().lower()
            normed.append(n)
            norm_to_orig.setdefault(n, s)
        counts = Counter(normed)
        top_norm, top_count = counts.most_common(1)[0]
        agreement = top_count / len(samples)
        return norm_to_orig[top_norm], agreement