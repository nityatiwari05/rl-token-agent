"""
Disk-backed cache, keyed by hash(prompt + sampling params + model name).
Critical for RL training: the same (state, action) pair will be visited
thousands of times across episodes, and we don't want to re-call the LLM
every time during development/debugging.
"""
import hashlib
import json
import os
from typing import Optional

from llm.base import LLMResponse


class DiskCache:
    def __init__(self, cache_dir: str = ".llm_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, model_name: str, prompt: str, n_samples: int, max_tokens: int, temperature: float) -> str:
        raw = f"{model_name}|{n_samples}|{max_tokens}|{temperature}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, model_name: str, prompt: str, n_samples: int, max_tokens: int, temperature: float) -> Optional[LLMResponse]:
        key = self._key(model_name, prompt, n_samples, max_tokens, temperature)
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return LLMResponse(**data)

    def set(self, model_name: str, prompt: str, n_samples: int, max_tokens: int, temperature: float, response: LLMResponse):
        key = self._key(model_name, prompt, n_samples, max_tokens, temperature)
        path = self._path(key)
        with open(path, "w") as f:
            json.dump(response.__dict__, f)