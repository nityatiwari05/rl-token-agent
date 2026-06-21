"""
Mock LLM provider for testing env/agent logic without a real model.

Behavior designed to mimic a real model's weaknesses so RL has
something real to learn:
- Answer quality depends on whether the correct supporting doc
  is actually present in the prompt (it looks for a special
  marker injected by the world generator: "ANSWER_MARKER:<value>")
- More samples (n_samples) + more relevant context -> higher
  probability of recovering the correct marker.
- Cheap tier (n_samples=1) is noisier than expensive tier (n_samples=3).
"""
import random
import re
from typing import List

from llm.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    def __init__(self, base_error_rate: float = 0.35, seed: int = 0):
        self.base_error_rate = base_error_rate
        self.rng = random.Random(seed)

    def count_tokens(self, text: str) -> int:
        return max(1, len(re.findall(r"\S+", text)))

    def _sample_once(self, prompt: str) -> str:
        # Look for the ground-truth marker the world generator embeds
        # in the relevant document, if it made it into the prompt.
        match = re.search(r"ANSWER_MARKER:([A-Za-z0-9]+)", prompt)
        if match is None:
            # No relevant doc in context at all -> essentially guessing
            return "UNKNOWN"
        true_answer = match.group(1)

        # More irrelevant docs in the prompt -> higher error rate.
        # This is what makes "retrieve less but better" actually matter
        # for the mock model, not just for token count.
        n_docs = len(re.findall(r"\[DOC\d+\]", prompt))
        noise_penalty = min(0.4, 0.04 * max(0, n_docs - 1))
        effective_error_rate = min(0.9, self.base_error_rate + noise_penalty)

        if self.rng.random() < effective_error_rate:
            # Simulate a wrong answer (hallucination)
            return f"WRONG_{self.rng.randint(0, 999)}"
        return true_answer

    def generate(self, prompt: str, n_samples: int = 1, max_tokens: int = 128, temperature: float = 0.0) -> LLMResponse:
        samples: List[str] = [self._sample_once(prompt) for _ in range(n_samples)]
        from collections import Counter
        counts = Counter(samples)
        majority, top_count = counts.most_common(1)[0]
        agreement = top_count / len(samples)
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = sum(self.count_tokens(s) for s in samples)
        return LLMResponse(
            text=majority,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            n_samples=n_samples,
            raw_samples=samples,
            agreement=agreement,
        )
    
