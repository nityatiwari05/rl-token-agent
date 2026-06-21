"""
Abstract interface every LLM provider must implement.

Design goal: env/agents NEVER call Ollama or any API directly.
They only ever call LLMProvider.generate(). This is the single
swap point for moving from Ollama -> a real hosted model.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    n_samples: int          # how many samples were drawn (1 for cheap, 3 for expensive, etc.)
    raw_samples: List[str]  # all sampled completions (len == n_samples)
    agreement: float        # fraction of samples agreeing with the majority answer, in [0,1]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    """Abstract base class. Implement this for any backend (Ollama, Anthropic, OpenAI, ...)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        n_samples: int = 1,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate n_samples completions for prompt, return aggregated LLMResponse."""
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Approximate or exact token count for a string, using this model's tokenizer."""
        raise NotImplementedError