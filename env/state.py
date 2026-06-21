"""
Vectorized observation. Never pass raw text to RL -- only numeric
summaries of context quality, cost spent, and budget remaining.
"""
import numpy as np


# Feature order (fixed, must match STATE_DIM):
# 0: n_docs_retrieved_so_far (normalized by max_docs)
# 1: best_retrieved_score (top doc relevance score, 0-1)
# 2: mean_retrieved_score (avg relevance score of retrieved docs, 0-1)
# 3: has_called_llm (0/1)
# 4: last_call_agreement (0-1, majority vote agreement; 0 if no call yet)
# 5: last_call_tier (0 = none, 0.5 = cheap, 1.0 = expensive)
# 6: tokens_used_frac (tokens used / budget)
# 7: steps_used_frac (steps used / max_steps)
# 8: refine_applied (0/1)

STATE_DIM = 9


def build_state(
    n_docs_retrieved: int,
    max_docs: int,
    best_score: float,
    mean_score: float,
    has_called_llm: bool,
    last_agreement: float,
    last_tier: float,
    tokens_used: int,
    token_budget: int,
    steps_used: int,
    max_steps: int,
    refine_applied: bool,
) -> np.ndarray:
    return np.array([
        min(1.0, n_docs_retrieved / max(1, max_docs)),
        best_score,
        mean_score,
        1.0 if has_called_llm else 0.0,
        last_agreement,
        last_tier,
        min(1.0, tokens_used / max(1, token_budget)),
        min(1.0, steps_used / max(1, max_steps)),
        1.0 if refine_applied else 0.0,
    ], dtype=np.float32)