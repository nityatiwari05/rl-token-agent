"""
TokenBudgetEnv: a gymnasium.Env where the agent controls retrieval
breadth, LLM tier, query refinement, and when to stop -- under a
hard token/step budget. See env/actions.py, env/state.py, env/reward.py.
"""
import random
from typing import List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.actions import Action, N_ACTIONS, RETRIEVE_K
from env.reward import terminal_reward
from env.state import STATE_DIM, build_state
from llm.base import LLMProvider
from world.generator import World
from world.retriever import RankedDoc, build_prompt, rank_documents


class TokenBudgetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        worlds: List[World],
        llm: LLMProvider,
        max_steps: int = 8,
        token_budget: int = 1500,
        max_docs: int = 12,
        lambda_token: float = 0.001,
        alpha_step: float = 0.02,
        refine_cost_tokens: int = 8,
        forced_stop_penalty: float = 0.05,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.worlds = worlds
        self.llm = llm
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.max_docs = max_docs
        self.lambda_token = lambda_token
        self.alpha_step = alpha_step
        self.refine_cost_tokens = refine_cost_tokens
        self.forced_stop_penalty = forced_stop_penalty
        self.rng = random.Random(seed)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)

        # Episode-local state
        self._world: Optional[World] = None
        self._ranked: List[RankedDoc] = []
        self._retrieved: List[RankedDoc] = []
        self._tokens_used = 0
        self._steps_used = 0
        self._has_called_llm = False
        self._last_agreement = 0.0
        self._last_tier = 0.0
        self._refine_applied = False
        self._final_answer: Optional[str] = None

    # ---------- gym API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._world = self.rng.choice(self.worlds)
        self._ranked = rank_documents(self._world)
        self._retrieved = []
        self._tokens_used = 0
        self._steps_used = 0
        self._has_called_llm = False
        self._last_agreement = 0.0
        self._last_tier = 0.0
        self._refine_applied = False
        self._final_answer = None
        return self._get_obs(), {}

    def step(self, action: int):
        action = Action(action)
        self._steps_used += 1
        terminated = False
        reward = 0.0
        info = {}

        if action in RETRIEVE_K:
            k = RETRIEVE_K[action]
            self._retrieved = self._ranked[:k]  # re-retrieve (not additive, simpler semantics)

        elif action == Action.REFINE_QUERY:
            self._refine_applied = True
            self._tokens_used += self.refine_cost_tokens
            # Simulate query rewriting: small boost to true-relevant doc's
            # effective rank by re-sorting with added signal noise reduced.
            self._ranked = sorted(
                self._ranked,
                key=lambda rd: rd.score + (0.1 if rd.doc.is_relevant else 0.0),
                reverse=True,
            )

        elif action in (Action.CALL_CHEAP, Action.CALL_EXPENSIVE):
            n_samples = 1 if action == Action.CALL_CHEAP else 3
            self._last_tier = 0.5 if action == Action.CALL_CHEAP else 1.0
            prompt = build_prompt(self._world.question, self._retrieved)
            response = self.llm.generate(prompt, n_samples=n_samples, max_tokens=16, temperature=0.0)
            self._tokens_used += response.total_tokens
            self._has_called_llm = True
            self._last_agreement = response.agreement
            self._final_answer = response.text

        elif action == Action.STOP:
            terminated = True
            is_correct = (self._final_answer is not None) and (
                self._final_answer.strip().upper() == self._world.answer.strip().upper()
            )
            reward = terminal_reward(
                is_correct=is_correct,
                tokens_used=self._tokens_used,
                steps_used=self._steps_used,
                lambda_token=self.lambda_token,
                alpha_step=self.alpha_step,
                explicit_stop=True,
                forced_stop_penalty=self.forced_stop_penalty,
            )
            info["is_correct"] = is_correct
            info["tokens_used"] = self._tokens_used
            info["steps_used"] = self._steps_used
            info["final_answer"] = self._final_answer
            info["true_answer"] = self._world.answer

        truncated = False
        if not terminated and (self._steps_used >= self.max_steps or self._tokens_used >= self.token_budget):
            # Forced stop -- budget/step exhausted without explicit STOP.
            truncated = True
            is_correct = (self._final_answer is not None) and (
                self._final_answer.strip().upper() == self._world.answer.strip().upper()
            )
            reward = terminal_reward(
                is_correct=is_correct,
                tokens_used=self._tokens_used,
                steps_used=self._steps_used,
                lambda_token=self.lambda_token,
                alpha_step=self.alpha_step,
                explicit_stop=False,
                forced_stop_penalty=self.forced_stop_penalty,
            )
            info["is_correct"] = is_correct
            info["tokens_used"] = self._tokens_used
            info["steps_used"] = self._steps_used
            info["final_answer"] = self._final_answer
            info["true_answer"] = self._world.answer
            info["forced_stop"] = True

        return self._get_obs(), reward, terminated, truncated, info

    # ---------- helpers ----------

    def _get_obs(self) -> np.ndarray:
        if self._retrieved:
            scores = [rd.score for rd in self._retrieved]
            best_score = max(scores)
            mean_score = sum(scores) / len(scores)
        else:
            best_score = 0.0
            mean_score = 0.0
        return build_state(
            n_docs_retrieved=len(self._retrieved),
            max_docs=self.max_docs,
            best_score=best_score,
            mean_score=mean_score,
            has_called_llm=self._has_called_llm,
            last_agreement=self._last_agreement,
            last_tier=self._last_tier,
            tokens_used=self._tokens_used,
            token_budget=self.token_budget,
            steps_used=self._steps_used,
            max_steps=self.max_steps,
            refine_applied=self._refine_applied,
        )