"""
Minimal REINFORCE (vanilla policy gradient) agent. Purpose: cheaply
validate that the environment/reward actually contain learnable signal
before spending time tuning PPO. Small MLP policy over the 9-dim state.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.actions import N_ACTIONS


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # logits


class ReinforceAgent:
    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS, lr: float = 3e-3, gamma: float = 0.99):
        self.policy = PolicyNet(state_dim, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, obs: np.ndarray, greedy: bool = False):
        state = torch.from_numpy(obs).float().unsqueeze(0)
        logits = self.policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), log_prob, entropy

    def run_episode(self, env):
        obs, _ = env.reset()
        log_probs = []
        entropies = []
        rewards = []
        term, trunc = False, False
        info = {}
        while not term and not trunc:
            action, log_prob, entropy = self.select_action(obs)
            obs, reward, term, trunc, info = env.step(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
        episode_return = sum(rewards)  # only terminal reward is nonzero here
        return log_probs, entropies, episode_return, info

    def train_batch(self, env, batch_size: int = 16, entropy_coef: float = 0.02) -> dict:
        """
        Collect `batch_size` full episodes, then do ONE gradient update using
        (return - batch_mean_return) as the advantage. This batch-mean baseline
        is what fixes the single-episode variance collapse: a single episode's
        whole trajectory shares one scalar return, so per-episode normalization
        is degenerate (std=0) -- you need the baseline computed ACROSS episodes.
        """
        batch_log_probs = []
        batch_entropies = []
        batch_returns = []
        infos = []

        for _ in range(batch_size):
            log_probs, entropies, ep_return, info = self.run_episode(env)
            # repeat the scalar return for every step in this episode so we
            # can flatten everything into one big loss tensor
            batch_returns.extend([ep_return] * len(log_probs))
            batch_log_probs.extend(log_probs)
            batch_entropies.extend(entropies)
            infos.append(info)

        returns_t = torch.tensor(batch_returns, dtype=torch.float32)
        baseline = returns_t.mean()
        advantage = returns_t - baseline
        std = advantage.std()
        if std > 1e-6:
            advantage = advantage / (std + 1e-8)

        log_probs_t = torch.stack(batch_log_probs)
        entropies_t = torch.stack(batch_entropies)

        pg_loss = -(log_probs_t * advantage.detach()).mean()
        entropy_loss = -entropy_coef * entropies_t.mean()
        loss = pg_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        accuracy = sum(i.get("is_correct", False) for i in infos) / len(infos)
        avg_tokens = sum(i.get("tokens_used", 0) for i in infos) / len(infos)
        return {
            "batch_accuracy": accuracy,
            "batch_avg_tokens": avg_tokens,
            "batch_avg_return": returns_t.mean().item(),
        }

    def evaluate(self, env, n_episodes: int) -> dict:
        results = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            term, trunc = False, False
            info = {}
            total_reward = 0.0
            while not term and not trunc:
                action, _, _ = self.select_action(obs, greedy=True)
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
            info["episode_reward"] = total_reward
            results.append(info)
        n = len(results)
        accuracy = sum(r.get("is_correct", False) for r in results) / n
        avg_tokens = sum(r.get("tokens_used", 0) for r in results) / n
        avg_steps = sum(r.get("steps_used", 0) for r in results) / n
        avg_reward = sum(r.get("episode_reward", 0.0) for r in results) / n
        n_correct = sum(r.get("is_correct", False) for r in results)
        tokens_per_correct = (sum(r.get("tokens_used", 0) for r in results) / n_correct) if n_correct > 0 else float("inf")
        acc_per_1k = (accuracy / (avg_tokens / 1000)) if avg_tokens > 0 else 0.0
        return {
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "tokens_per_correct": tokens_per_correct,
            "accuracy_per_1k_tokens": acc_per_1k,
            "n_episodes": n,
        }