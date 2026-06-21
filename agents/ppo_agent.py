"""
PPO training via stable-baselines3, on the same TokenBudgetEnv.
This is the env-stability upgrade over REINFORCE: clipped surrogate
objective + GAE should handle the reward variance much better.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def make_ppo(env, **ppo_kwargs):
    vec_env = DummyVecEnv([lambda: env])
    default_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=256,        # rollout length before each update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,       # entropy bonus -- same collapse risk as REINFORCE, so keep this on
        verbose=0,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    default_kwargs.update(ppo_kwargs)
    return PPO(**default_kwargs)


def evaluate_ppo(model, env, n_episodes: int) -> dict:
    results = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        term, trunc = False, False
        info = {}
        total_reward = 0.0
        while not term and not trunc:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
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