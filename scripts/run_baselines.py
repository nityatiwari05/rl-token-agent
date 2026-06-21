import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.generator import generate_dataset
from llm.mock_provider import MockProvider
from env.pomdp_env import TokenBudgetEnv
from agents.baselines import (
    AlwaysCheapPolicy, AlwaysExpensivePolicy, FixedPipelinePolicy,
    HeuristicThresholdPolicy, evaluate_policy,
)


def main():
    worlds = generate_dataset(n_worlds=300, seed=42, n_distractors_range=(3, 14))
    llm = MockProvider(base_error_rate=0.2, seed=42)
    env = TokenBudgetEnv(worlds=worlds, llm=llm, max_steps=6, token_budget=1500, seed=42)

    policies = {
        "always_cheap (k=5)": AlwaysCheapPolicy(k_retrieve=5),
        "always_expensive (k=5)": AlwaysExpensivePolicy(k_retrieve=5),
        "fixed_pipeline (k=3, cheap)": FixedPipelinePolicy(),
        "heuristic_threshold": HeuristicThresholdPolicy(),
    }

    n_episodes = 200
    print(f"{'Policy':<30} {'Acc':>6} {'AvgTok':>8} {'AvgStep':>8} {'AvgRew':>8} {'Tok/Correct':>12} {'Acc/1kTok':>10}")
    print("-" * 90)
    results = {}
    for name, policy in policies.items():
        res = evaluate_policy(env, policy, n_episodes=n_episodes)
        results[name] = res
        print(f"{name:<30} {res['accuracy']:>6.2f} {res['avg_tokens']:>8.1f} {res['avg_steps']:>8.2f} "
              f"{res['avg_reward']:>8.3f} {res['tokens_per_correct']:>12.1f} {res['accuracy_per_1k_tokens']:>10.3f}")

    return results


if __name__ == "__main__":
    main()