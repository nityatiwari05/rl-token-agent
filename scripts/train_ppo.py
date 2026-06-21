import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.generator import generate_dataset
from llm.mock_provider import MockProvider
from env.pomdp_env import TokenBudgetEnv
from agents.ppo_agent import make_ppo, evaluate_ppo
from agents.baselines import (
    AlwaysCheapPolicy, AlwaysExpensivePolicy, FixedPipelinePolicy,
    HeuristicThresholdPolicy, evaluate_policy,
)


def main():
    worlds = generate_dataset(n_worlds=300, seed=42, n_distractors_range=(3, 14))
    llm = MockProvider(base_error_rate=0.2, seed=42)
    env = TokenBudgetEnv(worlds=worlds, llm=llm, max_steps=6, token_budget=1500, seed=42)

    model = make_ppo(env)

    total_timesteps_per_chunk = 5000
    n_chunks = 16
    eval_env = TokenBudgetEnv(worlds=worlds, llm=llm, max_steps=6, token_budget=1500, seed=123)

    print("Training PPO...")
    for chunk in range(1, n_chunks + 1):
        model.learn(total_timesteps=total_timesteps_per_chunk, reset_num_timesteps=False)
        metrics = evaluate_ppo(model, eval_env, n_episodes=150)
        print(f"[timesteps {chunk * total_timesteps_per_chunk}] "
              f"acc={metrics['accuracy']:.2f} avg_tok={metrics['avg_tokens']:.1f} "
              f"avg_steps={metrics['avg_steps']:.2f} avg_reward={metrics['avg_reward']:.3f} "
              f"acc/1ktok={metrics['accuracy_per_1k_tokens']:.3f}")

    print("\n--- Final comparison: PPO vs all baselines (same eval env, 300 eps) ---")
    final_eval_env = TokenBudgetEnv(worlds=worlds, llm=llm, max_steps=6, token_budget=1500, seed=999)

    policies = {
        "always_cheap (k=5)": AlwaysCheapPolicy(k_retrieve=5),
        "always_expensive (k=5)": AlwaysExpensivePolicy(k_retrieve=5),
        "fixed_pipeline (k=3, cheap)": FixedPipelinePolicy(),
        "heuristic_threshold": HeuristicThresholdPolicy(),
    }
    print(f"{'Policy':<30} {'Acc':>6} {'AvgTok':>8} {'AvgStep':>8} {'AvgRew':>8} {'Tok/Correct':>12} {'Acc/1kTok':>10}")
    print("-" * 90)
    for name, policy in policies.items():
        res = evaluate_policy(final_eval_env, policy, n_episodes=300)
        print(f"{name:<30} {res['accuracy']:>6.2f} {res['avg_tokens']:>8.1f} {res['avg_steps']:>8.2f} "
              f"{res['avg_reward']:>8.3f} {res['tokens_per_correct']:>12.1f} {res['accuracy_per_1k_tokens']:>10.3f}")

    ppo_res = evaluate_ppo(model, final_eval_env, n_episodes=300)
    print(f"{'PPO (learned)':<30} {ppo_res['accuracy']:>6.2f} {ppo_res['avg_tokens']:>8.1f} {ppo_res['avg_steps']:>8.2f} "
          f"{ppo_res['avg_reward']:>8.3f} {ppo_res['tokens_per_correct']:>12.1f} {ppo_res['accuracy_per_1k_tokens']:>10.3f}")

    model.save("ppo_token_agent.zip")
    return model


if __name__ == "__main__":
    main()