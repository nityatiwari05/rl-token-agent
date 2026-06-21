import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.generator import generate_dataset
from llm.mock_provider import MockProvider
from env.pomdp_env import TokenBudgetEnv
from env.state import STATE_DIM
from agents.reinforce_agent import ReinforceAgent


def main():
    train_worlds = generate_dataset(n_worlds=300, seed=42, n_distractors_range=(3, 14))
    llm = MockProvider(base_error_rate=0.2, seed=42)
    env = TokenBudgetEnv(worlds=train_worlds, llm=llm, max_steps=8, token_budget=1500, seed=42)

    agent = ReinforceAgent(state_dim=STATE_DIM, lr=3e-3, gamma=0.99)

    n_updates = 400
    batch_size = 16
    eval_every = 50
    for upd in range(1, n_updates + 1):
        batch_metrics = agent.train_batch(env, batch_size=batch_size, entropy_coef=0.03)
        if upd % eval_every == 0:
            metrics = agent.evaluate(env, n_episodes=100)
            print(f"[update {upd}] (train batch: acc={batch_metrics['batch_accuracy']:.2f} "
                  f"tok={batch_metrics['batch_avg_tokens']:.1f}) | "
                  f"GREEDY EVAL acc={metrics['accuracy']:.2f} avg_tok={metrics['avg_tokens']:.1f} "
                  f"avg_steps={metrics['avg_steps']:.2f} avg_reward={metrics['avg_reward']:.3f} "
                  f"acc/1ktok={metrics['accuracy_per_1k_tokens']:.3f}")

    print("\nFinal evaluation (greedy policy, 300 eps):")
    final = agent.evaluate(env, n_episodes=300)
    for k, v in final.items():
        print(f"  {k}: {v}")

    return agent, final


if __name__ == "__main__":
    main()