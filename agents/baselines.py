"""
Baseline policies. Each is a callable: obs, env_internal_info -> action (int).
We pass the env itself (not just obs) because some baselines need access
to things like 'has the LLM been called yet' beyond the obs vector --
this keeps the baselines simple and readable rather than re-deriving
everything from the 9-dim state.
"""
from env.actions import Action


class AlwaysCheapPolicy:
    """Retrieve a fixed k, call cheap once, stop. No adaptivity."""
    def __init__(self, k_retrieve: int = 5):
        self.k_retrieve = {1: Action.RETRIEVE_1, 3: Action.RETRIEVE_3, 5: Action.RETRIEVE_5}[k_retrieve]

    def act(self, step_idx: int, env) -> Action:
        if step_idx == 0:
            return self.k_retrieve
        if step_idx == 1:
            return Action.CALL_CHEAP
        return Action.STOP


class AlwaysExpensivePolicy:
    """Retrieve a fixed k, call expensive once, stop."""
    def __init__(self, k_retrieve: int = 5):
        self.k_retrieve = {1: Action.RETRIEVE_1, 3: Action.RETRIEVE_3, 5: Action.RETRIEVE_5}[k_retrieve]

    def act(self, step_idx: int, env) -> Action:
        if step_idx == 0:
            return self.k_retrieve
        if step_idx == 1:
            return Action.CALL_EXPENSIVE
        return Action.STOP


class FixedPipelinePolicy:
    """The 'naive RAG' baseline: retrieve_3 -> call_cheap -> stop, every time,
    regardless of difficulty. This is what most production systems actually
    ship before anyone thinks about token efficiency."""
    def act(self, step_idx: int, env) -> Action:
        if step_idx == 0:
            return Action.RETRIEVE_3
        if step_idx == 1:
            return Action.CALL_CHEAP
        return Action.STOP


class HeuristicThresholdPolicy:
    """Retrieve cheaply first; call cheap; if confidence (agreement) is low,
    retry with more docs + expensive model; else stop. This is the 'smart
    engineer wrote some if-statements' baseline -- the bar RL actually
    needs to beat to be worth anything."""
    def __init__(self, agreement_threshold: float = 0.99):
        self.agreement_threshold = agreement_threshold

    def act(self, step_idx: int, env) -> Action:
        if step_idx == 0:
            return Action.RETRIEVE_3
        if step_idx == 1:
            return Action.CALL_CHEAP
        if step_idx == 2:
            # cheap call always has agreement 1.0 (n_samples=1), so use
            # whether the answer looks like UNKNOWN/garbage as our signal
            if env._final_answer in (None, "UNKNOWN"):
                return Action.RETRIEVE_5
            return Action.STOP
        if step_idx == 3:
            return Action.CALL_EXPENSIVE
        return Action.STOP


def run_episode(env, policy) -> dict:
    obs, info = env.reset()
    step_idx = 0
    term, trunc = False, False
    final_info = {}
    while not term and not trunc:
        action = policy.act(step_idx, env)
        obs, reward, term, trunc, info = env.step(int(action))
        step_idx += 1
        if term or trunc:
            final_info = info
            final_info["reward"] = reward
    return final_info


def evaluate_policy(env, policy, n_episodes: int) -> dict:
    results = []
    for _ in range(n_episodes):
        results.append(run_episode(env, policy))
    n = len(results)
    accuracy = sum(r.get("is_correct", False) for r in results) / n
    avg_tokens = sum(r.get("tokens_used", 0) for r in results) / n
    avg_steps = sum(r.get("steps_used", 0) for r in results) / n
    avg_reward = sum(r.get("reward", 0.0) for r in results) / n
    n_correct = sum(r.get("is_correct", False) for r in results)
    tokens_per_correct = (sum(r.get("tokens_used", 0) for r in results) / n_correct) if n_correct > 0 else float("inf")
    acc_per_1k_tokens = (accuracy / (avg_tokens / 1000)) if avg_tokens > 0 else 0.0
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "tokens_per_correct": tokens_per_correct,
        "accuracy_per_1k_tokens": acc_per_1k_tokens,
        "n_episodes": n,
    }