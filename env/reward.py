"""
Reward function. Computed only at STOP (terminal reward), per the
environment spec -- correctness minus token cost minus step penalty.
"""


def terminal_reward(
    is_correct: bool,
    tokens_used: int,
    steps_used: int,
    lambda_token: float = 0.001,
    alpha_step: float = 0.02,
    partial_credit: float = 0.0,
    explicit_stop: bool = True,
    forced_stop_penalty: float = 0.05,
) -> float:
    """
    R = 1[correct] - lambda * tokens - alpha * steps (+ optional partial credit
    for near-misses, e.g. UNKNOWN vs a wrong confident answer).

    forced_stop_penalty: extra penalty applied when the episode ends via
    budget/step exhaustion rather than an explicit STOP action. Without this,
    a policy has no incentive to ever call STOP, since running out the clock
    yields identical reward -- it'll just pad with free/no-op actions.
    """
    correctness = 1.0 if is_correct else partial_credit
    penalty = lambda_token * tokens_used + alpha_step * steps_used
    if not explicit_stop:
        penalty += forced_stop_penalty
    return correctness - penalty


def non_terminal_reward() -> float:
    """No shaping reward for intermediate steps -- keeps reward purely
    outcome+cost driven, per spec. (Kept as a function in case shaping
    is needed later, e.g. small bonus for retrieving a high-score doc.)"""
    return 0.0