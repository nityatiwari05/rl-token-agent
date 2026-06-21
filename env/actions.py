"""
Discrete action space. Kept deliberately small per project scoping.
"""
from enum import IntEnum


class Action(IntEnum):
    RETRIEVE_1 = 0
    RETRIEVE_3 = 1
    RETRIEVE_5 = 2
    CALL_CHEAP = 3       # 1 sample
    CALL_EXPENSIVE = 4   # 3 samples, majority vote
    REFINE_QUERY = 5     # re-rank with a small score boost (simulates query rewriting)
    STOP = 6


N_ACTIONS = len(Action)

RETRIEVE_K = {
    Action.RETRIEVE_1: 1,
    Action.RETRIEVE_3: 3,
    Action.RETRIEVE_5: 5,
}