"""Reward shaping — combine outcome rewards with process rewards for agent training."""

TASK = {
    "title": "Reward Shaping & Process Rewards",
    "difficulty": "Medium",
    "function_name": "compute_shaped_reward",
    "hint": (
        "For each step: base_reward from env. If process_reward_fn: add shaping. "
        "At end: call outcome_reward_fn(result) and add to final step. "
        "Return tensor of per-step shaped rewards."
    ),
    "tests": [
        {
            "name": "Only outcome reward (sparse)",
            "code": """
import torch
step_rewards = [0.0, 0.0, 0.0]
result = {"answer": "Paris", "correct": True}
outcome_fn = lambda r: 1.0 if r["correct"] else -1.0
shaped = {fn}(step_rewards, result, True, outcome_fn, process_reward_fn=None)
# Only last step gets outcome bonus
assert shaped[-1] == 1.0, f"Last step should have outcome reward: {shaped}"
assert shaped[0] == 0.0 and shaped[1] == 0.0, f"Non-terminal steps should be 0: {shaped}"
""",
        },
        {
            "name": "Process rewards shape intermediate steps",
            "code": """
import torch
step_rewards = [0.0, 0.0, 0.0]
result = {"answer": "42", "correct": True}
outcome_fn = lambda r: 5.0 if r["correct"] else 0.0
process_fn = lambda step_idx, context: 0.5 if step_idx < 2 else 0.0
shaped = {fn}(step_rewards, result, True, outcome_fn, process_reward_fn=process_fn)
assert shaped[0] == 0.5, f"Step 0: expected 0.5, got {shaped[0]}"
assert shaped[1] == 0.5, f"Step 1: expected 0.5, got {shaped[1]}"
assert shaped[2] == 5.0, f"Step 2: expected 5.0, got {shaped[2]}"
""",
        },
        {
            "name": "Negative outcome for incorrect answer",
            "code": """
import torch
step_rewards = [0.1, 0.1]
result = {"answer": "wrong", "correct": False}
shaped = {fn}(step_rewards, result, True, lambda r: -1.0 if not r["correct"] else 1.0)
assert abs(shaped[-1] - (-0.9)) < 1e-5, f"Last step: expected -0.9, got {shaped[-1]}"
assert abs(shaped[1] - (-0.9)) < 1e-5, f"Step 1: expected -0.9, got {shaped[1]}"
""",
        },
    ],
}
