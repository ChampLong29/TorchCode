"""Reasoning trace parsing — separate thinking tokens from action tokens in agent output."""

TASK = {
    "title": "Reasoning Trace Processor",
    "difficulty": "Medium",
    "function_name": "split_reasoning_action",
    "hint": (
        "Find positions of special tokens: <think>, </think>, <action>, </action>. "
        "Create bool masks: reasoning between think tags, action between action tags. "
        "Return both masks as tensors of same length as token_ids."
    ),
    "tests": [
        {
            "name": "Basic split with both sections",
            "code": """
import torch
# Layout: <think> REASON </think> <action> ACT </action> PAD
token_ids = torch.tensor([1, 10, 11, 12, 2, 3, 20, 21, 22, 4, 5])
special_map = {"<think>": 1, "</think>": 2, "<action>": 3, "</action>": 4}
r_mask, a_mask = {fn}(token_ids, special_map)
# Reasoning: positions 1,2,3 (ids 10,11,12 between 1 and 2)
assert r_mask[1] and r_mask[2] and r_mask[3], f"Reasoning mask wrong: {r_mask}"
assert not r_mask[0] and not r_mask[4], f"Non-reasoning tokens in mask"
# Action: positions 6,7,8 (ids 20,21,22 between 3 and 4)
assert a_mask[6] and a_mask[7] and a_mask[8], f"Action mask wrong at pos 6-8: {a_mask}"
assert not a_mask[5] and not a_mask[9], f"Action markers in mask: {a_mask}"
""",
        },
        {
            "name": "No reasoning section",
            "code": """
import torch
token_ids = torch.tensor([1, 10, 20, 2, 3])
special_map = {"<think>": 99, "</think>": 98, "<action>": 1, "</action>": 2}
r_mask, a_mask = {fn}(token_ids, special_map)
# No 99/98 pair found, so no reasoning
assert not r_mask.any(), f"Should be no reasoning tokens"
# Action: between position 0 (id=1) and position 3 (id=2), so positions 1,2 are action
assert a_mask[1] and a_mask[2], f"Missing action tokens at pos 1-2: {a_mask}"
assert not a_mask[0] and not a_mask[3]
""",
        },
        {
            "name": "Nested / repeated sections",
            "code": """
import torch
# Two pairs: <think>A</think> <action>B</action> <think>C</think> <action>D</action>
token_ids = torch.tensor([1, 5, 2, 3, 7, 4, 1, 9, 2, 3, 8, 4])
special_map = {"<think>": 1, "</think>": 2, "<action>": 3, "</action>": 4}
r_mask, a_mask = {fn}(token_ids, special_map)
# First reasoning: pos 1 (id=5)
assert r_mask[1], "First reasoning token missing"
# Second reasoning: pos 7 (id=9)
assert r_mask[7], "Second reasoning token missing"
# First action: pos 4 (id=7)
assert a_mask[4], f"First action token missing, a_mask={a_mask}"
# Second action: pos 10 (id=8)
assert a_mask[10], f"Second action token missing, a_mask={a_mask}"
""",
        },
    ],
}
