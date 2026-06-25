"""Trajectory rollout task."""

TASK = {
    "title": "Trajectory Rollout",
    "difficulty": "Medium",
    "function_name": "rollout",
    "hint": (
        "Loop: state = env.reset(), then for each step: action = policy_fn(state), "
        "next_state, reward, done = env.step(action). "
        "Store (state, action, reward, next_state, done) tuples. Stop on done or max_steps. "
        "Return trajectory as list of dicts."
    ),
    "tests": [
        {
            "name": "Fixed policy → known trajectory",
            "code": """
import torch

class MockEnv:
    def reset(self):
        self.state = torch.tensor([0.0])
        return self.state
    def step(self, action):
        s = self.state = self.state + action
        return s, float(s), abs(s) >= 3.0

env = MockEnv()
traj = {fn}(lambda s: torch.tensor(1.0), env, max_steps=5)
assert len(traj) == 3, f"Expected 3 steps, got {len(traj)}"
assert traj[0]["state"] == 0.0
assert traj[0]["action"] == 1.0
assert traj[-1]["reward"] == 3.0
assert traj[-1]["done"] == True
""",
        },
        {
            "name": "Random policy → returns list of dicts",
            "code": """
import torch, random
class SimpleEnv:
    def __init__(self): self.n = 0
    def reset(self): self.n = 0; return torch.tensor(0.0)
    def step(self, a):
        self.n += 1
        return torch.tensor(float(self.n)), 1.0, self.n >= 2

traj = {fn}(lambda s: torch.tensor(random.uniform(-1, 1)),
             SimpleEnv(), max_steps=10)
assert isinstance(traj, list) and len(traj) > 0, "Empty trajectory"
for t in traj:
    assert set(t.keys()) == {"state", "action", "reward", "next_state", "done"}, f"Bad keys: {t.keys()}"
""",
        },
        {
            "name": "Respects max_steps",
            "code": """
import torch
class InfiniteEnv:
    def reset(self): return 0.0
    def step(self, a): return 0.0, 0.0, False
traj = {fn}(lambda s: 0.0, InfiniteEnv(), max_steps=7)
assert len(traj) == 7, f"Expected 7, got {len(traj)}"
assert all(not t["done"] for t in traj)
""",
        },
    ],
}
