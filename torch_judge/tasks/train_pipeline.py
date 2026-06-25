"""Full RL training pipeline task (rollout + PPO loss + update)."""

TASK = {
    "title": "RL Training Pipeline",
    "difficulty": "Hard",
    "function_name": "train_rl_pipeline",
    "hint": (
        "Outer loop over epochs: 1) run rollout to collect trajectories, "
        "2) compute log-probs and advantages (GAE or simple reward-to-go), "
        "3) compute PPO-style clipped loss, 4) optimizer.step() + zero_grad(). "
        "Return list of mean losses per epoch."
    ),
    "tests": [
        {
            "name": "Loss decreases over epochs",
            "code": """
import torch
import torch.nn as nn

class SimplePolicy(nn.Module):
    def __init__(self): super().__init__(); self.w = nn.Parameter(torch.tensor([1.0]))
    def forward(self, x): return self.w.abs() * x  # pushes w away from zero

class SimpleEnv:
    def __init__(self): self.i = 0
    def reset(self): self.i = 0; return torch.tensor(1.0)
    def step(self, a):
        self.i += 1
        s = torch.tensor(float(self.i + 1))
        r = -float(a)  # penalty for large actions
        return s, r, self.i >= 3

policy = SimplePolicy()
ref = SimplePolicy()
ref.load_state_dict(policy.state_dict())
opt = torch.optim.SGD(policy.parameters(), lr=0.1)
losses = {fn}(policy, ref, SimpleEnv(), opt, epochs=5, rollouts_per_epoch=1)
assert len(losses) == 5, f"Expected 5 losses, got {len(losses)}"
# Loss should generally decrease (at least not explode)
assert losses[-1] < losses[0] + 1.0, f"Loss may have exploded: {losses}"
""",
        },
        {
            "name": "Returns list of scalars",
            "code": """
import torch
import torch.nn as nn
class Dummy(nn.Module):
    def forward(self, x): return x
class DummyEnv:
    def reset(self): return torch.tensor(0.0)
    def step(self, a): return torch.tensor(0.0), 1.0, True
class TrainableDummy(nn.Module):
    def __init__(self): super().__init__(); self.w = nn.Parameter(torch.tensor(1.0))
    def forward(self, x): return x * self.w
losses = {fn}(TrainableDummy(), TrainableDummy(), DummyEnv(),
              torch.optim.SGD(TrainableDummy().parameters(), lr=0.0),
              epochs=2, rollouts_per_epoch=1)
assert isinstance(losses, list), "Must return list"
assert all(isinstance(l, float) for l in losses), "All elements must be floats"
""",
        },
    ],
}
