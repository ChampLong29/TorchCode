"""Policy model configuration — set up model for agent RL training."""

TASK = {
    "title": "Policy & Value Head Configuration",
    "difficulty": "Medium",
    "function_name": "configure_policy_model",
    "hint": (
        "policy_head: nn.Linear(d_model, hidden) -> nn.ReLU -> nn.Linear(hidden, action_dim). "
        "value_head: nn.Linear(d_model, hidden) -> nn.ReLU -> nn.Linear(hidden, 1). "
        "Return both as nn.Module. Freeze base by setting requires_grad=False on all params. "
        "He initialization (Kaiming uniform) for Linear layers."
    ),
    "tests": [
        {
            "name": "Returns two nn.Module instances",
            "code": """
import torch, torch.nn as nn
class DummyBase(nn.Module):
    def __init__(self): super().__init__(); self.linear = nn.Linear(64, 128)
    def forward(self, x): return self.linear(x)
base = DummyBase()
p_head, v_head = {fn}(base, action_dim=8, hidden_dim=64)
assert isinstance(p_head, nn.Module), f"Policy head must be nn.Module, got {type(p_head)}"
assert isinstance(v_head, nn.Module), f"Value head must be nn.Module, got {type(v_head)}"
""",
        },
        {
            "name": "Policy head outputs correct shape",
            "code": """
import torch, torch.nn as nn
class DummyBase(nn.Module):
    def __init__(self): super().__init__(); self.linear = nn.Linear(64, 128)
    def forward(self, x): return self.linear(x)
base = DummyBase()
p_head, v_head = {fn}(base, action_dim=5, hidden_dim=32)
x = torch.randn(2, 10, 64)  # (B, L, d_model)
h = base(x)
logits = p_head(h)
assert logits.shape == (2, 10, 5), f"Policy output shape: expected (2,10,5), got {logits.shape}"
""",
        },
        {
            "name": "Value head outputs scalar per token",
            "code": """
import torch, torch.nn as nn
class DummyBase(nn.Module):
    def __init__(self): super().__init__(); self.linear = nn.Linear(64, 128)
    def forward(self, x): return self.linear(x)
base = DummyBase()
p_head, v_head = {fn}(base, action_dim=5, hidden_dim=32)
x = torch.randn(3, 8, 64)
h = base(x)
values = v_head(h)
assert values.shape == (3, 8, 1), f"Value output shape: expected (3,8,1), got {values.shape}"
""",
        },
    ],
}
