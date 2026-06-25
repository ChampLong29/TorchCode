"""GAE (Generalized Advantage Estimation) — the standard advantage estimator in PPO/actor-critic."""

TASK = {
    "title": "GAE (Generalized Advantage Estimation)",
    "difficulty": "Hard",
    "function_name": "compute_gae",
    "hint": (
        "Loop backwards over timesteps: delta = r_t + gamma * V(t+1) * (1-done) - V(t). "
        "A_t = delta + gamma * lam * (1-done) * A_(t+1). "
        "Return advantages and returns (A_t + V_t). GAE(lambda=1) = Monte Carlo returns. "
        "GAE(lambda=0) = TD(0)."
    ),
    "tests": [
        {
            "name": "GAE(0) matches TD error",
            "code": """
import torch
rewards = torch.tensor([1.0, 1.0, 1.0])
values = torch.tensor([0.0, 0.0, 0.0, 0.0])  # V_0..V_3, V_3 is bootstrap
dones = torch.tensor([False, False, False])
adv, ret = {fn}(rewards, values, dones, gamma=0.99, lam=0.0)
# With V=0 everywhere, delta = r each step, A_t = r_t (since lam=0)
assert torch.allclose(adv, rewards, atol=1e-5), f"GAE(0) should match TD(0) deltas"
# Returns: R_t = r_t + gamma*V(t+1) = r_t (since V=0)
assert torch.allclose(ret, rewards, atol=1e-5)
""",
        },
        {
            "name": "GAE(1) matches MC returns",
            "code": """
import torch
rewards = torch.tensor([0.0, 0.0, 1.0])
values = torch.tensor([0.0, 0.0, 0.0, 0.0])
dones = torch.tensor([False, False, False])
adv, ret = {fn}(rewards, values, dones, gamma=1.0, lam=1.0)
# MC returns: [1, 1, 1] (all timesteps get same total, since no discount)
expected_ret = torch.tensor([1.0, 1.0, 1.0])
assert torch.allclose(ret, expected_ret, atol=1e-5), f"Expected {expected_ret}, got {ret}"
assert torch.allclose(adv, expected_ret, atol=1e-5), "With V=0, GAE(1)=MC returns"
""",
        },
        {
            "name": "Done flag zeroes future advantage",
            "code": """
import torch
rewards = torch.tensor([1.0, 1.0])
values = torch.tensor([0.0, 0.0, 0.0])
dones = torch.tensor([False, True])
adv, ret = {fn}(rewards, values, dones, gamma=0.9, lam=0.5)
# t=1: done, so no future: A_1 = r_1 - V_1 + 0 = 1.0
assert torch.allclose(adv[1], torch.tensor(1.0), atol=1e-5), f"A_1 should be 1.0, got {adv[1]}"
# t=0: A_0 = delta_0 + gamma*lam*(1-done)*A_1 = 1.0 + 0.9*0.5*1*1.0 = 1.45
expected_A0 = 1.0 + 0.9 * 0.5 * 1.0
assert torch.allclose(adv[0], torch.tensor(expected_A0), atol=1e-5), f"A_0: expected {expected_A0}, got {adv[0]}"
""",
        },
    ],
}
