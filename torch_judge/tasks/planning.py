"""Planning & self-correction loop — plan-then-execute with error handling."""

TASK = {
    "title": "Planning & Self-Correction Loop",
    "difficulty": "Hard",
    "function_name": "planning_execute_loop",
    "hint": (
        "1) Generate plan via planner_fn(task, context). "
        "2) Execute each step via executor_fn(step, context). "
        "3) If step fails and retries < max_retries: re-plan remaining steps. "
        "4) Optional verifier_fn checks step results. "
        "5) Return (final_result, trace) where trace is a list of step execution dicts."
    ),
    "tests": [
        {
            "name": "Simple plan execution succeeds",
            "code": """
import torch
plan = ["step_a", "step_b", "step_c"]
plan_idx = [0]
def planner(task, ctx):
    return ["step_a", "step_b", "step_c"]
def executor(step, ctx):
    return {"result": f"done_{step}", "success": True, "step": step}
result, trace = {fn}("task", planner, executor)
assert result == {"result": "done_step_c", "success": True, "step": "step_c"}, f"Wrong result: {result}"
assert len(trace) == 3, f"Expected 3 steps, got {len(trace)}"
""",
        },
        {
            "name": "Retry on failure + re-plan",
            "code": """
import torch
attempts = {"step_b": 0}
def planner(task, ctx):
    if ctx.get("replan_count", 0) > 0:
        return ["step_x", "step_y"]  # new plan
    return ["step_a", "step_b", "step_c"]
def executor(step, ctx):
    if step == "step_b":
        attempts["step_b"] += 1
        if attempts["step_b"] < 3:
            return {"result": "error", "success": False, "step": step}
    return {"result": f"done_{step}", "success": True, "step": step}
result, trace = {fn}("task", planner, executor, max_retries=3)
# step_b should succeed on 3rd attempt
assert result["success"], f"Should eventually succeed: {result}"
assert attempts["step_b"] == 3, f"Expected 3 attempts, got {attempts}"
""",
        },
        {
            "name": "Returns failure after max retries",
            "code": """
import torch
def planner(task, ctx):
    return ["impossible_step"]
def executor(step, ctx):
    return {"result": "fail", "success": False}
result, trace = {fn}("hard task", planner, executor, max_retries=2)
assert not result["success"], f"Should fail after max retries: {result}"
assert len(trace) <= 3, f"Should stop early: {len(trace)}"
""",
        },
    ],
}
