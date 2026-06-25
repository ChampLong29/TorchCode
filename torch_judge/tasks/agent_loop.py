"""Minimal ReAct-style agent loop task."""

TASK = {
    "title": "ReAct Agent Loop",
    "difficulty": "Medium",
    "function_name": "react_agent_loop",
    "hint": (
        "Parse Thought/Action/Observation from model output using regex. "
        "Execute tool call, feed observation back. Stop on 'Final Answer:' or max_steps. "
        "Accumulate history as list of dicts."
    ),
    "tests": [
        {
            "name": "Single step → Final Answer",
            "code": """
import torch
responses = iter(["Final Answer: 42"])
res = {fn}(lambda _: next(responses), dict(), "what is the answer?", max_steps=5)
assert res == "42", f"Expected '42', got {res!r}"
""",
        },
        {
            "name": "Two tool-call + final answer",
            "code": """
import torch, re
responses = iter([
    "Thought: I need to look up the capital.\\nAction: lookup\\nAction Input: France",
    "Thought: I know it now.\\nFinal Answer: Paris",
])
tools = {"lookup": lambda x: "Paris" if x == "France" else "unknown"}
res = {fn}(lambda _: next(responses), tools, "capital of France?", max_steps=5)
assert res == "Paris", f"Expected 'Paris', got {res!r}"
""",
        },
        {
            "name": "Respects max_steps",
            "code": """
import torch
responses = iter([
    "Thought: thinking...\\nAction: search\\nAction Input: x",
    "Thought: still thinking...\\nAction: search\\nAction Input: y",
    "Thought: more...\\nAction: search\\nAction Input: z",
])
tools = {"search": lambda q: "no result for " + q}
res = {fn}(lambda _: next(responses), tools, "query", max_steps=2)
# Should return None or error message when max steps hit
assert res is None or isinstance(res, str), f"Unexpected: {res!r}"
""",
        },
        {
            "name": "No tools → direct answer",
            "code": """
import torch
responses = iter(["Final Answer: 3.14"])
res = {fn}(lambda _: next(responses), dict(), "what is pi?", max_steps=3)
assert res == "3.14", f"Expected '3.14', got {res!r}"
""",
        },
    ],
}
