"""Tool-use / function-calling task."""

TASK = {
    "title": "Tool Use (Function Calling)",
    "difficulty": "Medium",
    "function_name": "execute_tool_call",
    "hint": (
        "Parse structured tool-call from response_text using JSON or keyword pattern. "
        "Match tool_name against the tools dict, call it with parsed kwargs, return result. "
        "Return (tool_name, result) tuple, or None if no tool call detected."
    ),
    "tests": [
        {
            "name": "JSON-style tool call",
            "code": """
import torch, json
def add(a: int, b: int) -> int:
    return a + b
tools = {"add": add}
response = '{"tool": "add", "args": {"a": 3, "b": 5}}'
name, result = {fn}(response, tools)
assert name == "add", f"Expected 'add', got {name!r}"
assert result == 8, f"Expected 8, got {result}"
""",
        },
        {
            "name": "No tool call → None",
            "code": """
import torch
tools = {"search": lambda q: "found"}
res = {fn}("The answer is 42.", tools)
assert res is None, f"Expected None, got {res!r}"
""",
        },
        {
            "name": "Multiple tool types",
            "code": """
import torch, json
def greet(name: str) -> str:
    return f"Hello, {name}!"
def square(x: float) -> float:
    return x * x
tools = {"greet": greet, "square": square}
r1 = {fn}('{"tool": "greet", "args": {"name": "World"}}', tools)
assert r1 == ("greet", "Hello, World!"), f"got {r1}"
r2 = {fn}('{"tool": "square", "args": {"x": 7}}', tools)
assert r2 == ("square", 49), f"got {r2}"
""",
        },
    ],
}
