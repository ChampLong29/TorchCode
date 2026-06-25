"""Multi-agent message passing — route messages between specialized agents."""

TASK = {
    "title": "Multi-Agent Message Router",
    "difficulty": "Hard",
    "function_name": "agent_message_router",
    "hint": (
        "message = {'from': str, 'to': str|list, 'content': Any, 'type': str}. "
        "If 'to' is a single agent, call it directly. If it's a list, call each in sequence. "
        "If 'to' is '*' or 'broadcast', send to all agents except sender. "
        "Collect responses in order. Support 'type' filtering via routing_rules dict."
    ),
    "tests": [
        {
            "name": "Direct agent-to-agent message",
            "code": """
import torch
agents = {
    "alice": lambda msg: {"from": "alice", "to": msg["from"], "content": f"Reply to: {msg['content']}"},
    "bob": lambda msg: {"from": "bob", "to": msg["from"], "content": "Got it"},
}
responses = {fn}({"from": "alice", "to": "bob", "content": "Hello", "type": "request"}, agents)
assert len(responses) == 1, f"Expected 1 response, got {len(responses)}"
assert responses[0]["from"] == "bob", f"Response should come from bob: {responses[0]}"
""",
        },
        {
            "name": "Broadcast to all agents",
            "code": """
import torch
agents = {
    "alice": lambda msg: {"from": "alice", "status": "ok"},
    "bob": lambda msg: {"from": "bob", "status": "ok"},
    "charlie": lambda msg: {"from": "charlie", "status": "ok"},
}
responses = {fn}({"from": "system", "to": "*", "content": "ping"}, agents)
assert len(responses) == 3, f"Expected 3 broadcast responses, got {len(responses)}"
names = {r["from"] for r in responses}
assert names == {"alice", "bob", "charlie"}, f"Missing agents: {names}"
""",
        },
        {
            "name": "Multi-target routing with list",
            "code": """
import torch
agents = {
    "researcher": lambda msg: {"from": "researcher", "data": "found info"},
    "writer": lambda msg: {"from": "writer", "text": "summary"},
    "reviewer": lambda msg: {"from": "reviewer", "feedback": "looks good"},
}
responses = {fn}({"from": "manager", "to": ["researcher", "writer"], "content": "task"}, agents)
assert len(responses) == 2, f"Expected 2 responses, got {len(responses)}"
assert {r["from"] for r in responses} == {"researcher", "writer"}
""",
        },
    ],
}
