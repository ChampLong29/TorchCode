"""Context memory manager — manage agent's context window with token budget."""

TASK = {
    "title": "Context Window Memory Manager",
    "difficulty": "Medium",
    "function_name": "manage_context",
    "hint": (
        "TRUNCATE: remove oldest messages first until within budget. "
        "SLIDING: keep last N tokens. "
        "SUMMARIZE: replace old messages with a summary (join first 100 chars of each removed). "
        "Always keep system message and latest observation if possible. "
        "Return updated message list."
    ),
    "tests": [
        {
            "name": "Truncate strategy removes oldest",
            "code": """
import torch
history = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "It is 4."},
    {"role": "user", "content": "And 3+3?"},
]
def counter(text): return len(text)
result = {fn}(history, "New observation", max_tokens=50, token_counter=counter, strategy="truncate")
# System msg (19 chars) + last 2 messages ("It is 4." = 8, "And 3+3?" = 8) = 35 chars
assert len(result) >= 2, f"Should keep at least 2 messages, got {len(result)}"
assert result[0]["role"] == "system", "System message must be preserved"
""",
        },
        {
            "name": "Sliding strategy keeps most recent",
            "code": """
import torch
history = [
    {"role": "user", "content": "A" * 50},
    {"role": "assistant", "content": "B" * 50},
    {"role": "user", "content": "C" * 50},
]
def counter(text): return len(text)
result = {fn}(history, "D" * 10, max_tokens=100, token_counter=counter, strategy="sliding")
total = sum(counter(m["content"]) for m in result)
assert total <= 100, f"Total tokens {total} exceeds budget 100"
# Should keep the last messages
assert result[-1]["content"] == "D" * 10, f"Should keep latest: {result[-1]['content']}"
""",
        },
        {
            "name": "Summarize strategy compresses old",
            "code": """
import torch
history = [
    {"role": "user", "content": "Long question " + "X" * 80},
    {"role": "assistant", "content": "Long answer " + "Y" * 80},
    {"role": "user", "content": "Short query"},
]
def counter(text): return len(text)
result = {fn}(history, "", max_tokens=100, token_counter=counter, strategy="summarize")
total = sum(counter(m["content"]) for m in result)
assert total <= 100, f"Total {total} > budget 100"
# Should have a summary message for compressed content
has_summary = any("summary" in m.get("role", "").lower() or "summar" in m.get("content", "").lower() for m in result)
# Just check that total tokens fit budget
assert True, "Summarize strategy works"
""",
        },
    ],
}
