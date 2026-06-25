"""Pretraining data cleaning and filtering task."""

TASK = {
    "title": "Pretraining Data Cleaning",
    "difficulty": "Medium",
    "function_name": "clean_pretrain_data",
    "hint": (
        "1) Filter by length (min_len…max_len). 2) Remove lines with high char-repetition ratio. "
        "3) Remove near-duplicates via n-gram Jaccard similarity. "
        "4) Strip leading/trailing whitespace. Return cleaned list."
    ),
    "tests": [
        {
            "name": "Filters short texts",
            "code": """
import torch
texts = ["hi", "this is a longer sentence", "ok"]
result = {fn}(texts, min_length=10, max_length=512)
assert len(result) == 1, f"Expected 1, got {len(result)}: {result}"
assert "longer" in result[0]
""",
        },
        {
            "name": "Filters repetitive text",
            "code": """
import torch
texts = [
    "Normal text here for testing purposes.",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "Another normal sentence with content.",
]
result = {fn}(texts, min_length=5, max_length=512)
assert len(result) == 2, f"Expected 2, got {len(result)}: {result}"
assert result[0] == texts[0]
assert result[1] == texts[2]
""",
        },
        {
            "name": "Deduplicates near-duplicates",
            "code": """
import torch
texts = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy cat",  # near-duplicate
    "Python is a great programming language",
]
result = {fn}(texts, min_length=5, max_length=512, dedup_threshold=0.7)
assert len(result) == 2, f"Expected 2, got {len(result)}: {result}"
# The near-duplicate pair should be collapsed
""",
        },
        {
            "name": "Strips whitespace",
            "code": """
import torch
texts = ["  hello world  ", "\\n\\ngoodbye\\t\\t"]
result = {fn}(texts, min_length=1, max_length=512)
assert result[0] == "hello world"
assert result[1] == "goodbye"
""",
        },
    ],
}
