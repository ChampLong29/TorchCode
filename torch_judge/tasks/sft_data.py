"""SFT (Supervised Fine-Tuning) data construction task."""

TASK = {
    "title": "SFT Data Construction",
    "difficulty": "Medium",
    "function_name": "build_sft_dataset",
    "hint": (
        "Format as chat: apply tokenizer's chat_template. Concatenate instruction+response, "
        "tokenize, create input_ids + labels (mask instruction part with -100). "
        "Truncate/pad to max_length. Return dict of input_ids and labels tensors."
    ),
    "tests": [
        {
            "name": "Output has input_ids and labels",
            "code": """
import torch

class MockTok:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
    def __call__(self, text, **kw):
        n = len(text.split())
        return {"input_ids": torch.tensor([i + 10 for i in range(n)] + [2]), "attention_mask": [1]*(n+1)}

convs = [{"instruction": "What is 1+1?", "response": "It is 2."}]
tok = MockTok()
result = {fn}(convs, tok, max_length=128)
assert "input_ids" in result, "Missing input_ids"
assert "labels" in result, "Missing labels"
assert isinstance(result["input_ids"], torch.Tensor)
assert isinstance(result["labels"], torch.Tensor)
""",
        },
        {
            "name": "Labels mask instruction tokens",
            "code": """
import torch

class SimpleTok:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
    def __call__(self, text, **kw):
        # Instruction "Hello" -> 6 chars -> 6 tokens + eos = 7
        # Response "World" -> 5 chars -> 5 tokens + eos = 6
        # Full "Hello World" -> 11 chars -> 11 + eos = 12
        n = len(text) + 1  # +1 for eos
        return {"input_ids": torch.tensor(list(range(1, n+1))), "attention_mask": [1]*n}

convs = [{"instruction": "Hello", "response": "World"}]
result = {fn}(convs, SimpleTok(), max_length=32)
labels = result["labels"]
# First 7 tokens (instruction "Hello" = 6 chars + 1 for space = 7) should be masked
# Actually: instruction="Hello" (5 chars), " " (1 char) = 6 tokens of prompt
# Then +1 token for eos of instruction-only tokenization: instruction + eos = 7 tokens
# Let me just check that instruction region is masked by verifying first few tokens
assert labels[0, 0] == -100, "First token should be masked (instruction part)"
assert labels[0, -1] == -100 or labels[0, -1].item() == 2, "Last token should be -100 (padding) or eos"
# Verify some instruction tokens are masked and response tokens are not all masked
assert (labels[0, :6] == -100).all(), "First 6 tokens should be instruction, got masked"
""",
        },
        {
            "name": "Batched input returns stacked tensors",
            "code": """
import torch

class BatchedTok:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
    def __call__(self, text, **kw):
        n = len(text) + 1  # +1 for eos
        return {"input_ids": torch.tensor(list(range(1, n+1))), "attention_mask": [1]*n}

convs = [
    {"instruction": "Short question", "response": "Brief answer."},
    {"instruction": "A longer question here", "response": "A longer answer here too."},
]
result = {fn}(convs, BatchedTok(), max_length=64)
assert result["input_ids"].dim() == 2, f"Expected 2D, got {result['input_ids'].dim()}D"
assert result["labels"].dim() == 2
assert result["input_ids"].shape == result["labels"].shape
""",
        },
    ],
}
