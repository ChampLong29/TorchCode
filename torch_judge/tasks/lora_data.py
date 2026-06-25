"""LoRA data preparation task."""

TASK = {
    "title": "LoRA Data Preparation",
    "difficulty": "Medium",
    "function_name": "prepare_lora_data",
    "hint": (
        "Apply prompt_template.format(instruction=...). Tokenize prompt + completion together. "
        "Tokenize prompt-only (with completion='') to find boundary. "
        "Create labels that mask prompt tokens (-100) and keep completion tokens. "
        "Ensure proper padding and truncation. Return dict with input_ids and labels."
    ),
    "tests": [
        {
            "name": "Prompt template applied",
            "code": """
import torch

class MockTok:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
    def __call__(self, text, **kw):
        n = len(text) + 1  # +1 for eos token
        return {"input_ids": torch.tensor(list(range(1, n+1))), "attention_mask": [1]*n}

template = "Q: {instruction}\\nA: {completion}"
data = [{"instruction": "hi", "completion": "hello"}]
result = {fn}(data, MockTok(), max_length=64, prompt_template=template)
assert "input_ids" in result
assert "labels" in result
assert isinstance(result["input_ids"], torch.Tensor)
""",
        },
        {
            "name": "Prompt tokens properly masked",
            "code": """
import torch

class CharTok:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
    def __call__(self, text, **kw):
        ids = [ord(c) for c in text] + [self.eos_token_id]
        return {"input_ids": torch.tensor(ids), "attention_mask": [1]*len(ids)}

template = "### Instruction: {instruction}\\n### Response: {completion}"
data = [{"instruction": "Say hi", "completion": "hi"}]
result = {fn}(data, CharTok(), max_length=256, prompt_template=template)

labels = result["labels"][0]
prompt_text = template.format(instruction="Say hi", completion="")
expected_prompt_len = len(prompt_text) + 1  # +1 for eos token of prompt-only

# First expected_prompt_len tokens should be -100 (prompt region)
assert (labels[:expected_prompt_len] == -100).all(), \\
    f"First {expected_prompt_len} tokens (prompt region) should be -100"
# Completion tokens ("hi" + eos) should not be -100
assert (labels[expected_prompt_len:expected_prompt_len+3] != -100).any(), \\
    "Completion tokens should not be -100"
""",
        },
        {
            "name": "Truncation to max_length",
            "code": """
import torch

class TruncTok:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
    def __call__(self, text, **kw):
        ids = torch.arange(1, 200).tolist() + [self.eos_token_id]
        return {"input_ids": torch.tensor(ids), "attention_mask": [1]*len(ids)}

result = {fn}([{"instruction": "x", "completion": "y"}], TruncTok(),
              max_length=16, prompt_template="{instruction} {completion}")
assert result["input_ids"].shape[1] <= 16, \\
    f"Should be truncated to max_length=16, got shape {result['input_ids'].shape}"
assert result["labels"].shape[1] <= 16
""",
        },
    ],
}
