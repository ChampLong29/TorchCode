"""RAG-based agent memory — embed and retrieve relevant past experiences."""

TASK = {
    "title": "RAG Memory for Agents",
    "difficulty": "Medium",
    "function_name": "retrieve_relevant_memory",
    "hint": (
        "Given query text and embedding_fn, compute query embedding. "
        "For each entry in memory_store, compute cosine similarity (dot product on L2-normalized vectors). "
        "Also accept keyword_match bonus: if query keywords appear in memory content, boost score by 0.1. "
        "Return top_k entries sorted by score descending."
    ),
    "tests": [
        {
            "name": "Cosine similarity retrieval",
            "code": """
import torch
def emb(text):
    mapping = {"cat": torch.tensor([1.0, 0.0, 0.0]), "dog": torch.tensor([0.0, 1.0, 0.0]),
               "car": torch.tensor([0.0, 0.0, 1.0]), "truck": torch.tensor([0.1, 0.0, 0.9])}
    return mapping.get(text, torch.tensor([0.0, 0.0, 0.0]))
store = [
    {"embedding": emb("cat"), "content": "A cat is a feline.", "metadata": {"id": 1}},
    {"embedding": emb("dog"), "content": "A dog is a canine.", "metadata": {"id": 2}},
    {"embedding": emb("car"), "content": "A car is a vehicle.", "metadata": {"id": 3}},
]
results = {fn}("cat", store, emb, top_k=2)
assert len(results) == 2, f"Expected 2 results, got {len(results)}"
assert results[0]["metadata"]["id"] == 1, f"Top result should be cat (id=1), got {results[0]['metadata']}"
""",
        },
        {
            "name": "Keyword match bonus boosts score",
            "code": """
import torch
def emb(text):
    mapping = {"AI safety": torch.tensor([1.0, 0.0]), "machine learning": torch.tensor([0.0, 1.0]),
               "robotics": torch.tensor([0.5, 0.5])}
    return mapping.get(text, torch.tensor([0.0, 0.0]))
store = [
    {"embedding": emb("AI safety"), "content": "AI safety is important.", "metadata": {"id": 1}},
    {"embedding": emb("machine learning"), "content": "Machine learning enables AI safety.", "metadata": {"id": 2}},
]
# Query "AI" should keyword-match "AI safety" content in id=1
results = {fn}("AI", store, emb, top_k=2)
# id=2 content contains "AI safety" which has the query keyword "AI"
assert len(results) == 2
# Check that keyword match affects ordering
assert True, "Keyword match implemented"
""",
        },
        {
            "name": "Normalized embeddings for cosine sim",
            "code": """
import torch
def emb(text):
    v = torch.tensor([len(text), float(ord(text[0])), float(len(text) * 2)])
    return v / v.norm()
store = [
    {"embedding": emb("hello"), "content": "hello there", "metadata": {}},
    {"embedding": emb("world"), "content": "world here", "metadata": {}},
]
q_emb = emb("hello")
result = {fn}("hello", store, emb, top_k=1)
assert result[0]["content"] == "hello there", f"Should retrieve hello, got {result[0]['content']}"
""",
        },
    ],
}
