import sys
import json
from data.synthetic_dataset import generate_dataset

samples = generate_dataset(num_samples=10, seed=42)
print(f"Generated {len(samples)} samples")
print(f"Areas: {set(s['area'] for s in samples)}")

sample = samples[0]
print(f"\nSample 0 area: {sample['area']}")
print(f"\n--- FULL TEXT (first 1000 chars) ---")
print(sample["text"][:1000])
print("...")

required_patterns = ["<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<tool_calls>", "<tool_results>", "web_search"]
for p in required_patterns:
    assert p in sample["text"], f"Missing pattern: {p}"
print("\nAll required patterns present!")

token_estimate = len(sample["text"].split())
print(f"Approx token count: {token_estimate}")
assert token_estimate < 4000, f"Sample too long: {token_estimate} tokens"
print("Sample length OK!")
