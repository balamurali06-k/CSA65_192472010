# ============================================================
# QUESTION 9:
# Engineering Text Generation using GPT-2
# ============================================================

from transformers import pipeline

# Load pre-trained GPT-2 text generation model
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# Engineering-related prompt
prompt = """
Artificial Intelligence is transforming modern engineering by
"""

print("========================================")
print("     GPT-2 ENGINEERING TEXT GENERATOR")
print("========================================")

print("\nPrompt:")
print(prompt)

# Generate text
result = generator(
    prompt,
    max_new_tokens=100,
    num_return_sequences=1,
    do_sample=True
)

print("\nGenerated Output:")
print(result[0]["generated_text"])