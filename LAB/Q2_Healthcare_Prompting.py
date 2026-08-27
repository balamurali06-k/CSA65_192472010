# ============================================================
# Question 2: Zero-shot, One-shot, and Few-shot Prompting
# ============================================================

from transformers import pipeline

# Load text generation model
generator = pipeline("text-generation", model="gpt2")

# Zero-shot prompt
zero_shot = """
Write a 200-word blog on 'Applications of Artificial Intelligence
in Healthcare'.
"""

# One-shot prompt
one_shot = """
Example:
Topic: Artificial Intelligence in Education
Blog: Artificial Intelligence is transforming education by
providing personalized learning, automated assessment, and
intelligent tutoring systems.

Now write a 200-word blog on 'Applications of Artificial
Intelligence in Healthcare'.
"""

# Few-shot prompt
few_shot = """
Example 1:
Topic: AI in Education
Blog: AI helps students through personalized learning,
automated evaluation, and intelligent tutoring.

Example 2:
Topic: AI in Transportation
Blog: AI improves transportation through autonomous vehicles,
traffic prediction, and route optimization.

Now write a 200-word blog on 'Applications of Artificial
Intelligence in Healthcare'.
"""

# Generate outputs
print("========== ZERO-SHOT OUTPUT ==========")
print(generator(zero_shot, max_length=250, num_return_sequences=1)[0]["generated_text"])

print("\n========== ONE-SHOT OUTPUT ==========")
print(generator(one_shot, max_length=250, num_return_sequences=1)[0]["generated_text"])

print("\n========== FEW-SHOT OUTPUT ==========")
print(generator(few_shot, max_length=250, num_return_sequences=1)[0]["generated_text"])

# Comparison
print("\n========== COMPARISON ==========")
print("Zero-shot: Gives instructions without examples.")
print("One-shot: Provides one example to guide the model.")
print("Few-shot: Provides multiple examples for better guidance.")