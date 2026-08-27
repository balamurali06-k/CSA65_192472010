# ============================================================
# QUESTION 6:
# Zero-shot, One-shot, and Few-shot Prompting
# for Smart Fitness Watch Product Description
# ============================================================

from transformers import pipeline

# Load pre-trained text generation model
generator = pipeline("text-generation", model="gpt2")

# ------------------------------------------------------------
# ZERO-SHOT PROMPT
# ------------------------------------------------------------

zero_shot = """
Write an attractive product description for a Smart Fitness
Watch designed for engineering college students.
"""

# ------------------------------------------------------------
# ONE-SHOT PROMPT
# ------------------------------------------------------------

one_shot = """
Example:

Product: Smart Study Lamp
Description: A modern study lamp designed for students with
adjustable brightness and a comfortable design.

Now write an attractive product description for a Smart Fitness
Watch designed for engineering college students.
"""

# ------------------------------------------------------------
# FEW-SHOT PROMPT
# ------------------------------------------------------------

few_shot = """
Example 1:

Product: Smart Study Lamp
Description: A modern lamp that helps students study comfortably
with adjustable brightness.

Example 2:

Product: Smart Backpack
Description: A stylish backpack designed for students with useful
features for everyday college activities.

Now write an attractive product description for a Smart Fitness
Watch designed for engineering college students.
"""

# ------------------------------------------------------------
# GENERATE OUTPUTS
# ------------------------------------------------------------

print("\n========================================")
print("          ZERO-SHOT OUTPUT")
print("========================================")

result = generator(
    zero_shot,
    max_new_tokens=100,
    num_return_sequences=1
)

print(result[0]["generated_text"])


print("\n========================================")
print("           ONE-SHOT OUTPUT")
print("========================================")

result = generator(
    one_shot,
    max_new_tokens=100,
    num_return_sequences=1
)

print(result[0]["generated_text"])


print("\n========================================")
print("           FEW-SHOT OUTPUT")
print("========================================")

result = generator(
    few_shot,
    max_new_tokens=100,
    num_return_sequences=1
)

print(result[0]["generated_text"])


# ------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------

print("\n========================================")
print("              COMPARISON")
print("========================================")

print("Zero-shot : No example is provided.")
print("One-shot  : One example is provided.")
print("Few-shot  : Multiple examples are provided.")