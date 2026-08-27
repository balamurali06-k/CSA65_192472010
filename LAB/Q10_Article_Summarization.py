# ============================================================
# QUESTION 10:
# Zero-shot, One-shot and Few-shot Article Summarization
# ============================================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained FLAN-T5 model
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Technical article
article = """
Artificial Intelligence is increasingly used in engineering.
Machine learning helps engineers analyze large amounts of data,
predict equipment failures and optimize designs. Computer vision
is used for inspection and quality control. Natural language
processing helps engineers process technical documents. Robotics
combines AI with sensors and control systems to automate complex
tasks. These technologies improve efficiency, accuracy and safety.
"""


# ------------------------------------------------------------
# FUNCTION TO GENERATE SUMMARY
# ------------------------------------------------------------

def generate_summary(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


# ------------------------------------------------------------
# ZERO-SHOT PROMPT
# ------------------------------------------------------------

zero_shot = f"""
Summarize the following technical article in about 50 words:

{article}
"""


# ------------------------------------------------------------
# ONE-SHOT PROMPT
# ------------------------------------------------------------

one_shot = f"""
Example:

Article: Solar energy converts sunlight into electricity and
provides a clean source of power for homes and industries.

Summary: Solar energy converts sunlight into clean electricity
and supports sustainable power generation.

Now summarize the following article in about 50 words:

{article}
"""


# ------------------------------------------------------------
# FEW-SHOT PROMPT
# ------------------------------------------------------------

few_shot = f"""
Example 1:

Article: Robots automate industrial tasks and improve production
efficiency.

Summary: Robots automate industrial tasks and improve production
efficiency and accuracy.

Example 2:

Article: Artificial Intelligence analyzes large amounts of data
to help engineers make better decisions.

Summary: AI analyzes engineering data and supports better
decision-making.

Now summarize the following article in about 50 words:

{article}
"""


# ------------------------------------------------------------
# GENERATE OUTPUTS
# ------------------------------------------------------------

print("========================================")
print("       ZERO-SHOT SUMMARY")
print("========================================")

print(generate_summary(zero_shot))


print("\n========================================")
print("        ONE-SHOT SUMMARY")
print("========================================")

print(generate_summary(one_shot))


print("\n========================================")
print("        FEW-SHOT SUMMARY")
print("========================================")

print(generate_summary(few_shot))


# ------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------

print("\n========================================")
print("             COMPARISON")
print("========================================")

print("Zero-shot : No example is provided.")
print("One-shot  : One example is provided.")
print("Few-shot  : Multiple examples are provided.")

print("\nEvaluation Criteria:")
print("1. Accuracy")
print("2. Completeness")
print("3. Readability")