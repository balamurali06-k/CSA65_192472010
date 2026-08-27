# ============================================================
# QUESTION 8:
# AI Chatbot for College Student Queries
# ============================================================

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
import torch

# Load pre-trained Question Answering model
model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# College knowledge base
college_info = """
Admissions are conducted through the official college admission
process. Students should contact the admission office for
admission details.

Examinations are conducted according to the academic calendar.
Students should check the examination timetable regularly.

Students must maintain the required attendance percentage
according to college regulations.

The college has Computer Science, Electronics, Mechanical and
other engineering departments.

Campus facilities include a library, computer laboratories,
classrooms, a canteen, sports facilities and transportation.
"""

print("========================================")
print("        COLLEGE AI ASSISTANT")
print("========================================")
print("Ask questions about:")
print("Admissions")
print("Examinations")
print("Attendance")
print("Departments")
print("Campus Facilities")
print("Type 'exit' to stop.")
print("========================================")


while True:

    question = input("\nStudent: ")

    # Exit chatbot
    if question.lower() == "exit":
        print("Bot: Thank you! Goodbye.")
        break

    # Tokenize question and context
    inputs = tokenizer(
        question,
        college_info,
        return_tensors="pt"
    )

    # Generate model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Find answer start and end positions
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)

    # Make sure end position is valid
    if end_index < start_index:
        end_index = start_index

    # Extract answer tokens
    answer_tokens = inputs["input_ids"][0][
        start_index:end_index + 1
    ]

    # Convert tokens to text
    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    )

    print("Bot:", answer)