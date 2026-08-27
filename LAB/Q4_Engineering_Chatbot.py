# ============================================================
# QUESTION 4:
# Engineering Support Chatbot using NLP
# ============================================================

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load pre-trained BERT-based Question Answering model
model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Engineering knowledge base
context = """
Python is a programming language used for software development.
SQL is a language used to communicate with relational databases.
Computer networks allow computers to communicate with each other.
An IP address identifies a device on a computer network.
A database is used to store and manage structured information.
HTML is used to create the structure of web pages.
"""

print("========================================")
print("      ENGINEERING SUPPORT CHATBOT")
print("========================================")
print("Ask questions about programming,")
print("networking, databases and engineering.")
print("Type 'exit' to stop.")
print("========================================")


while True:

    question = input("\nStudent: ")

    if question.lower() == "exit":
        print("Bot: Thank you! Goodbye.")
        break

    # Tokenize question and context
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt"
    )

    # Generate model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Find start and end positions
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)

    # Make sure end comes after start
    if end_index < start_index:
        end_index = start_index

    # Extract answer tokens
    answer_tokens = inputs["input_ids"][0][
        start_index:end_index + 1
    ]

    # Convert tokens back to text
    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    )

    print("Bot:", answer)