from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

context = """
Artificial Intelligence is the simulation of human intelligence by machines.
It enables machines to learn, reason, and make decisions.
"""

question = input("Enter your question: ")

result = qa_pipeline(question=question, context=context)

print("\nAnswer:")
print(result["answer"])