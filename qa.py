from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Load syllabus
with open("syllabus.txt", "r", encoding="utf-8") as file:
    syllabus = file.read()

question = input("Ask a question about the syllabus: ")

prompt = f"""
Answer the question ONLY using the syllabus below.

If the answer is not present in the syllabus, say:
"Answer not found in the syllabus."

SYLLABUS:
{syllabus}

QUESTION:
{question}
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(response.text)