from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

prompt = input("Describe the program you want: ")

instruction = f"""
Generate a simple Python program for the following requirement:

{prompt}

Requirements:
- Return only the Python code.
- Keep the code beginner-friendly.
- Include comments explaining the main steps.
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=instruction
)

print("\nGenerated Python Code:")
print("=" * 50)
print(response.text)