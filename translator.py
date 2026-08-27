from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

error_message = input("Enter an English error message: ")

prompt = f"""
Translate the following English error message into Telugu.

Keep the technical meaning accurate and easy to understand.
Return only the Telugu translation.

Error message:
{error_message}
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

print("\nEnglish Error:")
print(error_message)

print("\nTelugu Translation:")
print(response.text)