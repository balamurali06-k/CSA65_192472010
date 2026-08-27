#Exercise-1
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

prompt = """
Write a poem about my college, Saveetha School of Engineering.
The poem must contain exactly 4 lines.
"""

temperatures = [0, 0.5, 1]

for temperature in temperatures:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )

    print("\n" + "=" * 50)
    print(f"TEMPERATURE: {temperature}")
    print("=" * 50)
    print(response.text)