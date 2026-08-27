from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

prompt = input("Describe the image you want to generate: ")

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=prompt
)

for part in response.candidates[0].content.parts:
    if part.inline_data:
        image = part.as_image()
        image.save("generated_image.png")
        print("\nImage generated successfully!")
        print("Saved as: generated_image.png")