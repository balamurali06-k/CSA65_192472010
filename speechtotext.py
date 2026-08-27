from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Read the audio file
with open("output.wav", "rb") as audio_file:
    audio_data = audio_file.read()

prompt = """
Transcribe the speech in this audio file accurately.
Return only the transcribed text.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        prompt,
        types.Part.from_bytes(
            data=audio_data,
            mime_type="audio/wav"
        )
    ]
)

print("\nSpeech-to-Text Result:")
print("=" * 50)
print(response.text)