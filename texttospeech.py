from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import wave

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def save_wav(filename, pcm_data):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_data)


text = input("Enter text to convert to speech: ")

response = client.models.generate_content(
    model="gemini-3.1-flash-tts-preview",
    contents=text,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore"
                )
            )
        )
    )
)

audio_data = response.candidates[0].content.parts[0].inline_data.data

save_wav("output.wav", audio_data)

print("\nText-to-Speech completed!")
print("Audio saved as: output.wav")