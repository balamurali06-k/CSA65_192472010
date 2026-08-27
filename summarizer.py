from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

abstract = """
This paper presents a machine learning-based system for detecting
cybersecurity threats in network traffic. The proposed approach uses
supervised learning algorithms to classify network activities as
normal or malicious. Experimental results on a benchmark dataset
show that the proposed model achieves high classification accuracy
while reducing false positives. The system can support real-time
network monitoring and improve the early detection of cyber attacks.
"""

prompt = f"""
Summarize the following IEEE paper abstract into ONE tweet.

Requirements:
- Maximum 280 characters.
- Include the main problem, proposed approach, and key result.
- Do not use hashtags.
- Return only the summary.

Abstract:
{abstract}
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

summary = response.text.strip()

print("=" * 60)
print("IEEE ABSTRACT → ONE TWEET")
print("=" * 60)
print(summary)
print("=" * 60)
print("Character count:", len(summary))
print("Within 280 characters:", len(summary) <= 280)