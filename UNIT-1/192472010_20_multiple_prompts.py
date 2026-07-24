from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompts = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning"
]

for prompt in prompts:
    print("\nPrompt:", prompt)

    result = generator(
        prompt,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7
    )

    print("Response:")
    print(result[0]["generated_text"])