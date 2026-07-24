from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = input("Enter a prompt: ")

result = generator(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)

print("\nGenerated Text:")
print(result[0]["generated_text"])