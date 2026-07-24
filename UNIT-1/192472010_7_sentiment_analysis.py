from transformers import pipeline

classifier = pipeline("sentiment-analysis")

text = input("Enter a sentence: ")

result = classifier(text)

print(result)