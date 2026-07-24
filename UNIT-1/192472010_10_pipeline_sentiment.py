from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

sentence = input("Enter a sentence: ")

result = sentiment(sentence)

print("\nSentiment Analysis Result:")
print(result)