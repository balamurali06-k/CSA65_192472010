from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentence = input("Enter a sentence: ")

inputs = tokenizer(sentence, return_tensors="pt")

outputs = model(**inputs)

print("\nContextual Embeddings:")
print(outputs.last_hidden_state)