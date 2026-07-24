from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentence1 = input("Enter first sentence: ")
sentence2 = input("Enter second sentence: ")

inputs1 = tokenizer(sentence1, return_tensors="pt")
inputs2 = tokenizer(sentence2, return_tensors="pt")

embedding1 = model(**inputs1).last_hidden_state
embedding2 = model(**inputs2).last_hidden_state

print("\nEmbedding Shape for Sentence 1:")
print(embedding1.shape)

print("\nEmbedding Shape for Sentence 2:")
print(embedding2.shape)