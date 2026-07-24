from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentence = input("Enter a sentence: ")

tokens = tokenizer.tokenize(sentence)

print("\nTokens:")
print(tokens)