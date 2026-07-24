from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = input("Enter a sentence: ")

tokens = tokenizer.tokenize(text)

print("Tokens:")
print(tokens)