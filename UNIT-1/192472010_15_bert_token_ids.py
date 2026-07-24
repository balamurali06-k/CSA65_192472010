from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentence = input("Enter a sentence: ")

tokens = tokenizer.tokenize(sentence)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)