from transformers import BertTokenizer, GPT2Tokenizer

text = input("Enter a sentence: ")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

bert_tokens = bert_tokenizer.tokenize(text)
gpt2_tokens = gpt2_tokenizer.tokenize(text)

print("\nBERT Tokens:")
print(bert_tokens)

print("\nGPT-2 Tokens:")
print(gpt2_tokens)