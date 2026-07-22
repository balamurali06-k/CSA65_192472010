import nltk
from nltk.tokenize 
import     sent_tokenize

nltk.download('punkt')

text = "tokenization is very intresting topic we can learn in gen-ai"

sentence = sent_tokenize(text)

print("sentence tokens:")
for sentence in sentences:
    print(sentence) 