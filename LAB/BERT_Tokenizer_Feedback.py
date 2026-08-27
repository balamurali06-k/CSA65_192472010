from transformers import BertTokenizer

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Set of student feedback sentences
feedback = [
    "The course was very interesting and useful.",
    "The lectures were boring and difficult to understand.",
    "I really enjoyed learning this subject.",
    "The assignments were too difficult."
]

# Tokenize each feedback sentence
for sentence in feedback:
    print("\nFeedback:", sentence)

    # Convert sentence into tokens
    tokens = tokenizer.tokenize(sentence)

    # Convert tokens into token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("Tokens:", tokens)
    print("Token IDs:", token_ids)