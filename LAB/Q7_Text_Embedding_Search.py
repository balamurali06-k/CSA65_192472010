# ============================================================
# QUESTION 7:
# Semantic Similarity Search using Text Embeddings
# ============================================================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Technical documents
documents = [
    "Machine learning algorithms are used to make predictions.",
    "Python is widely used for software development.",
    "Computer networks connect devices and exchange information.",
    "SQL is used for storing and retrieving database information.",
    "Robotics combines sensors, control systems and artificial intelligence."
]

# User query
query = "How are machines trained to make predictions?"

# Generate embeddings for documents
document_embeddings = model.encode(documents)

# Generate embedding for query
query_embedding = model.encode([query])

# Calculate cosine similarity
similarity_scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

# Sort documents from highest to lowest similarity
ranked_indices = similarity_scores.argsort()[::-1]

# Display results
print("========================================")
print("       SEMANTIC SIMILARITY SEARCH")
print("========================================")

print("\nQuery:")
print(query)

print("\nMost Relevant Documents:")

for index in ranked_indices:
    print(
        "\nSimilarity Score:",
        round(similarity_scores[index], 4)
    )
    print("Document:", documents[index])