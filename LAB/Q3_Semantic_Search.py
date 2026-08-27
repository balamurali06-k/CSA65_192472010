# ============================================================
# Question 3: Semantic Search using Cosine Similarity
# ============================================================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Engineering documents
documents = [
    "Python programming and object oriented programming concepts.",
    "Computer networks and communication protocols.",
    "Database management systems and SQL queries.",
    "Artificial intelligence and machine learning algorithms.",
    "Operating systems and process management."
]

# User query
query = "How can I learn about SQL and databases?"

# Generate document embeddings
document_embeddings = model.encode(documents)

# Generate query embedding
query_embedding = model.encode([query])

# Calculate cosine similarity
similarities = cosine_similarity(query_embedding, document_embeddings)[0]

# Display results
print("Query:", query)
print("\nSemantic Search Results:")

for i in similarities.argsort()[::-1]:
    print(f"{similarities[i]:.4f} -> {documents[i]}")