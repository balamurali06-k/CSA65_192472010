import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------------
# 1. Engineering Library Documents
# -----------------------------------------
documents = [
    "Data Structures is a fundamental computer science subject covering arrays, linked lists, stacks, queues, trees and graphs.",
    "Machine Learning teaches algorithms such as linear regression, decision trees, support vector machines and neural networks.",
    "Database Management Systems explains relational databases, SQL, normalization, transactions and database design.",
    "Computer Networks covers network protocols, TCP/IP, routing, switching, network security and wireless communication.",
    "Operating Systems explains process management, memory management, file systems, scheduling and synchronization.",
    "Artificial Intelligence introduces search algorithms, knowledge representation, machine learning and intelligent agents.",
    "Software Engineering focuses on software development life cycle, requirements analysis, testing and project management.",
    "Cloud Computing covers virtualization, cloud services, distributed systems and scalable computing resources."
]

# -----------------------------------------
# 2. Load Sentence Transformer Model
# -----------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------
# 3. Convert Documents into Vectors
# -----------------------------------------
embeddings = model.encode(documents)

# Convert to NumPy float32 format
embeddings = np.array(embeddings).astype("float32")

# -----------------------------------------
# 4. Create FAISS Vector Database
# -----------------------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add document vectors to FAISS
index.add(embeddings)

print("Number of documents stored:", index.ntotal)

# -----------------------------------------
# 5. Student Query
# -----------------------------------------
query = input("\nEnter your library query: ")

# Convert query into vector
query_vector = model.encode([query])
query_vector = np.array(query_vector).astype("float32")

# -----------------------------------------
# 6. Similarity Search
# -----------------------------------------
k = 3

distances, indices = index.search(query_vector, k)

# -----------------------------------------
# 7. Display Results
# -----------------------------------------
print("\nMost Relevant Documents:")
print("-----------------------------------------")

for i in range(k):
    doc_index = indices[0][i]

    print("\nRank:", i + 1)
    print("Document:", documents[doc_index])
    print("Distance:", distances[0][i])