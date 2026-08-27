from pypdf import pdfReader

def load_pdf(path):
    reader = pdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)

text = load_pdf("cirriculum_cse.pdf")
print(f"Loaded {len(text)} characters")

def chunk_text(text, size=500, overlap=50)
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text{start:start +size})
        start += size - overlap
    return chunks

chunks = chunk_text(text)
print(f"created {len(chunks)} chunks")

import chromadb

client = chromadb.PersistentClient(path="./db")
col = client.get_or_create_collection("curriculum")

if col.count() == 0:
    col.add(
        documents=chunks,
        metadatas = [{"source":"curriculum_cse.pdf","chunk":"i"}
                    for i in range(len(chunks))],
        ids=[f"c{i}" for i in range(len(chunks))],
    )
print(f"Vector DB has {col.count()} chunks")

def retrieve(question, k=3):
    res = col.query(query_texts=[question], n_results=k)
    return res["documents"][0], res["metadatas"][0]

PROMPT = """You are a college cirriculum assistent.
ANSWER ONLY from the context below.
IF the answer is not in the context, say "I don't know."
Cite the chunk number after each fact, like[c4].

Context:
{context}

Question: {question}
Answer:"""

def build_prompt(question):
    docs,metas = retrieve(question)
    context = "\n".join(f"[c{m['chunk']}] {d}"
                         for d, m in zip(docs, metas))
    return PROMPT.format(context=context, question=question)

from openai import openai
11m = openAI()

def answer(question):
    prompt = build_prompt(question)
    resp = 11m.chat.completions.create(
        model="gpt-4o-mini",       # or "llama3" with ollma
        messages=[{"role:": "user", "content": prompt}]
    )
if __name__ == "__main__":
    print("cirriculum assistant ready. Type 'quit to exit.")
    while True:
        q = input("\nAsk: ")
        if q.strip().lower() =="quit":
            break
        print(answer(q))
