from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model ONLY once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("rag/student_handbook.index")

# Load text chunks
with open("rag/chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f.readlines()]


def normalize_query(query: str) -> str:
    """
    Normalize user query to match PDF language
    """
    q = query.lower()

    replacements = {
        "email id": "email",
        "mail id": "email",
        "mail": "email",
        "e-mail": "email",
        "placement team": "placement",
        "placement office": "placement",
    }

    for k, v in replacements.items():
        q = q.replace(k, v)

    return q


def retrieve(query: str, top_k: int = 10, window: int = 3) -> str:
    """
    Retrieve relevant context using FAISS + windowing
    """
    query = normalize_query(query)

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    collected = []
    seen = set()

    for idx in indices[0]:
        for i in range(idx - window, idx + window + 1):
            if 0 <= i < len(chunks):
                text = chunks[i]

                # filter noise
                if len(text) < 15:
                    continue

                if text not in seen:
                    seen.add(text)
                    collected.append(text)

    # join and LIMIT size (important for speed + accuracy)
    context = "\n".join(collected)
    return context[:3000]   # hard limit
