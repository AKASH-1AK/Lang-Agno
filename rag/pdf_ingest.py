from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("Starting PDF ingestion...")

# 1. Load PDF
reader = PdfReader("data/student_handbook.pdf")
print("PDF loaded")

text_chunks = []

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        chunks = text.split("\n")
        text_chunks.extend(chunks)
    print(f"Page {i+1} processed")

print(f"Total text chunks: {len(text_chunks)}")

# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded")

# 3. Create embeddings
embeddings = model.encode(text_chunks)
print("Embeddings created")

# 4. Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# 5. Save index and chunks
faiss.write_index(index, "rag/student_handbook.index")

with open("rag/chunks.txt", "w", encoding="utf-8") as f:
    for chunk in text_chunks:
        f.write(chunk.replace("\n", " ") + "\n")

print("✅ PDF ingestion completed successfully")
