import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# chunks loading
chunks_path = os.path.join("..", "chunks", "document_chunks.txt")
with open(chunks_path, "r", encoding="utf-8") as f:
    raw_chunks = f.read().split("### Chunk")

# Clean and filter chunks
chunks = []
for chunk in raw_chunks:
    content = chunk.strip().split("###")[-1].strip()
    if len(content.split()) > 10:  
        chunks.append(content)

print(f"Loaded {len(chunks)} valid chunks.")

# Loading embedding model
model = SentenceTransformer("BAAI/bge-small-en")
print("Loaded embedding model: bge-small-en")

# Create embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors.")

vectordb_path = os.path.join("..", "vectordb")
os.makedirs(vectordb_path, exist_ok=True)

faiss.write_index(index, os.path.join(vectordb_path, "faiss_index.idx"))
with open(os.path.join(vectordb_path, "chunks.pkl"), "wb") as f:
    pickle.dump(chunks, f)

print("Saved FAISS index and chunks metadata to 'vectordb/'")