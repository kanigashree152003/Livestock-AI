import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# LOAD PROCESSED DATA
# -------------------------------
with open("rag/processed_documents.json", "r") as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} documents")

# -------------------------------
# EMBEDDING MODEL
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# CREATE EMBEDDINGS
# -------------------------------
embeddings = model.encode(documents)

embeddings = np.array(embeddings).astype("float32")

# -------------------------------
# NORMALIZE (IMPORTANT)
# -------------------------------
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# cosine similarity
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS index created")

# -------------------------------
# SAVE
# -------------------------------
faiss.write_index(index, "rag/faiss.index")

with open("rag/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ FAISS index saved successfully")