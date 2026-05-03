from sentence_transformers import SentenceTransformer
import faiss
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("rag/faiss.index")

with open("rag/documents.pkl", "rb") as f:
    documents = pickle.load(f)

query = "goat has itchy patches near eyes"

q_vec = model.encode([query])
faiss.normalize_L2(q_vec)

D, I = index.search(q_vec, 3)

# ✅ print top 3 results
for i in range(3):
    print("\n======================")
    print(f"Match {i+1}")
    print("----------------------")
    print(documents[I[0][i]])
    print("Score:", D[0][i])