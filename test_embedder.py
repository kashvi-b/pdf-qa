# test_embedder.py  (delete after testing)

from utils.pdf_loader import load_pdf
from utils.chunker import split_text
from utils.embedder import embed_chunks, embed_query

# --- Load and chunk the PDF ---
print("Loading PDF and chunking...")
text = load_pdf("data/your_file.pdf")   # <-- your filename
chunks = split_text(text)

# --- Embed all chunks ---
print("\nEmbedding chunks...")
embeddings = embed_chunks(chunks)

# --- Check the shape ---
print(f"\nNumber of chunks:     {len(chunks)}")
print(f"Embeddings shape:     {embeddings.shape}")
print(f"One vector (first 8 numbers): {embeddings[0][:8]}")

# --- Embed a test question ---
print("\nEmbedding a test question...")
query_vec = embed_query("What is this document about?")
print(f"Query vector shape:   {query_vec.shape}")
print(f"Query vector (first 8 numbers): {query_vec[0][:8]}")

# --- Quick sanity check: similar texts should be close ---
import numpy as np

v1 = embed_query("The cat sat on the mat")
v2 = embed_query("A kitten rested on a rug")
v3 = embed_query("Quantum physics equations")

# Cosine similarity: 1.0 = identical, 0.0 = unrelated, -1.0 = opposite
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("\n--- Similarity sanity check ---")
print(f"cat/mat vs kitten/rug:     {cosine_similarity(v1[0], v2[0]):.3f}  (should be HIGH)")
print(f"cat/mat vs quantum physics:{cosine_similarity(v1[0], v3[0]):.3f}  (should be LOW)")