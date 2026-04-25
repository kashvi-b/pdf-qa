# test_vectorstore.py  (delete after testing)

from utils.pdf_loader import load_pdf
from utils.chunker import split_text
from utils.embedder import embed_chunks, embed_query
from utils.vectorstore import build_and_save, load_index, search

# --- Build the index from scratch ---
print("=== Building index ===")
text       = load_pdf("data/your_file.pdf")   # <-- your filename
chunks     = split_text(text)
embeddings = embed_chunks(chunks)

build_and_save(chunks, embeddings, save_dir="vectorstore")

# --- Load it back (simulates a fresh restart) ---
print("\n=== Loading index from disk ===")
index, loaded_chunks = load_index(save_dir="vectorstore")

# --- Search with a question ---
print("\n=== Searching ===")
question    = "What is the purpose of this document?"  # <-- change to match your PDF
query_vec   = embed_query(question)
results     = search(query_vec, index, loaded_chunks, top_k=3)

print(f"\nQuestion: {question}\n")
for r in results:
    print(f"Rank {r['rank']}  |  Score: {r['score']}  (lower = better)")
    print(r["text"][:300])
    print("-" * 50)