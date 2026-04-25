# utils/vectorstore.py
import faiss
import numpy as np
import pickle
import os


def build_and_save(
    chunks:     list[str],
    embeddings: np.ndarray,
    metadatas:  list[dict],          # ← NEW: one dict per chunk
    save_dir:   str = "vectorstore",
):
    """
    Build a FAISS index and save it together with chunk texts
    and their metadata (source filename + page number).
    """
    embeddings = np.array(embeddings).astype("float32")
    dimension  = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(save_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))

    with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks, "metadatas": metadatas}, f)

    print(f"Saved {index.ntotal} vectors + metadata to {save_dir}/")


def load_index(save_dir: str = "vectorstore"):
    """
    Load FAISS index, chunk texts, and metadata from disk.
    Returns: (index, chunks, metadatas)
    """
    index = faiss.read_index(os.path.join(save_dir, "index.faiss"))

    with open(os.path.join(save_dir, "chunks.pkl"), "rb") as f:
        data = pickle.load(f)

    # Support old format (list) and new format (dict)
    if isinstance(data, dict):
        chunks    = data["chunks"]
        metadatas = data["metadatas"]
    else:
        chunks    = data
        metadatas = [{}] * len(chunks)   # empty metadata for old indexes

    print(f"Loaded {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks, metadatas


def search(
    query_embedding: np.ndarray,
    index,
    chunks:    list[str],
    metadatas: list[dict],
    top_k:     int = 3,
) -> list[dict]:
    """
    Search the FAISS index and return top_k results,
    each with its text, score, and metadata.
    """
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            "rank":     rank + 1,
            "score":    round(float(dist), 4),
            "text":     chunks[idx],
            "metadata": metadatas[idx],   # ← NEW: source + page attached
        })
    return results