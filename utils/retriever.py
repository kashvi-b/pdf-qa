from sklearn.metrics.pairwise import cosine_similarity

def retrieve_top_chunks(query_vec, chunk_vecs, chunks, top_k=3):
    similarities = cosine_similarity(query_vec, chunk_vecs)[0]

    # get top indices
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [chunks[i] for i in top_indices]