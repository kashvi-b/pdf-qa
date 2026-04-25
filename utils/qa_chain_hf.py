import requests

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    return f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""


def get_answer(question: str, context_chunks: list[str]) -> str:
    prompt = build_prompt(question, context_chunks)

    response = requests.post(
        HF_API_URL,
        json={"inputs": prompt},
    )

    try:
        result = response.json()

        if isinstance(result, list):
            return result[0]["generated_text"]
        else:
            return str(result)

    except Exception as e:
        return f"Error: {str(e)}"


def ask(question, index, chunks, top_k=3):
    from utils.embedder import embed_query
    from utils.vectorstore import search

    query_vec = embed_query(question)
    results = search(query_vec, index, chunks, top_k=top_k)

    context_chunks = [r["text"] for r in results]

    return {
        "answer": get_answer(question, context_chunks),
        "sources": results
    }