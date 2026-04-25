# utils/qa_chain.py
# This module takes a question + retrieved chunks
# and returns a grounded answer using a LOCAL LLM (Ollama)

import ollama


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """
    Builds the prompt for the local LLM.
    """

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant that answers questions about a document.

You will be given some excerpts from a PDF document, followed by a question.
Your job is to answer the question using ONLY the information in the excerpts.

Rules:
- If the answer is clearly present in the excerpts, answer it directly.
- If the answer is not in the excerpts, say: "I could not find this information in the provided document."
- Do not use any outside knowledge — only what is in the excerpts below.
- Keep your answer concise and clear.

--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Question: {question}

Answer (based only on the document):

    return prompt


def get_answer(question: str, context_chunks: list[str]) -> str:
    prompt = build_prompt(question, context_chunks)

    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']

    except Exception as e:
        return f"⚠️ Error: Make sure Ollama is running.\n\nDetails: {str(e)}"

def ask(
    question: str,
    index,
    chunks: list[str],
    top_k: int = 3
) -> dict:
    """
    Full pipeline:
    1. Embed question
    2. Search FAISS
    3. Send to local LLM
    4. Return answer + sources
    """

    from utils.embedder import embed_query
    from utils.vectorstore import search

    # Step 1: Embed the question
    query_vec = embed_query(question)

    # Step 2: Search FAISS
    results = search(query_vec, index, chunks, top_k=top_k)

    # Step 3: Extract text chunks
    context_chunks = [r["text"] for r in results]

    print(f"Sending question to LOCAL model with {len(context_chunks)} context chunks...")

    # Step 4: Get answer from Ollama
    answer = get_answer(question, context_chunks)

    return {
        "answer": answer,
        "sources": results
    }