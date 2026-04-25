import streamlit as st
from pypdf import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PDF QA (RAG)", layout="wide")

MODEL = "llama-3.1-8b-instant"

st.title("📄 PDF QA (Clean RAG Version)")

# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# ---------------- TEXT CHUNKING ----------------
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# ---------------- VECTOR STORE ----------------
def build_index(chunks):
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings

# ---------------- SEARCH ----------------
def search(query, chunks, index, k=3):
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    results = [chunks[i] for i in indices[0]]
    return "\n\n".join(results)

# ---------------- PDF UPLOAD ----------------
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:

    all_text = ""

    for file in uploaded_files:
        reader = PdfReader(file)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

    if not all_text.strip():
        st.error("❌ No text extracted (PDF may be scanned)")
        st.stop()

    st.success("✅ PDFs processed")

    # ---------------- CHUNK + INDEX ----------------
    chunks = split_text(all_text)
    index, embeddings = build_index(chunks)

    # ---------------- QUESTION ----------------
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Thinking..."):

            # Retrieve relevant context
            context = search(question, chunks, index)

            prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

            try:
                headers = {
                    "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": MODEL,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }

                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data
                )

                if response.status_code != 200:
                    st.error(f"❌ HTTP Error {response.status_code}")
                    st.write(response.text)
                else:
                    result = response.json()

                    if "choices" in result:
                        answer = result["choices"][0]["message"]["content"]
                        st.success("✅ Answer:")
                        st.write(answer)
                    else:
                        st.error("❌ Unexpected API response")
                        st.write(result)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

                        
