import streamlit as st
from pypdf import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PDF QA (RAG)", layout="wide")
MODEL = "llama-3.1-8b-instant"

st.title("📄 PDF QA (Enhanced RAG)")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# ---------------- SESSION MEMORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CHUNKING ----------------
def split_text(text, page_num, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append({
            "text": text[start:end],
            "page": page_num
        })

        start += chunk_size - overlap

    return chunks

# ---------------- BUILD INDEX ----------------
def build_index(chunks):
    texts = [c["text"] for c in chunks]

    embeddings = embed_model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings

# ---------------- SEARCH ----------------
def search(query, chunks, index, k=3):
    query_embedding = embed_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = [chunks[i] for i in indices[0]]
    return results

# ---------------- PDF UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []

    # Extract text + chunk with page tracking
    for file in uploaded_files:
        reader = PdfReader(file)

        for i, page in enumerate(reader.pages):
            text = page.extract_text()

            if text:
                chunks = split_text(text, page_num=i+1)
                all_chunks.extend(chunks)

    if not all_chunks:
        st.error("❌ No text extracted (PDF may be scanned)")
        st.stop()

    st.success("✅ PDFs processed")

    # Build FAISS index
    index, embeddings = build_index(all_chunks)

    # ---------------- USER QUESTION ----------------
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Thinking..."):

            # Retrieve top chunks
            top_chunks = search(question, all_chunks, index, k=3)

            context = "\n\n".join([c["text"] for c in top_chunks])

            # Add chat history
            history_text = ""
            for q, a in st.session_state.history[-3:]:
                history_text += f"Q: {q}\nA: {a}\n"

            prompt = f"""
Use the context below and previous conversation to answer.

Chat History:
{history_text}

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

                        # Save to memory
                        st.session_state.history.append((question, answer))

                        # ---------------- DISPLAY ----------------
                        st.success("✅ Answer:")
                        st.write(answer)

                        # Show sources
                        st.markdown("### 📚 Sources:")
                        pages = set([c["page"] for c in top_chunks])

                        for p in pages:
                            st.write(f"Page {p}")

                    else:
                        st.error("❌ Unexpected API response")
                        st.write(result)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ---------------- SHOW CHAT HISTORY ----------------
if st.session_state.history:
    st.markdown("## 💬 Chat History")

    for q, a in reversed(st.session_state.history):
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {a}")
        st.write("---")