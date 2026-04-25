# 📄 PDF Question Answering System (RAG-Based)

## 🚀 Overview

This project is an AI-powered **PDF Question Answering System** that allows users to upload a document and ask natural language questions.

It uses a **Retrieval-Augmented Generation (RAG)** pipeline to retrieve relevant information from the document and generate accurate answers using an LLM.

The goal of this project is to demonstrate practical skills in:

* Natural Language Processing (NLP)
* Information Retrieval
* Vector Search
* LLM Integration

---

## 🎯 Features

* 📂 Upload PDF documents
* ❓ Ask questions in natural language
* 🔍 Retrieves relevant content using semantic search
* 🤖 Generates accurate answers using LLM
* 📚 Displays source pages (citation support)
* 💬 Maintains chat history (memory)

---

## 🧠 How It Works (RAG Pipeline)

1. **PDF Parsing**

   * Extracts text from PDF using `pypdf`

2. **Text Chunking**

   * Splits large text into smaller overlapping chunks

3. **Embedding Generation**

   * Converts text chunks into vector representations using Sentence Transformers

4. **Vector Storage**

   * Stores embeddings in FAISS for efficient similarity search

5. **Query Processing**

   * Converts user query into embedding
   * Retrieves top-k similar chunks

6. **Answer Generation**

   * Sends retrieved context + question to LLM (Groq API)
   * Generates final answer

---

## 🏗️ Tech Stack

| Category       | Tools                |
| -------------- | -------------------- |
| Frontend/UI    | Streamlit            |
| Backend        | Python               |
| PDF Processing | pypdf                |
| Embeddings     | SentenceTransformers |
| Vector Search  | FAISS                |
| LLM API        | Groq                 |
| Deployment     | Streamlit Cloud      |

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/kashvi-b/pdf-qa.git
cd pdf-qa
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Setup API Key

Create file:

```text
.streamlit/secrets.toml
```

Add your API key:

```toml
GROQ_API_KEY = "your_api_key_here"
```

---

## ▶️ Run the App

```bash
streamlit run app_pro.py
```

---

## 🌐 Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to Streamlit Cloud
3. Select repository
4. Set main file:

   ```
   app_pro.py
   ```
5. Add API key in **Secrets**
6. Deploy 🚀

---

## 📁 Project Structure

```
pdf-qa/
│
├── app_pro.py
├── requirements.txt
├── runtime.txt
├── .gitignore
│
├── utils/
│   ├── chunker.py
│   ├── embedder.py
│   ├── vectorstore.py
│
└── data/ (optional)
```

---

## ⚠️ Limitations

* Does not support scanned PDFs (no OCR)
* Performance depends on document size
* Requires API key for LLM access

---

## 🔮 Future Improvements

* OCR support for image-based PDFs
* Chat-style conversational UI
* Persistent vector database
* Highlight answers within PDF

---

## 💡 Key Learnings

* Built a complete RAG pipeline from scratch
* Implemented semantic search using vector embeddings
* Integrated LLM APIs for real-world applications
* Handled deployment constraints and dependency issues

---

## 👤 Author

**Kashvi Bhardwaj**

---

## ⭐ If you like this project

Give it a star on GitHub!
