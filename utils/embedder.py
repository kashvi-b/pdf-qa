from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

def embed_chunks(chunks):
    embeddings = vectorizer.fit_transform(chunks)
    return embeddings

def embed_query(query):
    return vectorizer.transform([query])