# test_qa.py  (delete after testing)

from utils.vectorstore import load_index
from utils.qa_chain import ask

# Load the FAISS index we built in Step 5
print("Loading index...")
index, chunks = load_index(save_dir="vectorstore")

# Ask a question that is answered somewhere in your PDF
question = "what are the chances of my train getting delayed?"   # <-- change to match your PDF

print(f"\nQuestion: {question}\n")
result = ask(question, index, chunks, top_k=3)

# Print the answer
print("=" * 50)
print("ANSWER:")
print(result["answer"])

# Print the source chunks Claude used
print("\n" + "=" * 50)
print("SOURCE CHUNKS USED:")
for source in result["sources"]:
    print(f"\nRank {source['rank']}  |  Score: {source['score']}")
    print(source["text"][:200])
    print("-" * 40)