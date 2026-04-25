# test_chunker.py  (delete after testing)

from utils.pdf_loader import load_pdf
from utils.chunker import split_text, preview_chunks

# Step 1: load the PDF text (from our last step)
print("Loading PDF...")
text = load_pdf("data/your_file.pdf")   # <-- your filename here

# Step 2: split it into chunks
print("\nSplitting into chunks...")
chunks = split_text(text, chunk_size=500, chunk_overlap=50)

# Step 3: preview a few chunks so you can see what they look like
preview_chunks(chunks, n=3)

# Step 4: basic stats
print(f"\nTotal chunks: {len(chunks)}")
print(f"Avg chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")