# test_loader.py  (you can delete this after testing)

from utils.pdf_loader import load_pdf, load_pdf_by_page

# --- Test 1: load the whole PDF as one string ---
print("=== Test 1: Full text ===")
text = load_pdf("data/your_file.pdf")   # <-- change this to your filename
print(text[:500])                        # Print just the first 500 characters

# --- Test 2: load page by page ---
print("\n=== Test 2: Page by page ===")
pages = load_pdf_by_page("data/your_file.pdf")  # <-- same filename
for p in pages[:2]:   # Show just first 2 pages
    print(f"\n--- Page {p['page']} ---")
    print(p["text"][:200])   # First 200 chars of each page