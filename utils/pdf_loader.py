# utils/pdf_loader.py
import fitz  # PyMuPDF


def load_pdf(pdf_path: str) -> str:
    """Return full text of a PDF as one string."""
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def load_pdf_by_page(pdf_path: str) -> list[dict]:
    """
    Return a list of dicts — one per page — each carrying:
      page   : 1-based page number
      text   : extracted text
      source : the file path (used as citation source)
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        pages.append({
            "page":   i + 1,
            "text":   page.get_text(),
            "source": pdf_path,        # ← NEW: tracks origin file
        })
    doc.close()
    return pages