# extraction/pdf_loader.py

import fitz  # PyMuPDF
from pathlib import Path


def load_pdf(pdf_path: str) -> str:
    """
    Load PDF from path, extract all text.
    Returns clean string. Raises if file missing or unreadable.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    doc = fitz.open(str(path))

    if doc.page_count == 0:
        raise ValueError(f"PDF has no pages: {pdf_path}")

    pages_text = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction
        cleaned = _clean_text(text)
        if cleaned:
            pages_text.append(cleaned)

    doc.close()

    if not pages_text:
        raise ValueError(f"No extractable text found in: {pdf_path}")

    full_text = "\n\n".join(pages_text)
    return full_text


def _clean_text(text: str) -> str:
    """
    Remove excessive whitespace, empty lines.
    Keep medical content intact.
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped:  # drop empty lines
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def get_pdf_metadata(pdf_path: str) -> dict:
    """
    Optional: extract PDF metadata.
    Useful for logging which file was processed.
    """
    doc = fitz.open(pdf_path)
    meta = {
        "page_count": doc.page_count,
        "file_name": Path(pdf_path).name,
        "file_size_kb": round(Path(pdf_path).stat().st_size / 1024, 2)
    }
    doc.close()
    return meta