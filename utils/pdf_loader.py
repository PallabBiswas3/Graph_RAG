"""
pdf_loader.py — Load real PDF documents into the GraphRAG pipeline.
Uses PyMuPDF (fitz) if available, falls back to pdfplumber, then pypdf.
Preserves heading structure for DKG chapter detection.
"""

from __future__ import annotations
import re
import os
from typing import List, Dict, Optional


def load_pdf(filepath: str) -> str:
    """
    Extract text from a PDF file, preserving heading structure.
    Returns markdown-style text with ## headings detected from font size.
    """
    # Try PyMuPDF first (best quality)
    try:
        import fitz  # PyMuPDF
        return _load_with_pymupdf(filepath)
    except ImportError:
        pass

    # Try pdfplumber
    try:
        import pdfplumber
        return _load_with_pdfplumber(filepath)
    except ImportError:
        pass

    # Try pypdf
    try:
        from pypdf import PdfReader
        return _load_with_pypdf(filepath)
    except ImportError:
        pass

    raise ImportError(
        "No PDF library found. Install one of:\n"
        "  pip install pymupdf\n"
        "  pip install pdfplumber\n"
        "  pip install pypdf"
    )


def _load_with_pymupdf(filepath: str) -> str:
    """Extract text with font-size-based heading detection."""
    import fitz
    doc = fitz.open(filepath)
    lines = []
    font_sizes = []

    # First pass: collect all font sizes to determine heading thresholds
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 12)
                        font_sizes.append(size)

    if not font_sizes:
        # Fallback: plain text extraction
        return "\n".join(page.get_text() for page in doc)

    avg_size = sum(font_sizes) / len(font_sizes)
    heading1_threshold = avg_size * 1.4
    heading2_threshold = avg_size * 1.2

    # Second pass: extract with heading markers
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    text = " ".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()
                    if not text:
                        continue

                    max_size = max(
                        (span.get("size", 12) for span in line.get("spans", [])),
                        default=12
                    )

                    if max_size >= heading1_threshold and len(text) < 100:
                        lines.append(f"\n# {text}")
                    elif max_size >= heading2_threshold and len(text) < 100:
                        lines.append(f"\n## {text}")
                    else:
                        lines.append(text)

    doc.close()
    return "\n".join(lines)


def _load_with_pdfplumber(filepath: str) -> str:
    """Extract text using pdfplumber."""
    import pdfplumber
    texts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def _load_with_pypdf(filepath: str) -> str:
    """Extract text using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(filepath)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def load_pdf_as_doc(
    filepath: str,
    doc_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Load a PDF and return a dict ready to pass to pipeline.index_documents().

    Returns:
        {
            "doc_id":    "paper_transformers",
            "doc_title": "Attention Is All You Need",
            "text":      "# Introduction\n...",
            "metadata":  {"author": "...", "year": "..."}
        }
    """
    filename = os.path.basename(filepath)
    title    = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
    doc_id   = doc_id or re.sub(r'\W+', '_', title.lower())

    print(f"[PDFLoader] Loading '{filename}'...")
    text = load_pdf(filepath)
    print(f"[PDFLoader] Extracted {len(text):,} characters from '{filename}'.")

    return {
        "doc_id":    doc_id,
        "doc_title": title,
        "text":      text,
        "metadata":  metadata or {},
    }


def load_pdf_folder(
    folder_path: str,
    metadata_map: Optional[Dict[str, dict]] = None,
) -> List[dict]:
    """
    Load all PDFs from a folder.

    Args:
        folder_path  : path to folder containing PDF files
        metadata_map : optional dict mapping filename → metadata dict
                       e.g. {"paper.pdf": {"author": "Smith", "year": "2023"}}

    Returns:
        List of doc dicts ready for pipeline.index_documents()
    """
    metadata_map = metadata_map or {}
    docs = []

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[PDFLoader] No PDF files found in '{folder_path}'")
        return []

    print(f"[PDFLoader] Found {len(pdf_files)} PDF(s) in '{folder_path}'")
    for filename in sorted(pdf_files):
        filepath = os.path.join(folder_path, filename)
        meta     = metadata_map.get(filename, {})
        try:
            doc = load_pdf_as_doc(filepath, metadata=meta)
            docs.append(doc)
        except Exception as e:
            print(f"[PDFLoader] Failed to load '{filename}': {e}")

    return docs