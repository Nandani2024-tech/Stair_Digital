# ingestion/loader.py
import os
import hashlib
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import Optional
from models import ParsedPage
from config import (
    UPLOAD_DIR,
    MAX_PDF_SIZE_MB,
    FILE_SIZE_ERROR_MESSAGE,
    CORRUPTED_PDF_MESSAGE,
    PASSWORD_PROTECTED_MESSAGE,
)
from ingestion.parser import parse_pdf_to_pages


@dataclass
class LoaderResult:
    success:        bool
    doc_id:         Optional[str]
    pages:          list[ParsedPage]
    page_count:     int
    total_chars:    int
    scanned_pages:  int
    error:          Optional[str]
    source_path:    Optional[str]


def load_pdf(file_source) -> LoaderResult:
    """
    Main entry point for loading a PDF. 
    Handles validation, disk saving, and extraction routing.
    
    file_source: Either a Streamlit UploadedFile or a local path string.
    """
    # ── 1. Read Raw Bytes ────────────────────────────────
    if isinstance(file_source, str):
        # Local path
        if not os.path.exists(file_source):
            return LoaderResult(
                success=False, error=f"File not found: {file_source}",
                doc_id="error", pages=[], page_count=0,
                total_chars=0, scanned_pages=0, source_path=None
            )
        with open(file_source, "rb") as f:
            raw_bytes = f.read()
            filename = os.path.basename(file_source)
    else:
        # Streamlit UploadedFile
        raw_bytes = file_source.read()
        filename = file_source.name

    # ── 2. Basic Validation ─────────────────────────────
    if len(raw_bytes) == 0:
        return LoaderResult(
            success=False, doc_id="none", pages=[], page_count=0,
            total_chars=0, scanned_pages=0,
            error="The uploaded file is empty (0 bytes). Please re-export or re-download the PDF and try again.",
            source_path=None,
        )

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        return LoaderResult(
            success=False, doc_id="none", pages=[], page_count=0,
            total_chars=0, scanned_pages=0,
            error=FILE_SIZE_ERROR_MESSAGE.format(size=MAX_PDF_SIZE_MB),
            source_path=None,
        )

    # ── 3. Save to Disk (Persistence) ────────────────────
    doc_id = hashlib.md5(raw_bytes).hexdigest()[:12]
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(raw_bytes)

    # ── 4. Open with PyMuPDF ─────────────────────────────
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception as e:
        err_msg = str(e).lower()
        if "password" in err_msg or "authenticated" in err_msg:
             return LoaderResult(
                success=False, doc_id=doc_id, pages=[], page_count=0,
                total_chars=0, scanned_pages=0,
                error=PASSWORD_PROTECTED_MESSAGE,
                source_path=save_path,
            )
        return LoaderResult(
            success=False, doc_id=doc_id, pages=[], page_count=0,
            total_chars=0, scanned_pages=0,
            error=f"{CORRUPTED_PDF_MESSAGE} Detail: {e}",
            source_path=save_path,
        )

    # ── 5. Parse ────────────────────────────────────────
    try:
        parsed_pages = parse_pdf_to_pages(doc)
        doc.close()

        total_chars   = sum(len(p.raw_text) for p in parsed_pages)
        scanned_count = sum(1 for p in parsed_pages if p.raw_text.strip() == "" or (len(p.raw_text) < 30)) 
        # Note: scanned logic can be more complex, but this is a start (Phase 2)

        return LoaderResult(
            success=True,
            doc_id=doc_id,
            pages=parsed_pages,
            page_count=len(parsed_pages),
            total_chars=total_chars,
            scanned_pages=scanned_count,
            error=None,
            source_path=save_path,
        )
    except Exception as e:
        return LoaderResult(
            success=False, doc_id=doc_id, pages=[], page_count=0,
            total_chars=0, scanned_pages=0,
            error=f"Parsing error: {e}",
            source_path=save_path,
        )
