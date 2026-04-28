# ingestion/pipeline.py
from dataclasses import dataclass, field
from typing import Optional
from models import ParsedPage, Chunk
from ingestion.loader import load_pdf, LoaderResult
from ingestion.cleaner import clean_pages
from ingestion.metadata import detect_sections
from ingestion.chunker import chunk_pages


@dataclass
class PipelineResult:
    success:        bool
    doc_id:         Optional[str]
    pages:          list[ParsedPage]
    chunks:         list[Chunk]
    page_count:     int
    chunk_count:    int
    total_chars:    int
    scanned_pages:  int
    error:          Optional[str]
    source_path:    Optional[str]


def run_ingestion_pipeline(file_source) -> PipelineResult:
    """
    Full ingestion pipeline:
      load_pdf -> clean_pages -> detect_sections -> chunk_pages
    Returns PipelineResult - never raises.
    """
    _empty = PipelineResult(
        success=False, doc_id=None, pages=[], chunks=[],
        page_count=0, chunk_count=0, total_chars=0,
        scanned_pages=0, error=None, source_path=None,
    )

    # Step 1: Load
    loader_result: LoaderResult = load_pdf(file_source)
    if not loader_result.success:
        _empty.doc_id       = loader_result.doc_id
        _empty.scanned_pages= loader_result.scanned_pages
        _empty.error        = loader_result.error
        _empty.source_path  = loader_result.source_path
        return _empty

    # Step 2: Clean
    try:
        cleaned_pages = clean_pages(loader_result.pages)
    except Exception as e:
        _empty.doc_id      = loader_result.doc_id
        _empty.page_count  = loader_result.page_count
        _empty.scanned_pages = loader_result.scanned_pages
        _empty.error       = f"Cleaning step failed: {e}"
        _empty.source_path = loader_result.source_path
        return _empty

    # Step 3: Detect sections
    try:
        sectioned_pages = detect_sections(cleaned_pages)
    except Exception as e:
        _empty.doc_id      = loader_result.doc_id
        _empty.page_count  = loader_result.page_count
        _empty.scanned_pages = loader_result.scanned_pages
        _empty.error       = f"Section detection failed: {e}"
        _empty.source_path = loader_result.source_path
        return _empty

    # Step 4: Chunk
    try:
        chunks = chunk_pages(sectioned_pages, loader_result.doc_id)
    except Exception as e:
        _empty.doc_id      = loader_result.doc_id
        _empty.page_count  = len(sectioned_pages)
        _empty.scanned_pages = loader_result.scanned_pages
        _empty.error       = f"Chunking step failed: {e}"
        _empty.source_path = loader_result.source_path
        return _empty

    total_chars = sum(len(p.raw_text) for p in sectioned_pages)

    return PipelineResult(
        success=True,
        doc_id=loader_result.doc_id,
        pages=sectioned_pages,
        chunks=chunks,
        page_count=len(sectioned_pages),
        chunk_count=len(chunks),
        total_chars=total_chars,
        scanned_pages=loader_result.scanned_pages,
        error=None,
        source_path=loader_result.source_path,
    )
