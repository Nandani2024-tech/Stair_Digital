import sys
import os
sys.path.insert(0, os.getcwd())

from ingestion.pipeline import run_ingestion_pipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.debug_chunker <pdf_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"\nRunning full pipeline on: {path}\n")
    result = run_ingestion_pipeline(path)

    if not result.success:
        print(f"[FAILED] {result.error}")
        sys.exit(1)

    print(f"[SUCCESS]")
    print(f"  doc_id:      {result.doc_id}")
    print(f"  pages:       {result.page_count}")
    print(f"  chunks:      {result.chunk_count}")
    print(f"  total chars: {result.total_chars:,}")
    print(f"\n--- Chunk inventory ---")

    for chunk in result.chunks:
        pages = (
            f"p{chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"p{chunk.page_start}-{chunk.page_end}"
        )
        section = chunk.section_title or "(no section)"
        preview = chunk.text[:100].replace("\n", " ")
        print(
            f"\n  {chunk.chunk_id}"
            f"\n    pages:   {pages}"
            f"\n    section: {section!r}"
            f"\n    tokens:  {chunk.token_count}"
            f"\n    preview: {preview!r}"
        )

    print(f"\n--- Section chunk counts ---")
    section_counts: dict = {}
    for chunk in result.chunks:
        key = chunk.section_title or "(no section)"
        section_counts[key] = section_counts.get(key, 0) + 1
    for sec, count in section_counts.items():
        print(f"  {count:>3}  {sec!r}")

    # Integrity checks
    print(f"\n--- Integrity checks ---")
    oversized = [c for c in result.chunks if c.token_count > 512]
    undersized = [c for c in result.chunks if len(c.text.strip()) < 100]
    missing_id = [c for c in result.chunks if not c.chunk_id]
    missing_section = [c for c in result.chunks if c.section_title is None]

    print(f"  Oversized chunks  (>512 tokens):     {len(oversized)}")
    print(f"  Undersized chunks (<100 chars):       {len(undersized)}")
    print(f"  Chunks missing chunk_id:              {len(missing_id)}")
    print(f"  Chunks with no section_title:         {len(missing_section)}")

    if oversized:
        print(f"\n  [WARN] Oversized chunk IDs:")
        for c in oversized:
            print(f"    {c.chunk_id} - {c.token_count} tokens")
