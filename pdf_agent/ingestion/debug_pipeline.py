# ingestion/debug_pipeline.py
import sys
from ingestion.pipeline import run_ingestion_pipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.debug_pipeline <pdf_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"\nRunning ingestion pipeline on: {path}\n")
    result = run_ingestion_pipeline(path)

    if not result.success:
        print(f"[FAILED] {result.error}")
        sys.exit(1)

    print(f"[SUCCESS]")
    print(f"  doc_id:        {result.doc_id}")
    print(f"  pages:         {result.page_count}")
    print(f"  total chars:   {result.total_chars:,}")
    print(f"  scanned pages: {result.scanned_pages}")
    print(f"\n--- Page summaries ---")

    for page in result.pages[:10]:
        section = page.section_title or "(none)"
        preview = page.raw_text[:120].replace("\n", " ")
        print(
            f"\nPage {page.page_number:>3} | "
            f"section={section!r:40} | "
            f"chars={len(page.raw_text):>5} | "
            f"preview={preview!r}"
        )

    if result.page_count > 10:
        print(f"\n  ... and {result.page_count - 10} more pages.")

    # Section coverage report
    print(f"\n--- Section coverage ---")
    headings_seen = {}
    for page in result.pages:
        if page.section_title and page.section_title not in headings_seen:
            headings_seen[page.section_title] = page.page_number

    for heading, first_page in headings_seen.items():
        print(f"  Page {first_page:>3}: {heading!r}")
