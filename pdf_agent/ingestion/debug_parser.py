# ingestion/debug_parser.py
import sys
from ingestion.loader import load_pdf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.debug_parser <pdf_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"\nLoading: {path}")

    result = load_pdf(path)

    if not result.success:
        print(f"\n[FAILED] {result.error}")
        sys.exit(1)

    print(f"\n[SUCCESS]")
    print(f"  doc_id:        {result.doc_id}")
    print(f"  pages:         {len(result.pages)}")
    print(f"  total chars:   {result.total_chars}")
    print(f"  scanned pages: {result.scanned_pages}")

    print("\n--- Page previews ---\n")
    for page in result.pages:
        print(f"Page {page.page_number}")
        print(f"  blocks:       {len(page.blocks)}")
        print(f"  chars:        {len(page.raw_text)}")
        preview = page.raw_text[:60].replace("\n", " ")
        print(f"  text preview: '{preview}...'")
        
        # Print first few blocks and their fonts
        for i, block in enumerate(page.blocks[:3]):
            # Extract font info from spans
            fonts = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_info = f"font={span.get('font')} size={span.get('size'):.1f} flags={span.get('flags')}"
                    text_preview = span.get("text", "").strip()[:40]
                    fonts.append(f"{font_info} text='{text_preview}'")
            
            if fonts:
                print(f"  block[{i}]: {fonts[0]}")
        print("-" * 20)
