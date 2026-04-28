# ingestion/parser.py
import fitz
from models import ParsedPage


def parse_pdf_to_pages(doc: fitz.Document) -> list[ParsedPage]:
    """
    Extracts text and block-level metadata from all pages in a PyMuPDF document.
    """
    parsed_pages: list[ParsedPage] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        
        # We use getting text as a dict to preserve block structure
        # which is useful for heading detection and cleaning later.
        content_dict = page.get_text("dict")
        
        raw_text_parts = []
        text_blocks = []

        for block in content_dict.get("blocks", []):
            # Type 0 is text. Type 1 is image.
            if block.get("type") == 0:
                text_blocks.append(block)
                # Extract text for the raw_text preview/grounding
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                
                if block_text.strip():
                    raw_text_parts.append(block_text.strip())

        # Join text with double newlines to separate blocks clearly
        full_text = "\n\n".join(raw_text_parts)

        parsed_pages.append(ParsedPage(
            page_number=page_num,
            raw_text=full_text,
            blocks=text_blocks,
            section_title=None, # Populated in Phase 3
            is_ocr_derived=False, # Grounded extraction
        ))

    return parsed_pages
