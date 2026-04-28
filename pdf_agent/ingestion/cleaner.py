# ingestion/cleaner.py
import re
from collections import Counter
from models import ParsedPage


# -- patterns that identify standalone page number lines ------
_PAGE_NUM_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*$"),                     # bare integer
    re.compile(r"^\s*[-–]\s*\d{1,4}\s*[-–]\s*$"),       # - N -
    re.compile(r"^\s*[Pp]age\s*:?\s*\d{1,4}\s*$"),      # Page N / page N
    re.compile(r"^\s*[ivxlcdmIVXLCDM]{1,6}\s*$"),       # Roman numerals
]


def _is_page_number_line(line: str) -> bool:
    if len(line.strip()) > 80:
        return False
    return any(p.match(line) for p in _PAGE_NUM_PATTERNS)


def _collect_boundary_lines(pages: list[ParsedPage]) -> Counter:
    """
    Collect lines that appear in the first 2 or last 2 lines
    of each page's raw_text. Count occurrences across all pages.
    """
    counter: Counter = Counter()
    for page in pages:
        lines = page.raw_text.splitlines()
        boundary = lines[:2] + lines[-2:]
        for line in boundary:
            stripped = line.strip()
            if stripped and len(stripped) <= 80:
                counter[stripped] += 1
    return counter


def clean_pages(pages: list[ParsedPage]) -> list[ParsedPage]:
    """
    Return a new list[ParsedPage] with:
      - repeated header/footer lines removed (>= 3 pages)
      - standalone page number lines removed
      - raw_text updated to reflect cleaned content
      - blocks untouched
    """
    if not pages:
        return pages

    # -- 1. Detect repeated boundary noise ---------------
    boundary_counts = _collect_boundary_lines(pages)
    noise_lines: set[str] = {
        line for line, count in boundary_counts.items()
        if count >= 3
    }

    if noise_lines:
        print(
            f"[cleaner] Detected {len(noise_lines)} repeated "
            f"header/footer line(s) across pages:"
        )
        for nl in sorted(noise_lines):
            print(f"  -> stripped: {nl!r}")

    # -- 2. Clean each page ------------------------------
    cleaned: list[ParsedPage] = []
    total_stripped = 0

    for page in pages:
        original_lines  = page.raw_text.splitlines()
        filtered_lines  = []
        page_stripped   = 0

        for line in original_lines:
            stripped = line.strip()

            # Remove repeated header/footer noise
            if stripped in noise_lines:
                page_stripped += 1
                continue

            # Remove standalone page numbers
            if _is_page_number_line(line):
                page_stripped += 1
                continue

            filtered_lines.append(line)

        total_stripped += page_stripped

        # Collapse runs of 3+ blank lines into 2
        collapsed: list[str] = []
        blank_run = 0
        for line in filtered_lines:
            if line.strip() == "":
                blank_run += 1
                if blank_run <= 2:
                    collapsed.append(line)
            else:
                blank_run = 0
                collapsed.append(line)

        cleaned_text = "\n".join(collapsed).strip()

        cleaned.append(ParsedPage(
            page_number=page.page_number,
            raw_text=cleaned_text,
            blocks=page.blocks,            # preserved unchanged
            section_title=page.section_title,  # still None here
            is_ocr_derived=page.is_ocr_derived,
        ))

    print(
        f"[cleaner] Done. Stripped {total_stripped} noise lines "
        f"across {len(pages)} pages."
    )
    return cleaned
