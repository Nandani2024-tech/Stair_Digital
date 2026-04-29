# ingestion/metadata.py
import re
import statistics
from typing import Optional
from models import ParsedPage
from ingestion.cleaner import _is_page_number_line


# -- numbered section pattern ---------------------------------
_SECTION_NUM_RE = re.compile(
    r"^\s*(\d+\.)+(\d+)?"          # 1.  /  1.2  /  1.2.3
    r"|^\s*[A-Z]\.\s"              # A.  B.
    r"|^\s*[IVXLCDM]{1,6}\.\s",   # I.  II.  IV.
    re.IGNORECASE,
)


def _compute_median_font_size(pages: list[ParsedPage]) -> float:
    """Collect all span font sizes across all pages; return median."""
    sizes: list[float] = []
    for page in pages:
        for block in page.blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    if size > 0:
                        sizes.append(size)
    if not sizes:
        return 12.0   # safe fallback
    return statistics.median(sizes)


def _is_noise_heading(candidate: str) -> bool:
    """Check if the text is a common non-semantic label (source, chart, disclaimer)."""
    c = candidate.lower().strip()
    
    # Common attribution/source patterns
    if c.startswith("source:") or c.startswith("source :"):
        return True
    if c.startswith("chart") and any(char.isdigit() for char in c):
        return True
    if c.startswith("table") and any(char.isdigit() for char in c):
        return True
    if c == "disclaimer":
        return True
    if c.startswith("note:"):
        return True
    
    # Very short or common noise
    if len(c) < 3:
        return True
        
    return False


def _score_block_as_heading(
    block: dict,
    median_font: float,
) -> tuple[int, str]:
    """
    Score a single block as a potential heading.
    Returns (score, candidate_text).
    score >= 4 -> treat as heading.
    """
    lines = block.get("lines", [])
    if not lines:
        return 0, ""

    # Only consider blocks with 1-2 lines
    if len(lines) > 2:
        return 0, ""

    # Collect all span text + max font size + flags
    all_text_parts: list[str] = []
    max_size = 0.0
    any_bold = False

    for line in lines:
        for span in line.get("spans", []):
            text = span.get("text", "").strip()
            if text:
                all_text_parts.append(text)
            size  = span.get("size", 0)
            flags = span.get("flags", 0)
            if size > max_size:
                max_size = size
            if flags & 16:        # bold flag
                any_bold = True

    candidate = " ".join(all_text_parts).strip()

    if not candidate or len(candidate) > 120:
        return 0, ""

    if _is_page_number_line(candidate):
        return 0, ""

    # NEW: Penalize or skip noise headings
    if _is_noise_heading(candidate):
        return 0, ""

    score = 0

    # Font size signal
    if median_font > 0 and max_size > median_font * 1.15: # slightly more permissive font check
        score += 3

    # Bold signal
    if any_bold:
        score += 2

    # Short text signal
    if len(candidate) < 80:
        score += 1

    # Title / sentence case
    words = candidate.split()
    if words and words[0][0].isupper():
        score += 1

    # Numbered section pattern
    if _SECTION_NUM_RE.match(candidate):
        score += 2

    # ALL CAPS and short
    if candidate.isupper() and len(candidate) < 60:
        score += 2 # stronger signal for all-caps headings

    return score, candidate


def detect_sections(pages: list[ParsedPage]) -> list[ParsedPage]:
    """
    Populate section_title on each ParsedPage using block-level
    font and style signals. Carries forward the last heading
    when a page has no detectable heading of its own.
    """
    if not pages:
        return pages

    median_font = _compute_median_font_size(pages)
    print(f"[metadata] Median font size across document: {median_font:.1f}pt")

    updated: list[ParsedPage] = []
    last_heading: Optional[str] = None
    HEADING_THRESHOLD = 4

    for page in pages:
        best_score   = 0
        best_heading = None

        for block in page.blocks:
            score, candidate = _score_block_as_heading(block, median_font)
            if score >= HEADING_THRESHOLD and score > best_score:
                best_score   = score
                best_heading = candidate

        if best_heading:
            # Clean the heading string
            section_title = best_heading.rstrip(".:").strip()
            last_heading  = section_title
            print(
                f"[metadata] Page {page.page_number}: "
                f"heading detected -> {section_title!r} (score={best_score})"
            )
        else:
            # Inherit from previous page
            section_title = last_heading
            if last_heading:
                print(
                    f"[metadata] Page {page.page_number}: "
                    f"no heading - inheriting {last_heading!r}"
                )
            else:
                print(
                    f"[metadata] Page {page.page_number}: "
                    f"no heading detected, section_title=None"
                )

        updated.append(ParsedPage(
            page_number=page.page_number,
            raw_text=page.raw_text,
            blocks=page.blocks,
            section_title=section_title,
            is_ocr_derived=page.is_ocr_derived,
        ))

    headings_found = sum(1 for p in updated if p.section_title)
    print(
        f"[metadata] Section detection complete. "
        f"{headings_found}/{len(pages)} pages have a section_title."
    )
    return updated
