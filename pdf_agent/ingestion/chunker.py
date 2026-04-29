# ingestion/chunker.py
import re
from typing import Optional
from models import ParsedPage, Chunk
from config import (
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    MIN_CHUNK_LENGTH_CHARS,
)


def _count_tokens(text: str) -> int:
    """Word-level token approximation."""
    return len(text.split())


def _split_text_into_token_windows(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Split text into windows of max_tokens words with overlap.
    Tries to split at sentence boundaries to avoid cutting mid-thought.
    """
    # Robust sentence splitting that preserves punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    windows: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        s_tokens = _count_tokens(sentence)
        
        # If adding this sentence exceeds budget
        if current_tokens + s_tokens > max_tokens and current_chunk:
            windows.append(" ".join(current_chunk))
            
            # Create overlap from the end of current_chunk
            # We try to keep last few sentences that fit in overlap_tokens
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk):
                st = _count_tokens(s)
                if overlap_count + st <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_count += st
                else:
                    break
            
            current_chunk = overlap_sentences
            current_tokens = overlap_count

        current_chunk.append(sentence)
        current_tokens += s_tokens

    if current_chunk:
        windows.append(" ".join(current_chunk))

    return [w for w in windows if w.strip()]


def _group_pages_by_section(
    pages: list[ParsedPage],
) -> list[dict]:
    """
    Group consecutive pages sharing the same section_title.
    Returns list of dicts:
      {
        "section_title": str | None,
        "pages": list[ParsedPage],   # the pages in this group
      }
    """
    if not pages:
        return []

    groups: list[dict] = []
    current_title = pages[0].section_title
    current_group: list[ParsedPage] = [pages[0]]

    for page in pages[1:]:
        if page.section_title == current_title:
            current_group.append(page)
        else:
            groups.append({
                "section_title": current_title,
                "pages": current_group,
            })
            current_title = page.section_title
            current_group = [page]

    groups.append({
        "section_title": current_title,
        "pages": current_group,
    })

    return groups


def chunk_pages(
    pages: list[ParsedPage],
    doc_id: str,
) -> list[Chunk]:
    """
    Convert list[ParsedPage] into list[Chunk].
    Section boundaries are respected as hard splits.
    Each chunk carries page_start, page_end, section_title.
    """
    if not pages:
        return []

    section_groups = _group_pages_by_section(pages)
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for group in section_groups:
        section_title: Optional[str] = group["section_title"]
        group_pages: list[ParsedPage] = group["pages"]

        # Build full section text, tracking page boundaries per word
        page_paragraphs: list[tuple[int, str]] = []
        for page in group_pages:
            if page.raw_text.strip():
                page_paragraphs.append((page.page_number, page.raw_text))

        if not page_paragraphs:
            continue

        # -- Build segment list: each segment = one page's text --
        segments: list[tuple[int, int, str]] = [
            (_count_tokens(text), page_num, text)
            for page_num, text in page_paragraphs
        ]

        # Sliding window over segments
        buffer_texts:   list[str] = []
        buffer_pages:   list[int] = []
        buffer_tokens:  int = 0

        def _flush_buffer(
            buf_texts: list[str],
            buf_pages: list[int],
            idx: int,
        ) -> tuple[list[Chunk], int]:
            nonlocal all_chunks
            local_chunks: list[Chunk] = []
            combined = " ".join(buf_texts).strip()
            if len(combined) < MIN_CHUNK_LENGTH_CHARS:
                return local_chunks, idx

            token_count = _count_tokens(combined)
            if token_count <= CHUNK_SIZE_TOKENS:
                # Single chunk
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{idx:03d}",
                    text=combined,
                    page_start=min(buf_pages),
                    page_end=max(buf_pages),
                    section_title=section_title,
                    doc_id=doc_id,
                    token_count=token_count,
                    is_ocr_derived=any(
                        p.is_ocr_derived
                        for p in group_pages
                        if p.page_number in buf_pages
                    ),
                )
                local_chunks.append(chunk)
                idx += 1
            else:
                # Sub-split within this buffer
                sub_windows = _split_text_into_token_windows(
                    combined,
                    CHUNK_SIZE_TOKENS,
                    CHUNK_OVERLAP_TOKENS,
                )
                print(
                    f"[chunker] Page(s) {buf_pages} in section "
                    f"{section_title!r} required sub-splitting "
                    f"into {len(sub_windows)} sub-chunks."
                )
                for sub_text in sub_windows:
                    if len(sub_text.strip()) < MIN_CHUNK_LENGTH_CHARS:
                        continue
                    chunk = Chunk(
                        chunk_id=f"{doc_id}_chunk_{idx:03d}",
                        text=sub_text,
                        page_start=min(buf_pages),
                        page_end=max(buf_pages),
                        section_title=section_title,
                        doc_id=doc_id,
                        token_count=_count_tokens(sub_text),
                        is_ocr_derived=False,
                    )
                    local_chunks.append(chunk)
                    idx += 1

            return local_chunks, idx

        for seg_tokens, page_num, seg_text in segments:
            if buffer_tokens + seg_tokens > CHUNK_SIZE_TOKENS and buffer_texts:
                # Flush current buffer
                new_chunks, chunk_index = _flush_buffer(
                    buffer_texts, buffer_pages, chunk_index
                )
                all_chunks.extend(new_chunks)

                # Seed overlap: take last OVERLAP tokens from buffer
                overlap_text  = " ".join(buffer_texts)
                overlap_words = overlap_text.split()[-CHUNK_OVERLAP_TOKENS:]
                buffer_texts  = [" ".join(overlap_words)] if overlap_words else []
                buffer_pages  = [buffer_pages[-1]] if buffer_pages else []
                buffer_tokens = _count_tokens(" ".join(buffer_texts))

            buffer_texts.append(seg_text)
            buffer_pages.append(page_num)
            buffer_tokens += seg_tokens

        # Flush remaining buffer for this section
        if buffer_texts:
            new_chunks, chunk_index = _flush_buffer(
                buffer_texts, buffer_pages, chunk_index
            )
            all_chunks.extend(new_chunks)

    # -- Summary log -------------------------------------
    section_counts: dict[str, int] = {}
    for chunk in all_chunks:
        key = chunk.section_title or "(no section)"
        section_counts[key] = section_counts.get(key, 0) + 1

    print(f"[chunker] Total chunks produced: {len(all_chunks)}")
    for sec, count in section_counts.items():
        print(f"  {count:>3} chunk(s) - section: {sec!r}")

    return all_chunks
