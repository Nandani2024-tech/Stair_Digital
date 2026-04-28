from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class ResponseType(Enum):
    ANSWER      = "answer"       # grounded answer with citations
    REFUSAL     = "refusal"      # Gate 1 or Gate 2 triggered
    ERROR       = "error"        # unhandled exception, clean message
    CLARIFY     = "clarify"      # query too ambiguous to rewrite

@dataclass
class ParsedPage:
    page_number: int             # 1-indexed
    raw_text: str
    blocks: list[dict]           # PyMuPDF block dicts preserved
    section_title: Optional[str] # detected heading, None if unknown
    is_ocr_derived: bool = False

@dataclass
class Chunk:
    chunk_id: str                # e.g. "doc_abc123_chunk_017"
    text: str
    page_start: int
    page_end: int
    section_title: Optional[str]
    doc_id: str
    token_count: int
    is_ocr_derived: bool = False

@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float                 # lower = more similar (Chroma L2 distance)
    rank: int                    # 1-indexed position in result list

@dataclass
class RerankedHit:
    chunk: Chunk
    cross_encoder_score: float   # higher = more relevant
    original_rank: int
    reranked_rank: int

@dataclass
class Citation:
    page: int
    section: Optional[str]
    chunk_id: str

    def render(self) -> str:
        from config import CITATION_FORMAT, CITATION_FORMAT_NO_SECTION
        if self.section:
            return CITATION_FORMAT.format(page=self.page, section=self.section)
        return CITATION_FORMAT_NO_SECTION.format(page=self.page)

@dataclass
class AgentResponse:
    response_type: ResponseType
    answer: Optional[str]        # None when response_type is REFUSAL or ERROR
    citations: list[Citation] = field(default_factory=list)
    refusal_reason: Optional[str] = None
    error_message: Optional[str] = None
    rewritten_query: Optional[str] = None   # populated if query was rewritten
    retrieval_hits: list[RetrievalHit] = field(default_factory=list)
    gate1_passed: Optional[bool] = None
    gate2_passed: Optional[bool] = None

@dataclass
class ConversationTurn:
    role: str                    # "user" or "assistant"
    content: str
    citations: list[Citation] = field(default_factory=list)
    response_type: Optional[ResponseType] = None
