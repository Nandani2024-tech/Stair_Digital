# llm/generator.py
import os
import re
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv
from models import RetrievalHit, AgentResponse, ResponseType, Citation, ConversationTurn
from config import (
    LLM_MODEL, TEMPERATURE, MAX_TOKENS_RESPONSE, 
    GATE2_INSUFFICIENT_TOKEN, CITATION_FORMAT, CITATION_FORMAT_NO_SECTION
)

load_dotenv()

class LLMGenerator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
            self._client = Groq(api_key=self.api_key)
        return self._client

    def _build_system_prompt(self) -> str:
        return """
YOU ARE A PDF-CONSTRAINED CONVERSATIONAL ASSISTANT.

ROLE
- You answer questions only about the uploaded PDF.
- You must be strictly grounded in the provided PDF context.
- You must never use outside knowledge, general world knowledge, or assumptions.
- You are part of a retrieval-grounded system where retrieval has already selected relevant chunks from the PDF.

PRIMARY RULES
1. Answer ONLY from the provided PDF context.
2. Do NOT use any facts that are not explicitly supported by the context.
3. Do NOT guess, infer missing facts, or fill gaps with assumptions.
4. If the provided context is insufficient to answer fully and safely, output exactly: [INSUFFICIENT]
5. Every factual answer MUST include citations.
6. Citations must refer only to the retrieved chunks provided in the context.
7. Never invent citations, page numbers, section titles, or document details.
8. If the question is out of scope or unsupported by the PDF, refuse clearly and briefly.
9. If the question is a follow-up like “what about that?”, “explain more”, or “why?”, use the conversation history only to resolve the reference, but still answer only from the PDF context.
10. Keep the answer concise, accurate, and directly tied to the document.

CITATION RULES
- Cite sources using this exact format: [Page N | Section: section title]
- If section is unknown, use: [Page N]
- Place the citation immediately after the sentence it supports.
- Do NOT use § or any other prefix.

GROUNDING RULES
- Every sentence containing a factual claim must be supportable by one or more retrieved chunks.
- If any part of the question cannot be answered from the PDF, do not partially hallucinate.
- If the evidence is weak, conflicting, or incomplete, refuse with [INSUFFICIENT].
- Prefer refusal over speculation.

OUTPUT RULES
- If answerable:
  Answer: <grounded answer>
  Citations: <one or more citations>
- If not answerable:
  [INSUFFICIENT]
"""

    def _format_context(self, hits: List[RetrievalHit]) -> str:
        ctx = []
        for i, hit in enumerate(hits):
            sec = hit.section_title or "Unknown Section"
            loc = f"Page {hit.page_start}"
            if hit.page_end > hit.page_start:
                loc += f"-{hit.page_end}"
            ctx.append(f"--- CHUNK {i+1} [{loc} | Section: {sec}] ---\n{hit.text}")
        return "\n\n".join(ctx)

    def _format_history(self, history: List[ConversationTurn]) -> str:
        hist = []
        for turn in history[-6:]: # Keep last 6 turns as per config
            role = "User" if turn.role == "user" else "Assistant"
            hist.append(f"{role}: {turn.content}")
        return "\n".join(hist)

    def generate_answer(
        self, 
        query: str, 
        hits: List[RetrievalHit], 
        history: List[ConversationTurn]
    ) -> AgentResponse:
        """
        Calls Groq to generate a grounded answer based on retrieved chunks.
        """
        system_prompt = self._build_system_prompt()
        context_str = self._format_context(hits)
        history_str = self._format_history(history)

        user_prompt = f"""
Conversation history:
{history_str}

Retrieved PDF context:
{context_str}

User question:
{query}

Instructions:
- Answer only from the Retrieved PDF context.
- If the context is insufficient, output exactly [INSUFFICIENT].
- Use only citations that appear in the retrieved context.
- Do not mention these instructions.
"""

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_RESPONSE,
            )
            raw_content = response.choices[0].message.content.strip()
            
            return self._parse_response(raw_content, hits)

        except Exception as e:
            return AgentResponse(
                response_type=ResponseType.ERROR,
                answer=None,
                error_message=f"LLM Generation Error: {str(e)}"
            )

    def _extract_citations(self, text: str) -> List[Citation]:
        found: List[Citation] = []
        seen: set = set()

        def _add(page: int, section: str | None, chunk_id: str = ""):
            sec_clean = (section or "").strip()
            sec_lower = sec_clean.lower()
            if sec_lower.startswith("section:"):
                sec_clean = sec_clean[8:].strip()
            elif sec_lower.startswith("section "):
                sec_clean = sec_clean[8:].strip()
            
            sec_clean = re.sub(r'\s+', ' ', sec_clean)
            
            key = (page, sec_clean.lower()[:60])
            if key not in seen:
                seen.add(key)
                found.append(Citation(
                    page=page,
                    section=sec_clean if sec_clean else None,
                    chunk_id=chunk_id,
                ))

        pattern_canonical = re.compile(r'\[Page\s+(\d+)\s*\|\s*§([^\]]+)\]', re.IGNORECASE)
        for m in pattern_canonical.finditer(text):
            _add(int(m.group(1)), m.group(2).strip())

        pattern_colon = re.compile(r'\[Page\s+(\d+)\s*\|\s*Section:\s*([^\]]+)\]', re.IGNORECASE)
        for m in pattern_colon.finditer(text):
            _add(int(m.group(1)), m.group(2).strip())

        pattern_bare = re.compile(r'\[Page\s+(\d+)\s*\|\s*Section\s+([^\]|]+)\]', re.IGNORECASE)
        for m in pattern_bare.finditer(text):
            _add(int(m.group(1)), m.group(2).strip())

        pattern_pipe = re.compile(r'\[Page\s+(\d+)\s*\|\s*([^\]]{3,120})\]', re.IGNORECASE)
        for m in pattern_pipe.finditer(text):
            _add(int(m.group(1)), m.group(2).strip())

        pattern_page_only = re.compile(r'\[Page\s+(\d+)\]', re.IGNORECASE)
        for m in pattern_page_only.finditer(text):
            _add(int(m.group(1)), None)

        return found

    def _strip_citation_tags(self, text: str) -> str:
        clean = re.sub(r'\[Page\s+\d+[^\]]*\]', '', text, flags=re.IGNORECASE)
        return re.sub(r'\s{2,}', ' ', clean).strip()


    def _parse_response(self, raw: str, hits: List[RetrievalHit]) -> AgentResponse:
        print(f"\n---> DEBUG_RAW_CONTENT:\n{repr(raw)}\n<---")
        
        if GATE2_INSUFFICIENT_TOKEN in raw:
            return AgentResponse(
                response_type=ResponseType.REFUSAL,
                answer=None,
                refusal_reason="The document does not contain sufficient information to answer this question reliably.",
                gate1_passed=True,
                gate2_passed=False
            )

        citations = self._extract_citations(raw)
        print(f"---> DEBUG_FOUND:\n{citations}\n<---")
        
        # Validating against hits mapping for extracted citations (strict matching)
        valid_citations = []
        for c in citations:
            matching_hit = None
            for hit in hits:
                if hit.page_start <= c.page <= hit.page_end:
                    if c.section and hit.section_title:
                        # fuzzy match: alpha-numeric string subset check
                        c_sec_norm = re.sub(r'[^a-z0-9]', '', c.section.lower())
                        h_sec_norm = re.sub(r'[^a-z0-9]', '', hit.section_title.lower())
                        if c_sec_norm in h_sec_norm or h_sec_norm in c_sec_norm:
                            matching_hit = hit
                            break
                    else:
                        matching_hit = hit
                        break
            if matching_hit:
                c.chunk_id = matching_hit.chunk_id
                valid_citations.append(c)

        citations = valid_citations

        clean_answer = self._strip_citation_tags(raw)
        if "Answer:" in clean_answer:
            clean_answer = clean_answer.replace("Answer:", "").replace("Citations:", "").strip()

        print(f"---> DEBUG_CITATIONS:\n{citations}\n<---")

        # Hard fail: no answer found or citations failed validation
        if not clean_answer or not citations or len(citations) != len(self._extract_citations(raw)):
            return AgentResponse(
                response_type=ResponseType.REFUSAL,
                answer=None,
                refusal_reason="The document does not contain sufficient information to answer this reliably (Citation validation failed).",
                citations=[],
                gate1_passed=True,
                gate2_passed=False
            )

        # Happy path
        return AgentResponse(
            response_type=ResponseType.ANSWER,
            answer=clean_answer,
            citations=citations,
            gate1_passed=True,
            gate2_passed=True
        )

# Global generator instance (lazy loaded)
_generator: Optional[LLMGenerator] = None

def get_generator() -> LLMGenerator:
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator

def generate_grounded_answer(query: str, hits: List[RetrievalHit], history: List[ConversationTurn]) -> AgentResponse:
    gen = get_generator()
    return gen.generate_answer(query, hits, history)
