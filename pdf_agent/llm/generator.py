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
- You answer questions ONLY using the specifically provided PDF context.
- You must be strictly grounded in the provided PDF context.
- You must never use outside knowledge, general world knowledge, or assumptions.

PRIMARY RULES
1. IF THE ANSWER IS NOT EXPLICITLY IN THE CONTEXT, YOU MUST OUTPUT ONLY: [INSUFFICIENT]
2. Do NOT explain that the information is missing.
3. Do NOT say "The document does not mention...". 
4. DO NOT provide meta-commentary about the context.
5. If you cannot find the EXACT fact (e.g. a specific name, a specific date), even if the general topic is mentioned, you MUST output: [INSUFFICIENT]
6. Every factual answer MUST include citations in the format: [Page N | Section: name]
7. Citations must refer ONLY to the chunks provided.

OUTPUT FORMAT
- Answerable: 
  Answer: <your concise grounded answer>
- Not Answerable:
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
        """
        Extracts citations from raw text using multiple regex patterns. 
        Tolerant of spacing, case, and page numbers.
        """
        found: List[Citation] = []
        seen: set = set()

        def _add(page_str: str, section: str | None, chunk_id: str = ""):
            # Handle page ranges like "2-3", take the first page for simplicity
            try:
                page_part = re.split(r'[-\u2013\u2014]', page_str)[0].strip()
                page = int(page_part)
            except (ValueError, IndexError):
                return

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

        # Robust extraction for various formats
        patterns = [
            re.compile(r'\[Page\s*(\d+(?:-\d+)?)\s*\|\s*§\s*([^\]]+)\]', re.IGNORECASE),
            re.compile(r'\[Page\s*(\d+(?:-\d+)?)\s*\|\s*Section:\s*([^\]]+)\]', re.IGNORECASE),
            re.compile(r'\[Page\s*(\d+(?:-\d+)?)\s*\|\s*Section\s+([^\]|]+)\]', re.IGNORECASE),
            re.compile(r'\[Page\s*(\d+(?:-\d+)?)\s*\|\s*([^\]]{2,120})\]', re.IGNORECASE),
            re.compile(r'\[Page\s*(\d+(?:-\d+)?)\]', re.IGNORECASE)
        ]

        for pattern in patterns:
            for m in pattern.finditer(text):
                p_val = m.group(1)
                s_val = m.group(2) if len(m.groups()) > 1 else None
                _add(p_val, s_val)

        return found

    def _strip_citation_tags(self, text: str) -> str:
        """Removes citation tags from the text while preserving punctuation."""
        return re.sub(r'\[Page\s*\d+[^\]]*\]', '', text, flags=re.IGNORECASE).strip()

    def _parse_response(self, raw: str, hits: List[RetrievalHit]) -> AgentResponse:
        """
        STABILIZED PARSER: Extract citations and answer text without strict label dependency.
        - If >=1 valid citation found -> PASS.
        - If parsing failure -> REFUSAL (not ERROR).
        """
        print(f"\n---> DEBUG_RAW_CONTENT:\n{repr(raw)}\n<---")
        
        # 1. Immediate Refusal check
        refusal_keywords = [
            "not explicitly mentioned", 
            "not found in the provided context",
            "context does not reveal",
            "no mention of",
            "does not contain information about",
            "information is not available",
            "neither the document nor the context",
            "context does not provide"
        ]
        
        lower_raw = raw.lower()
        if (GATE2_INSUFFICIENT_TOKEN in raw) or any(kw in lower_raw for kw in refusal_keywords):
            return AgentResponse(
                response_type=ResponseType.REFUSAL,
                answer=None,
                refusal_reason="The document does not contain sufficient information to answer this question reliably.",
                gate1_passed=True,
                gate2_passed=False
            )

        # 2. Extract Citations (Robust)
        extracted_citations = self._extract_citations(raw)
        valid_citations = []
        for c in extracted_citations:
            for hit in hits:
                if hit.page_start <= c.page <= hit.page_end:
                    if c.section and hit.section_title:
                        # Fuzzy match section titles
                        c_sec = re.sub(r'[^a-z0-9]', '', c.section.lower())
                        h_sec = re.sub(r'[^a-z0-9]', '', hit.section_title.lower())
                        if c_sec in h_sec or h_sec in c_sec:
                            c.chunk_id = hit.chunk_id
                            valid_citations.append(c)
                            break
                    else:
                        c.chunk_id = hit.chunk_id
                        valid_citations.append(c)
                        break
        
        # 3. Clean Answer Text
        # Tolerance: Strip tags, strip specific labels if present, but don't fail if missing
        clean_answer = self._strip_citation_tags(raw)
        
        # Clean up labels like "Answer:" or "Citations:" if the model used them
        labels = ["Answer:", "Response:", "Citations:", "Sources:"]
        for label in labels:
            if clean_answer.lower().startswith(label.lower()):
                clean_answer = clean_answer[len(label):].strip()
        
        # Also clean up if they are in the middle (sometimes models repeat them)
        clean_answer = re.sub(r'(?i)\n(Answer|Citations|Sources|Response):\s*', '\n', clean_answer).strip()

        # 4. Gate 2 Critical Decision
        # REQUIREMENT: Simplified - if >=1 valid citation maps -> PASS
        gate2_passed = (len(valid_citations) >= 1) and (len(clean_answer) > 5)

        if not gate2_passed:
            return AgentResponse(
                response_type=ResponseType.REFUSAL,
                answer=None,
                refusal_reason="I cannot find sufficient evidence in the document to support an answer to this query.",
                citations=[],
                gate1_passed=True,
                gate2_passed=False
            )

        # 5. Success Path
        return AgentResponse(
            response_type=ResponseType.ANSWER,
            answer=clean_answer,
            citations=valid_citations,
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
