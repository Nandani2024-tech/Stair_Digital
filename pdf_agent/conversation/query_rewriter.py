import re
import os
import json
from typing import List
from dataclasses import dataclass
from models import ConversationTurn
from config import LLM_MODEL
from groq import Groq

@dataclass
class RewriteResult:
    rewritten_query: str
    used_history: bool
    confidence: float
    needs_clarification: bool
    rewrite_type: str        # 'none', 'heuristic', 'llm', 'clarification'
    history_turns_used: int
    query_type: str = "standalone" # 'standalone', 'follow_up', 'ambiguous'
    intent: str = "factual"
    anchor_used: str = "none"
    substitution_success: bool = False
    needs_context: bool = False
    dependency_type: str = "independent"
    is_valid_query: bool = True

class QueryRewriter:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self._client = None
        self.last_semantic_query = ""
        
    @property
    def client(self):
        if self._client is None and self.api_key:
            self._client = Groq(api_key=self.api_key)
        return self._client
        
    def _extract_subject(self, query: str) -> str:
        """Extract ONLY noun phrase / topic by removing question scaffolding."""
        if not query: return None
        
        # Scaffolding and verbs to remove
        junk = [
             r"^what (does|is|are|was|were)\s+(the\s+)?report\s+say\s+about\s+",
             r"^what (is|are|were|was)\s+(the\s+)?",
             r"^tell\s+me\s+about\s+",
             r"^explain\s+(the\s+)?",
             r"^describe\s+(the\s+)?",
             r"^summarize\s+(the\s+)?",
             r"^elaborate\s+on\s+(the\s+)?",
             r"^can\s+you\s+(explain|tell\s+me\s+about)\s+",
             r"\s+mentioned\s+in\s+the\s+report$",
             r"\s+in\s+the\s+report$",
             r"\s+mentioned$",
             r"\s+report\s+says\s+about\s+"
        ]
        
        q = query.strip()
        for pattern in junk:
            q = re.sub(pattern, "", q, flags=re.IGNORECASE).strip()
            
        q = q.rstrip("?.")
        # Final clean of leading prepositions/articles
        q = re.sub(r"^(about|on|the|of|for|a|an)\s+", "", q, flags=re.IGNORECASE)
        
        return q if len(q.split()) >= 1 else None
        
    def _get_cleaned_words(self, text: str) -> set:
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return set(clean.split())

    def _extract_context(self, history: List[ConversationTurn]) -> tuple[str, str, str, int]:
        user_turns = []
        assistant_turn = None
        
        for t in reversed(history):
            if t.role == "user" and len(user_turns) < 2:
                user_turns.insert(0, t.content)
            elif t.role == "assistant" and assistant_turn is None:
                assistant_turn = t.content
                
            if len(user_turns) == 2 and assistant_turn is not None:
                break
                
        turns_used = len(user_turns) + (1 if assistant_turn else 0)
        
        current_query = user_turns[-1] if user_turns else ""
        previous_query = user_turns[-2] if len(user_turns) >= 2 else None
        
        context_parts = []
        if len(user_turns) == 2:
            context_parts.append(f"User: {user_turns[0]}")
        if assistant_turn:
            content = assistant_turn[:500] + "..." if len(assistant_turn) > 500 else assistant_turn
            context_parts.append(f"Assistant: {content}")
        context_parts.append(f"User: {current_query}")
        
        return "\n".join(context_parts), current_query, previous_query, turns_used

    def _detect_dependency(self, query: str, context_str: str) -> dict:
        """Classify the query based on semantic dependency."""
        if not self.client:
            return {"type": "independent", "reason": "No LLM client"}

        prompt = f"""Classify the query:

1. independent → fully meaningful alone
2. dependent → relies on previous context

Return STRICT JSON:
{{
  "type": "independent" or "dependent",
  "reason": "short explanation"
}}

Context:
{context_str}

Query: {query}
"""
        try:
            res = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(res.choices[0].message.content)
        except Exception as e:
            return {"type": "independent", "reason": f"Error: {str(e)}"}

    def _semantic_merge(self, query: str, base: str) -> str:
        """Merge follow-up with semantic base using LLM."""
        if not self.client or not base:
            return f"{query} {base}".strip() if base else query
            
        prompt = f"""Rewrite the query into a concise, self-contained SEARCH QUERY.
- Preserve interrogative intent (question form OR keyword query)
- DO NOT answer the question
- DO NOT add explanations
- ONLY inject missing context from history
- Max 12-15 words
- Must remain a query OR keyword search
- No full sentences like answers

Context Topic: {base}
Follow-up Query: {query}
Standalone Query:"""

        try:
            res = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"[QueryRewriter] Merge Error: {e}")
            return f"{query} {base}"

    def rewrite(self, query: str, history: List[ConversationTurn]) -> RewriteResult:
        """Process query, detect dependency, and rewrite if necessary (Always runs)."""
        context_str, _, previous_user_query, turns_used = self._extract_context(history)
        
        # FIX 1 & 2: Anchor and Semantic Base
        anchor = self._extract_subject(previous_user_query) if previous_user_query else "none"
        semantic_base = self.last_semantic_query or (anchor if anchor != "none" else "")
        if semantic_base == "none": semantic_base = ""

        # FIX 3: Intent-aware shorthand
        shorthands = {
            "explain": "Explain",
            "summarize": "Summarize",
            "elaborate": "Elaborate",
            "tell me more": "Elaborate",
            "expand on": "Elaborate"
        }
        
        q_lower = query.lower().strip()
        shorthand_triggered = False
        rewritten_query = query
        rewrite_type = "none"
        used_history = False
        substitution_success = False

        for prefix, verb in shorthands.items():
            if q_lower.startswith(prefix) and (len(q_lower.split()) <= 4 or "that" in q_lower or "it" in q_lower):
                if semantic_base:
                    rewritten_query = f"{verb} {semantic_base}"
                    rewrite_type = "intent_shorthand"
                    used_history = True
                    shorthand_triggered = True
                    break

        # FIX 4: Strategy Upgrade
        dep_type = "independent"
        needs_context = False
        
        if not shorthand_triggered:
            # 1. Dependency Detection
            dep_data = self._detect_dependency(query, context_str)
            dep_type = dep_data.get("type", "independent")
            needs_context = (dep_type == "dependent")

            if needs_context:
                rewritten_query = self._semantic_merge(query, semantic_base)
                rewrite_type = "semantic_merge"
                used_history = True
            else:
                rewritten_query = query # clean(query)
                rewrite_type = "none"

        # 3. Validation
        is_valid_query = True
        rejection_phrases = ["the report states", "it mentions", "the document says", "this report", "the document mentions", "according to"]
        if any(p in rewritten_query.lower() for p in rejection_phrases):
            is_valid_query = False
            rewritten_query = f"{query} {anchor}" if anchor != "none" else query
            rewrite_type = "fallback_validation"

        # FIX 2: Store State (Subject of current rewritten query)
        self.last_semantic_query = self._extract_subject(rewritten_query) or rewritten_query

        # Logging MANDATORY (FIX 5)
        print(f"\n[QueryUnderstanding]")
        print(f"raw: \"{query}\"")
        print(f"dependency: {dep_type}")
        print(f"needs_context: {needs_context}")
        print(f"semantic_base: <{semantic_base}>")
        print(f"rewritten: \"{rewritten_query}\"")
        print(f"final_query: <{rewritten_query}>")
        print(f"rewrite_type: {rewrite_type}")
        print(f"anchor_used: {anchor}")
        print(f"is_valid_query: {is_valid_query}\n")

        return RewriteResult(
            rewritten_query=rewritten_query,
            used_history=used_history,
            confidence=0.7 if used_history else 1.0,
            needs_clarification=False,
            rewrite_type=rewrite_type,
            history_turns_used=turns_used,
            query_type="follow_up" if needs_context else "standalone",
            anchor_used=anchor,
            substitution_success=substitution_success,
            needs_context=needs_context,
            dependency_type=dep_type
        )
