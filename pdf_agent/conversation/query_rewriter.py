import re
import os
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
    intent: str = "factual"
    anchor_used: str = "none"
    substitution_success: bool = False

class QueryRewriter:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self._client = None
        
    @property
    def client(self):
        if self._client is None and self.api_key:
            self._client = Groq(api_key=self.api_key)
        return self._client
        
    def _extract_subject(self, query: str) -> str:
        stop_phrases = [
            "what does the document say about ", "what is ", "explain ", 
            "tell me about ", "describe ", "can you explain ", 
            "please explain ", "what about "
        ]
        q = query.strip()
        q_lower = q.lower()
        for phrase in stop_phrases:
            if q_lower.startswith(phrase):
                q = q[len(phrase):].strip()
                break
        
        q = q.rstrip(' ?.')
        return q if len(q.split()) >= 1 else None
        
    def _get_cleaned_words(self, text: str) -> set:
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return set(clean.split())

    def _extract_context(self, history: List[ConversationTurn]) -> tuple[str, str, str, int]:
        user_turns = []
        assistant_turn = None
        
        # We index purely safely across histories
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

    def rewrite(self, query: str, history: List[ConversationTurn]) -> RewriteResult:
        q_lower = query.lower().strip()
        words = q_lower.split()
        word_count = len(words)
        
        intent_mode = "factual"
        clarification_phrases = ["what about it", "explain this more", "and then"]
        expand_phrases = ["explain more", "go deeper", "elaborate", "tell me more"]
        reasoning_phrases = ["why", "importance", "impact", "significance", "effect", "meaning", "cause", "how"]
        
        if any(p in q_lower for p in expand_phrases):
             intent_mode = "followup_expand"
        elif any(p in q_lower for p in reasoning_phrases):
             intent_mode = "reasoning"
        elif any(p in q_lower for p in clarification_phrases):
             intent_mode = "clarification"

        if not history:
            return RewriteResult(query, False, 1.0, False, "none", 0, intent=intent_mode)
            
        q_lower = query.lower().strip()
        words = q_lower.split()
        word_count = len(words)
        
        context_str, current_user_query, previous_user_query, turns_used = self._extract_context(history)
        
        anchor = None
        if previous_user_query:
            extracted = self._extract_subject(previous_user_query)
            if extracted and extracted.lower() not in ["this", "topic", "thing", "it", "that", ""]:
                anchor = extracted
                
        if anchor and len(anchor.split()) < 1:
            anchor = None
        
        if word_count < 6 and not anchor and intent_mode == "factual":
             intent_mode = "clarification"
             
        needs_clarification = (intent_mode == "clarification")
        
        if needs_clarification and not anchor:
             return RewriteResult(query, False, 0.0, True, "clarification", 0, intent="clarification")

        # Fall out if history lacks referenceable tokens
        if not previous_user_query or not anchor:
             return RewriteResult(query, False, 1.0, False, "none", 0, intent=intent_mode)
             
        # 2. Intent Classification - Semantic Substitution
        pronoun_match = re.search(r'\b(it|this|that|there)\b', q_lower)
        
        if pronoun_match and len(anchor.split()) >= 1:
            def _replacer(match):
                word = match.group(0).lower()
                if word == "there":
                    return f"related to {anchor}"
                return anchor
                
            new_query, count = re.subn(r'\b(it|this|that|there)\b', _replacer, query, count=1, flags=re.IGNORECASE)
            
            if count > 0:
                 return RewriteResult(
                     new_query, True, 0.8, False, "heuristic", turns_used,
                     intent=intent_mode, anchor_used=anchor, substitution_success=True
                 )

        # 3. Intent Classification - LLM Rewrite (Controlled)
        if self.client and pronoun_match:
            prompt = f"""Rewrite the query to be fully self-contained using only conversation context. Do NOT add new facts.
If it is already self-contained, return the original query exactly.

Conversation Context:
{context_str}

Current Query: {query}
Standalone Query:"""

            try:
                res = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=60
                )
                rewritten = res.choices[0].message.content.strip()
                
                orig_words = self._get_cleaned_words(query)
                rewritten_words = self._get_cleaned_words(rewritten)
                
                if len(rewritten_words) <= 2 * len(orig_words) + 10:
                    return RewriteResult(rewritten, True, 0.6, False, "llm", turns_used, intent=intent_mode, anchor_used=(anchor or "none"))
            except Exception as e:
                print(f"[QueryRewriter] LLM Error: {e}")
                
        # 4. Fallback (Standalone Native)
        return RewriteResult(query, False, 1.0, False, "none", turns_used, intent=intent_mode, anchor_used=(anchor or "none"))
