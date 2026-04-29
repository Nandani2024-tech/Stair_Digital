import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from copy import deepcopy
from models import ConversationTurn, AgentResponse, ResponseType, Chunk, Citation, RetrievalResult
from retrieval.hallucination_gate import evaluate_retrieval_gate, Gate1Decision
from conversation.query_rewriter import QueryRewriter
import time
import re

logs = []

def log_event(event_type, **kwargs):
    logs.append({"event": event_type, **kwargs})

# Mock state
state = {
    "chat_history": [],
    "last_retrievals": [],
    "last_trace": {},
    "doc_id": "test_doc"
}

def safe_to_reuse_context(query: str, anchor: str, state) -> bool:
    if not anchor or anchor == "none":
        return False
    if not state.get("last_retrievals") or not state.get("last_trace", {}).get("gate1_passed"):
        return False
        
    stopwords = {
        "why", "is", "it", "the", "this", "that", "there", "a", "an", "what", "how", "are", "important", 
        "for", "in", "of", "and", "about", "to", "explain", "more", "tell", 
        "me", "go", "deeper", "elaborate", "impact", "significance"
    }
    
    query_words = set(re.findall(r'\w+', query.lower())) - stopwords
    anchor_words = set(re.findall(r'\w+', anchor.lower())) - stopwords
    
    diff = query_words.difference(anchor_words)
    return len(diff) == 0

rewriter = QueryRewriter()
# mock LLM entirely so we don't need real API key for testing structural routing
# We just want to test intent, routing, context reuse.
rewriter._client = None

def run_turn(query: str, last_retrievals_mock=None):
    if last_retrievals_mock is not None:
        state["last_retrievals"] = last_retrievals_mock
        state["last_trace"] = {"gate1_passed": True}
        
    user_turn = ConversationTurn(role="user", content=query)
    state["chat_history"].append(user_turn)
    
    turn_id = f"turn_{len(state['chat_history']):03d}"
    
    rewrite_result = rewriter.rewrite(query, state["chat_history"])
    intent = getattr(rewrite_result, "intent", "factual")
    is_continuation, context_reused, safe_reuse_passed = False, False, False

    if intent == "followup_expand":
        if state.get("last_retrievals"):
             is_continuation = context_reused = safe_reuse_passed = True
        else:
             rewrite_result.needs_clarification = True
    elif intent == "reasoning":
        safe_reuse_passed = safe_to_reuse_context(query, rewrite_result.anchor_used, state)
        if safe_reuse_passed:
             is_continuation = context_reused = True

    if rewrite_result.needs_clarification:
        log_event(
            "query_rewritten",
            turn_id=turn_id,
            intent=intent,
            is_continuation=is_continuation,
            context_reused=context_reused,
            safe_reuse_passed=safe_reuse_passed,
            anchor_used=rewrite_result.anchor_used,
            clarification_trigger=True
        )
        return
        
    log_event(
        "query_rewritten",
        turn_id=turn_id,
        intent=intent,
        is_continuation=is_continuation,
        context_reused=context_reused,
        safe_reuse_passed=safe_reuse_passed,
        anchor_used=rewrite_result.anchor_used,
        clarification_trigger=False
    )
    
    if context_reused:
         search_result = RetrievalResult(
              success=True, 
              query=rewrite_result.rewritten_query, 
              doc_id=state["doc_id"], 
              hits=state["last_retrievals"],
              error=None
         )
         gate = evaluate_retrieval_gate(search_result, is_continuation=True)
         log_event("gate_evaluated", passed=gate.passed, reason=gate.reason, is_continuation=True)
    else:
         # Fake retrieval
         log_event("retrieval_executed", query=query)
         search_result = RetrievalResult(success=True, query=query, doc_id="test", hits=[], error=None)
         gate = evaluate_retrieval_gate(search_result, is_continuation=False)
         log_event("gate_evaluated", passed=gate.passed, reason=gate.reason, is_continuation=False)

def run_case_1():
    global logs
    logs = []
    print("=== Case 1: Reasoning Fix ===")
    state["chat_history"] = [
        ConversationTurn(role="user", content="Explain GDP growth"),
        ConversationTurn(role="assistant", content="GDP growth is an economic indicator...")
    ]
    # provide a mock last retrieval
    last_retrievals_mock = [Chunk(chunk_id="1", doc_id="doc", text="GDP", page_start=1, page_end=1, section_title="", token_count=10)]
    run_turn("Why is it important?", last_retrievals_mock)
    for l in logs: print(l)

def run_case_2():
    global logs
    logs = []
    print("\n=== Case 2: Follow-up Expand (No prior) ===")
    state["chat_history"] = []
    state["last_retrievals"] = []
    state["last_trace"] = {}
    run_turn("Explain more")
    for l in logs: print(l)
    
    logs = []
    print("\n=== Case 2: Follow-up Expand (With prior) ===")
    state["chat_history"] = [
        ConversationTurn(role="user", content="Explain inflation"),
        ConversationTurn(role="assistant", content="Inflation is...")
    ]
    last_retrievals_mock = [Chunk(chunk_id="1", doc_id="doc", text="Inf", page_start=1, page_end=1, section_title="", token_count=10)]
    run_turn("Explain more", last_retrievals_mock)
    for l in logs: print(l)

def run_case_3():
    global logs
    logs = []
    print("\n=== Case 3: Topic Shift Protection ===")
    state["chat_history"] = [
        ConversationTurn(role="user", content="Explain GDP growth"),
        ConversationTurn(role="assistant", content="GDP growth is...")
    ]
    last_retrievals_mock = [Chunk(chunk_id="1", doc_id="doc", text="GDP", page_start=1, page_end=1, section_title="", token_count=10)]
    run_turn("Why is it important for startups?", last_retrievals_mock)
    for l in logs: print(l)

if __name__ == "__main__":
    run_case_1()
    run_case_2()
    run_case_3()
