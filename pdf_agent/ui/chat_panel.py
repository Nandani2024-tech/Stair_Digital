# ui/chat_panel.py
import streamlit as st
from models import (
    ConversationTurn, AgentResponse, ResponseType,
    Citation, RetrievalHit, Chunk
)
import time
from retrieval.searcher import search_document, RetrievalResult
from retrieval.hallucination_gate import evaluate_retrieval_gate, evaluate_context_reuse, Gate1Decision
from llm.generator import generate_grounded_answer
from ui.source_preview import render_source_preview
from logs.logger import log_event
from config import TOP_K_RERANK
from retrieval.reranker import Reranker
from conversation.query_rewriter import QueryRewriter
import re

@st.cache_resource
def get_reranker():
    return Reranker()

@st.cache_resource
def get_rewriter():
    return QueryRewriter()

def _render_citation_chips_html(citations: list[Citation]) -> str:
    if not citations:
        return ""
    # FIX 4: Dark mode high-contrast chips
    chips_html = " ".join(
        f'<span style="background-color: #1e293b; color: #e2e8f0; padding: 3px 10px; border-radius: 12px; font-size: 0.85rem; margin-right: 6px; border: 1px solid #334155; display: inline-block; white-space: nowrap; margin-bottom: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">'
        f'{c.render()}</span>'
        for c in citations
    )
    return "<br><br>" + chips_html

def render_chat_panel():
    st.subheader("Chat")
    
    # Global Debug Toggle
    debug_mode = st.sidebar.checkbox("Debug mode: Show hidden sources", value=False, key="global_debug_mode")

    # ── render history ───────────────────────────────────
    for idx, turn in enumerate(st.session_state["chat_history"]):
        with st.chat_message(turn.role):
            if turn.role == "user":
                st.markdown(turn.content)
            else:
                rt = turn.response_type
                if rt == ResponseType.ANSWER:
                    display_content = turn.content + _render_citation_chips_html(turn.citations or [])
                    st.markdown(display_content, unsafe_allow_html=True)
                elif rt == ResponseType.REFUSAL:
                    st.warning(
                        f"**Out of scope**\n\n{turn.content}",
                        icon="🔍",
                    )
                elif rt == ResponseType.ERROR:
                    st.error(
                        f"**Agent error**\n\n{turn.content}",
                        icon="❌",
                    )
                elif rt == ResponseType.CLARIFY:
                    st.warning(f"**Ambiguous Input**\n\n{turn.content}", icon="🤔")
                else:
                    st.markdown(turn.content)
                    
                if getattr(turn, "retrieved_chunks", None):
                    if rt == ResponseType.REFUSAL:
                        if st.session_state.get("global_debug_mode", False):
                            render_source_preview(turn.retrieved_chunks, turn_index=idx)
                    else:
                        render_source_preview(turn.retrieved_chunks, turn_index=idx)

    # ── input ────────────────────────────────────────────
    query = st.chat_input(
        "Ask about the document…",
        disabled=st.session_state["processing"],
    )

    if query is None:
        return

    if not st.session_state["indexed"]:
        st.warning("No document loaded. Please upload a PDF before asking questions.", icon="📄")
        return

    if not query.strip():
        st.warning("Query is empty. Please type a question.", icon="⚠️")
        return

    user_turn = ConversationTurn(role="user", content=query)
    st.session_state["chat_history"].append(user_turn)

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state["processing"] = True
    current_turn_index = len(st.session_state["chat_history"])
    turn_id = f"turn_{current_turn_index:03d}"
    
    try:
        with st.chat_message("assistant"):
            log_event(
                "turn_started",
                turn_id=turn_id,
                doc_id=st.session_state.get("doc_id"),
                raw_query=query[:200],
            )
            
            # 1. Intent & Rewriting
            with st.spinner("Analyzing intent…"):
                rewriter = get_rewriter()
                rewrite_result = rewriter.rewrite(query, st.session_state["chat_history"])
                
                query_type = getattr(rewrite_result, "query_type", "standalone")
                rewritten_query = rewrite_result.rewritten_query
                dependency_type = getattr(rewrite_result, "dependency_type", "independent")
                is_valid_query = getattr(rewrite_result, "is_valid_query", True)
                
                if rewrite_result.needs_clarification:
                    st.warning("**Ambiguous Input**\n\nCould you clarify what you are referring to?", icon="🤔")
                    assistant_turn = ConversationTurn(
                        role="assistant",
                        content="Could you clarify what you are referring to?",
                        response_type=ResponseType.CLARIFY,
                    )
                    st.session_state["chat_history"].append(assistant_turn)
                    st.session_state["last_trace"] = {
                        "query_raw": query,
                        "query_type": query_type,
                        "dependency_type": dependency_type,
                        "needs_clarification": True,
                        "timestamp": time.time(),
                    }
                    st.session_state["turn_traces"].append(st.session_state["last_trace"].copy())
                    log_event("turn_completed", turn_id=turn_id, response_type="clarify")
                    return

            # 2. Retrieval Decision (Context Reuse vs. Fresh Search)
            with st.spinner("Retrieving relevant chunks…"):
                reuse_decision = {"reuse_context": False, "confidence": 0.0, "reason": "default"}
                
                if query_type == "follow_up" and st.session_state.get("last_retrievals"):
                    reuse_decision = evaluate_context_reuse(rewritten_query, st.session_state["last_retrievals"])
                
                if reuse_decision["reuse_context"]:
                    search_result = RetrievalResult(
                        success=True, 
                        query=rewritten_query, 
                        doc_id=st.session_state["doc_id"], 
                        hits=st.session_state["last_retrievals"]
                    )
                    gate_source = "reused_context"
                else:
                    search_result = search_document(rewritten_query, st.session_state["doc_id"])
                    st.session_state["last_retrievals"] = search_result.hits
                    gate_source = "new_retrieval"
                
                # 3. Hallucination Gate (REQUIRED for all paths)
                gate: Gate1Decision = evaluate_retrieval_gate(search_result)
                gate.source = gate_source
                gate.reuse_confidence = reuse_decision["confidence"]

                # Trace event
                trace_event = {
                    "query_raw": query,
                    "query_rewritten": rewritten_query if rewritten_query != query else None,
                    "query_type": query_type,
                    "dependency_type": dependency_type,
                    "is_valid_query": is_valid_query,
                    "context_reuse_decision": reuse_decision["reuse_context"],
                    "reuse_confidence": reuse_decision["confidence"],
                    "reuse_reason": reuse_decision["reason"],
                    "gate1_passed": gate.passed,
                    "gate1_reason": gate.reason,
                    "best_distance": gate.best_distance,
                    "source": gate_source,
                    "timestamp": time.time(),
                }
                
                if gate.passed and gate_source == "new_retrieval":
                    # Rerank only for fresh retrieval
                    reranker = get_reranker()
                    search_result.hits = reranker.rerank(rewritten_query, search_result.hits)[:TOP_K_RERANK]
                    st.session_state["last_retrievals"] = search_result.hits
                
                st.session_state["last_trace"] = trace_event

            # 4. Response Generation
            retrieved_chunks = [
                {
                    "chunk_id": h.chunk_id, "page": h.page_start,
                    "section": h.section_title or "Unknown Section",
                    "score": h.distance, "text": h.text, "rank": h.rank
                } for h in search_result.hits
            ] if search_result.hits else []

            if not gate.passed:
                st.warning(f"**Out of scope**\n\n{gate.reason}", icon="🔍")
                assistant_turn = ConversationTurn(
                    role="assistant", content=gate.reason,
                    response_type=ResponseType.REFUSAL, retrieved_chunks=retrieved_chunks
                )
            else:
                with st.spinner("Generating grounded answer…"):
                    response: AgentResponse = generate_grounded_answer(
                        rewritten_query, search_result.hits, st.session_state["chat_history"]
                    )

                # Update trace
                st.session_state["last_trace"].update({
                    "gate2_passed": getattr(response, "gate2_passed", False),
                    "response_type": response.response_type.value if response.response_type else None,
                    "citations": [c.render() for c in getattr(response, "citations", [])],
                    "refusal_reason": getattr(response, "refusal_reason", None),
                    "error_message": getattr(response, "error_message", None),
                })

                if response.response_type == ResponseType.ANSWER:
                    assistant_turn = ConversationTurn(
                        role="assistant", content=response.answer,
                        response_type=response.response_type, citations=response.citations,
                        retrieved_chunks=retrieved_chunks
                    )
                    st.markdown(response.answer + _render_citation_chips_html(response.citations), unsafe_allow_html=True)
                elif response.response_type == ResponseType.ERROR:
                    st.error(f"**Agent error**\n\n{response.error_message}", icon="❌")
                    assistant_turn = ConversationTurn(
                        role="assistant", content=response.error_message,
                        response_type=ResponseType.ERROR, retrieved_chunks=retrieved_chunks
                    )
                else:
                    # REFUSAL or other
                    st.warning(f"**Out of scope**\n\n{response.refusal_reason}", icon="🔍")
                    assistant_turn = ConversationTurn(
                        role="assistant", content=response.refusal_reason,
                        response_type=ResponseType.REFUSAL, retrieved_chunks=retrieved_chunks
                    )

            if retrieved_chunks and (assistant_turn.response_type != ResponseType.REFUSAL or st.session_state.get("global_debug_mode")):
                render_source_preview(retrieved_chunks, turn_index=current_turn_index)

            st.session_state["chat_history"].append(assistant_turn)
            
            # Persist trace to history
            if "last_trace" in st.session_state:
                st.session_state["turn_traces"].append(st.session_state["last_trace"].copy())
                
            log_event("turn_completed", turn_id=turn_id, response_type=assistant_turn.response_type.value if assistant_turn.response_type else "None")

    finally:
        st.session_state["processing"] = False
