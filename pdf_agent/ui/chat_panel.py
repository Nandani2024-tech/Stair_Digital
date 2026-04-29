import streamlit as st
from models import (
    ConversationTurn, AgentResponse, ResponseType,
    Citation, RetrievalHit, Chunk
)
import time
from retrieval.searcher import search_document, RetrievalResult
from retrieval.hallucination_gate import evaluate_retrieval_gate, Gate1Decision
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

def _stub_agent_response(query: str, history: list) -> AgentResponse:
    """
    STUB — simulates agent pipeline result.
    Replace entirely in Phase 8.
    Demonstrates all three response types for UI testing.
    """
    time.sleep(0.8)
    q = query.lower()

    if any(w in q for w in ["error", "crash", "fail"]):
        return AgentResponse(
            response_type=ResponseType.ERROR,
            answer=None,
            error_message=(
                "Simulated error: LLM API timeout after 3 retries. "
                "Check your GROQ_API_KEY and network connection."
            ),
        )
    if any(w in q for w in ["president", "weather", "stock", "bitcoin"]):
        return AgentResponse(
            response_type=ResponseType.REFUSAL,
            answer=None,
            refusal_reason=(
                "No relevant section found in the uploaded PDF "
                "(best match score: 0.18, threshold: 0.35). "
                "This topic does not appear to be covered in the document."
            ),
            gate1_passed=False,
        )
    stub_chunk = Chunk(
        chunk_id="stub_chunk_001",
        text="This is stub retrieved content.",
        page_start=3, page_end=4,
        section_title="Introduction",
        doc_id="stub_doc",
        token_count=42,
    )
    return AgentResponse(
        response_type=ResponseType.ANSWER,
        answer=(
            "This is a stub answer grounded in the document. "
            "In Phase 8 this will be replaced with a real "
            "LLM-generated response citing the retrieved chunks."
        ),
        citations=[
            Citation(page=3, section="Introduction",  chunk_id="stub_chunk_001"),
            Citation(page=4, section="Background",    chunk_id="stub_chunk_002"),
        ],
        gate1_passed=True,
        gate2_passed=True,
        rewritten_query=query if len(query) < 20 else None,
    )

def _render_citation_chips_html(citations: list[Citation]) -> str:
    if not citations:
        return ""
    chips_html = " ".join(
        f'<span style="background-color: #f0f2f6; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-right: 5px; border: 1px solid #dfe1e5;">'
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
                else:
                    st.markdown(turn.content)
                    
                # Always safely render source preview based on turn type
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

    # Guard: no document
    if not st.session_state["indexed"]:
        st.warning(
            "No document loaded. Please upload a PDF before asking questions.",
            icon="📄",
        )
        return

    # Guard: empty string (shouldn't happen with chat_input, but be safe)
    if not query.strip():
        st.warning("Query is empty. Please type a question.", icon="⚠️")
        return

    # Append user turn
    user_turn = ConversationTurn(role="user", content=query)
    st.session_state["chat_history"].append(user_turn)

    with st.chat_message("user"):
        st.markdown(query)

    # Process
    st.session_state["processing"] = True
    current_turn_index = len(st.session_state["chat_history"])
    turn_id = f"turn_{current_turn_index:03d}"
    
    try:
        with st.chat_message("assistant"):
            log_event(
                "turn_started",
                turn_id=turn_id,
                doc_id=st.session_state.get("doc_id"),
                raw_query=query[:200] + "..." if len(query) > 203 else query,
            )
            
            # Phase 8: Intercept with Deterministic/Semantic Rewriting
            with st.spinner("Analyzing intent…"):
                rewriter = get_rewriter()
                rewrite_result = rewriter.rewrite(query, st.session_state["chat_history"])
                
                intent = getattr(rewrite_result, "intent", "factual")
                is_continuation, context_reused, safe_reuse_passed = False, False, False

                # Handle Continuation Branching Safely
                if intent == "followup_expand":
                    if st.session_state.get("last_retrievals"):
                         is_continuation = context_reused = safe_reuse_passed = True
                    else:
                         rewrite_result.needs_clarification = True
                elif intent == "reasoning":
                    safe_reuse_passed = safe_to_reuse_context(query, rewrite_result.anchor_used, st.session_state)
                    if safe_reuse_passed:
                         is_continuation = context_reused = True
                
                if rewrite_result.needs_clarification:
                    log_event(
                        "query_rewritten",
                        turn_id=turn_id,
                        raw_query=query[:200] + "...",
                        rewritten_query=query,
                        rewrite_type=rewrite_result.rewrite_type,
                        rewrite_confidence=rewrite_result.confidence,
                        history_turns_used=rewrite_result.history_turns_used,
                        anchor_used=rewrite_result.anchor_used,
                        substitution_success=rewrite_result.substitution_success,
                        intent=intent,
                        is_continuation=is_continuation,
                        context_reused=context_reused,
                        safe_reuse_passed=safe_reuse_passed
                    )
                    st.warning("**Ambiguous Input**\n\nCould you clarify what you are referring to?", icon="🤔")
                    
                    assistant_turn = ConversationTurn(
                        role="assistant",
                        content="Could you clarify what you are referring to?",
                        response_type=ResponseType.CLARIFY,
                    )
                    st.session_state["chat_history"].append(assistant_turn)
                    st.session_state["last_trace"] = {
                        "query_raw": query,
                        "query_rewritten": None,
                        "needs_clarification": True,
                        "timestamp": time.time(),
                        "intent": intent
                    }
                    log_event("turn_completed", turn_id=turn_id, doc_id=st.session_state.get("doc_id"), response_type="clarify")
                    return
                
                rewritten_query = rewrite_result.rewritten_query
                log_event(
                    "query_rewritten",
                    turn_id=turn_id,
                    raw_query=query[:200] + "..." if len(query) > 203 else query,
                    rewritten_query=rewritten_query[:200] + "..." if len(rewritten_query) > 203 else rewritten_query,
                    rewrite_type=rewrite_result.rewrite_type,
                    rewrite_confidence=rewrite_result.confidence,
                    history_turns_used=rewrite_result.history_turns_used,
                    anchor_used=rewrite_result.anchor_used,
                    substitution_success=rewrite_result.substitution_success,
                    intent=intent,
                    is_continuation=is_continuation,
                    context_reused=context_reused,
                    safe_reuse_passed=safe_reuse_passed
                )
            
            with st.spinner("Retrieving relevant chunks…"):
                # Real Retrieval (Phase 6)
                if context_reused:
                     search_result = RetrievalResult(
                          success=True, 
                          query=rewritten_query, 
                          doc_id=st.session_state["doc_id"], 
                          hits=st.session_state["last_retrievals"],
                          error=None
                     )
                     gate: Gate1Decision = evaluate_retrieval_gate(search_result, is_continuation=True)
                else:
                     search_result: RetrievalResult = search_document(rewritten_query, st.session_state["doc_id"])
                     st.session_state["last_retrievals"] = search_result.hits
                     gate: Gate1Decision = evaluate_retrieval_gate(search_result, is_continuation=False)
                
                # Trace event for observability
                trace_event = {
                    "query_raw": query,
                    "query_rewritten": rewritten_query if rewritten_query != query else None,
                    "hits_found": len(search_result.hits),
                    "timestamp": time.time(),
                    "gate1_passed": gate.passed,
                    "gate1_reason": gate.reason,
                    "threshold": gate.threshold,
                    "best_distance": gate.best_distance,
                }
                if gate.passed:
                    trace_event.update({
                        "top_hit_page": gate.top_hit_page,
                        "top_hit_section": gate.top_hit_section,
                    })
                    
                    # Phase 6: Reranker execution explicitly requested AFTER passing Gate 1 logic
                    reranker = get_reranker()
                    search_result.hits = reranker.rerank(rewritten_query, search_result.hits)[:TOP_K_RERANK]
                    st.session_state["last_retrievals"] = search_result.hits
                
                st.session_state["last_trace"] = trace_event

            # Display retrieval hits via sidebar mode and prepare chunks
            retrieved_chunks = []
            if search_result.success and search_result.hits:
                retrieved_chunks = [
                    {
                        "chunk_id": getattr(hit, "chunk_id", "Unknown"),
                        "page": getattr(hit, "page_start", "Unknown"),
                        "section": getattr(hit, "section_title", None) or "Unknown Section",
                        "score": getattr(hit, "diagnostic_similarity", getattr(hit, "distance", 0.0)),
                        "text": getattr(hit, "text", ""),
                        "rank": getattr(hit, "rank", i + 1)
                    }
                    for i, hit in enumerate(search_result.hits)
                ]
                
                log_event(
                    "retrieval_completed",
                    turn_id=turn_id,
                    doc_id=st.session_state.get("doc_id"),
                    top_k=len(retrieved_chunks),
                    retrieval_hits=[
                        {
                            "chunk_id": chunk["chunk_id"],
                            "page": chunk["page"],
                            "section": chunk["section"],
                            "score": chunk["score"]
                        }
                        for chunk in retrieved_chunks
                    ],
                )
            elif not search_result.success:
                st.error(f"Retrieval failed: {search_result.error}")

            log_event(
                "gate_decision",
                turn_id=turn_id,
                gate_passed=gate.passed,
                gate_reason=gate.reason,
                top_score=gate.best_distance,
                threshold=gate.threshold,
            )

            # Gate 1 Logic: If failed, refuse. If passed, continue to stub.
            if not gate.passed:
                log_event("llm_skipped", turn_id=turn_id, reason="retrieval_gate_failed")
                st.warning(f"**Out of scope**\n\n{gate.reason}", icon="🔍")
                if retrieved_chunks and st.session_state.get("global_debug_mode", False):
                    render_source_preview(retrieved_chunks, turn_index=current_turn_index)
                assistant_turn = ConversationTurn(
                    role="assistant",
                    content=gate.reason,
                    response_type=ResponseType.REFUSAL,
                    retrieved_chunks=retrieved_chunks
                )
            else:
                log_event(
                    "llm_called",
                    turn_id=turn_id,
                    context_chunk_ids=[chunk["chunk_id"] for chunk in retrieved_chunks],
                    context_pages=sorted(list({chunk["page"] for chunk in retrieved_chunks if isinstance(chunk.get("page"), int)})),
                    context_char_count=sum(len(chunk.get("text", "")) for chunk in retrieved_chunks),
                )
                
                # Real LLM Generation (Phase 8)
                with st.spinner("Generating grounded answer…"):
                    response: AgentResponse = generate_grounded_answer(
                        rewritten_query, search_result.hits, st.session_state["chat_history"]
                    )

                # Sync Trace Panel metadata
                st.session_state["last_trace"].update({
                    "gate2_passed": getattr(response, "gate2_passed", None),
                    "response_type": response.response_type.value if response.response_type else None,
                    "citations": [c.render() for c in getattr(response, "citations", [])],
                    "refusal_reason": getattr(response, "refusal_reason", None),
                    "error_message": getattr(response, "error_message", None),
                    "query_rewritten": getattr(response, "rewritten_query", None),
                })

                # Render response
                if response.response_type == ResponseType.ANSWER:
                    assistant_turn = ConversationTurn(
                        role="assistant",
                        content=response.answer,
                        response_type=response.response_type,
                        citations=response.citations,
                        retrieved_chunks=retrieved_chunks
                    )
                    display_content = response.answer + _render_citation_chips_html(response.citations or [])
                    st.markdown(display_content, unsafe_allow_html=True)
                    
                    if retrieved_chunks:
                        render_source_preview(retrieved_chunks, turn_index=current_turn_index)

                elif response.response_type == ResponseType.REFUSAL:
                    st.warning(
                        f"**Out of scope**\n\n{response.refusal_reason}",
                        icon="🔍",
                    )
                    if retrieved_chunks and st.session_state.get("global_debug_mode", False):
                        render_source_preview(retrieved_chunks, turn_index=current_turn_index)
                    assistant_turn = ConversationTurn(
                        role="assistant",
                        content=response.refusal_reason,
                        response_type=ResponseType.REFUSAL,
                        retrieved_chunks=retrieved_chunks
                    )

                elif response.response_type == ResponseType.ERROR:
                    st.error(
                        f"**Agent error**\n\n{response.error_message}\n\n"
                        "Check the trace panel for details.",
                        icon="❌",
                    )
                    assistant_turn = ConversationTurn(
                        role="assistant",
                        content=response.error_message,
                        response_type=ResponseType.ERROR,
                        retrieved_chunks=retrieved_chunks
                    )
                else:
                    st.markdown("_(no response)_")
                    assistant_turn = ConversationTurn(
                        role="assistant", content="", response_type=None,
                        retrieved_chunks=retrieved_chunks
                    )
                    
                # Response Logging
                citation_pages = []
                citation_sections = []
                preview = ""
                if response.response_type == ResponseType.ANSWER:
                    citation_pages = [getattr(c, "page", None) for c in response.citations]
                    citation_sections = [getattr(c, "section", None) for c in response.citations]
                    preview = response.answer[:160] if response.answer else ""
                elif response.response_type == ResponseType.REFUSAL:
                    preview = response.refusal_reason[:160] if response.refusal_reason else ""
                elif response.response_type == ResponseType.ERROR:
                    preview = response.error_message[:160] if response.error_message else ""
                
                log_event(
                    "response_generated",
                    turn_id=turn_id,
                    response_type=response.response_type.value,
                    citation_pages=citation_pages,
                    citation_sections=citation_sections,
                    answer_preview=preview,
                )

            st.session_state["chat_history"].append(assistant_turn)
            
            log_event(
                "turn_completed",
                turn_id=turn_id,
                doc_id=st.session_state.get("doc_id"),
                response_type=assistant_turn.response_type.value if assistant_turn.response_type else "None"
            )

    finally:
        st.session_state["processing"] = False
