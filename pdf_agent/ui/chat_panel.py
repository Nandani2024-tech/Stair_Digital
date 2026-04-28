import streamlit as st
from models import (
    ConversationTurn, AgentResponse, ResponseType,
    Citation, RetrievalHit, Chunk
)
import time

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

def _render_citation_chips(citations: list[Citation]):
    chips_html = " ".join(
        f'<span style="background-color: #f0f2f6; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-right: 5px; border: 1px solid #dfe1e5;">'
        f'{c.render()}</span>'
        for c in citations
    )
    st.markdown(chips_html, unsafe_allow_html=True)

def render_chat_panel():
    st.subheader("Chat")

    # ── render history ───────────────────────────────────
    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn.role):
            if turn.role == "user":
                st.markdown(turn.content)
            else:
                rt = turn.response_type
                if rt == ResponseType.ANSWER:
                    st.markdown(turn.content)
                    if turn.citations:
                        _render_citation_chips(turn.citations)
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
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response: AgentResponse = _stub_agent_response(
                query, st.session_state["chat_history"]
            )

        # Build trace event for this turn
        trace_event = {
            "query_raw":       query,
            "query_rewritten": response.rewritten_query,
            "gate1_passed":    response.gate1_passed,
            "gate2_passed":    response.gate2_passed,
            "response_type":   response.response_type.value if response.response_type else None,
            "citations":       [c.render() for c in response.citations],
            "refusal_reason":  response.refusal_reason,
            "error_message":   response.error_message,
        }
        st.session_state["last_trace"] = trace_event

        # Render response
        if response.response_type == ResponseType.ANSWER:
            st.markdown(response.answer)
            if response.citations:
                _render_citation_chips(response.citations)
            assistant_turn = ConversationTurn(
                role="assistant",
                content=response.answer,
                citations=response.citations,
                response_type=ResponseType.ANSWER,
            )

        elif response.response_type == ResponseType.REFUSAL:
            st.warning(
                f"**Out of scope**\n\n{response.refusal_reason}",
                icon="🔍",
            )
            assistant_turn = ConversationTurn(
                role="assistant",
                content=response.refusal_reason,
                response_type=ResponseType.REFUSAL,
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
            )
        else:
            st.markdown("_(no response)_")
            assistant_turn = ConversationTurn(
                role="assistant", content="", response_type=None
            )

        st.session_state["chat_history"].append(assistant_turn)

    st.session_state["processing"] = False
