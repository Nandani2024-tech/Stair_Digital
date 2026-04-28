import streamlit as st

def _gate_badge(passed: bool | None, label: str):
    if passed is True:
        st.success(f"{label}: PASS", icon="✅")
    elif passed is False:
        st.error(f"{label}: FAIL — LLM was not called", icon="🚫")
    else:
        st.caption(f"{label}: not evaluated")

def render_trace_panel():
    with st.sidebar:
        st.markdown("### Turn trace")
        st.caption("Live diagnostic for last query")
        st.markdown("---")

        trace = st.session_state.get("last_trace")

        if not trace:
            st.info("No turn processed yet.", icon="🔎")
            return

        # Query
        st.markdown("**Raw query**")
        st.code(trace.get("query_raw") or "—", language=None)

        if trace.get("query_rewritten"):
            st.markdown("**Rewritten query**")
            st.code(trace["query_rewritten"], language=None)
            st.caption("↑ follow-up resolved using conversation history")

        st.markdown("---")

        # Gates
        st.markdown("**Hallucination gates**")
        _gate_badge(trace.get("gate1_passed"), "Gate 1 — retrieval threshold")
        _gate_badge(trace.get("gate2_passed"), "Gate 2 — LLM citation check")

        st.markdown("---")

        # Response type
        rt = trace.get("response_type")
        st.markdown("**Response type**")
        if rt == "answer":
            st.success("ANSWER — grounded response sent", icon="💬")
        elif rt == "refusal":
            st.warning("REFUSAL — query out of scope", icon="🔍")
        elif rt == "error":
            st.error("ERROR — pipeline failure", icon="❌")
        else:
            st.caption("—")

        # Citations
        citations = trace.get("citations", [])
        st.markdown("**Citations used**")
        if citations:
            for c in citations:
                st.markdown(f"- `{c}`")
        else:
            st.caption("None")

        # Refusal / error details
        if trace.get("refusal_reason"):
            st.markdown("---")
            st.markdown("**Refusal reason**")
            st.warning(trace["refusal_reason"], icon="ℹ️")

        if trace.get("error_message"):
            st.markdown("---")
            st.markdown("**Error detail**")
            st.error(trace["error_message"], icon="❌")

        st.markdown("---")
        st.caption("Retrieval scores, rerank ranks, and chunk IDs appear here from Phase 6 onward.")
