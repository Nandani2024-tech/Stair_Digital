# ui/trace_panel.py
import streamlit as st

def _gate_badge(passed: bool | None, label: str):
    if passed is True:
        st.success(f"{label}: PASS", icon="✅")
    elif passed is False:
        # FIX 3: Trace accuracy
        if "Gate 2" in label:
            st.error(f"{label}: FAIL — citation/grounding validation failed", icon="🚫")
        else:
            st.error(f"{label}: FAIL — retrieval relevance threshold not met", icon="🚫")
    else:
        st.caption(f"{label}: not evaluated")

def render_trace_panel():
    with st.sidebar:
        st.markdown("### Turn Trace")
        st.caption("Live diagnostic for last query")
        st.markdown("---")

        trace = st.session_state.get("last_trace")

        if not trace:
            st.info("No turn processed yet.", icon="🔎")
            return

        # 1. Intent & Type
        st.markdown("**Intent Analysis**")
        q_type = trace.get("query_type", "standalone")
        st.write(f"Type: `{q_type.upper()}`")
        
        st.markdown("**Raw query**")
        st.code(trace.get("query_raw") or "—", language=None)

        if trace.get("query_rewritten"):
            st.markdown("**Rewritten query**")
            st.code(trace["query_rewritten"], language=None)
            st.caption("↑ context-aware resolution")

        st.markdown("---")

        # 2. Context Grounding Decision
        st.markdown("**Grounding Decision**")
        if trace.get("context_reuse_decision"):
             st.success(f"CONTEXT REUSED", icon="♻️")
             st.caption(f"Reason: {trace.get('reuse_reason')}")
             st.caption(f"Sem. Conf: `{trace.get('reuse_confidence', 0.0):.4f}`")
        else:
             st.info("FRESH RETRIEVAL", icon="📡")
             st.caption(f"Reason: {trace.get('reuse_reason') or 'source drift'}")

        st.markdown("---")

        # 3. Gates
        st.markdown("**Hallucination Gates**")
        _gate_badge(trace.get("gate1_passed"), "Gate 1 — retrieval barrier")
        _gate_badge(trace.get("gate2_passed"), "Gate 2 — citation validator")

        st.markdown("---")

        # 4. Response Info
        rt = trace.get("response_type")
        st.markdown("**Response Status**")
        if rt == "answer":
            st.success("ANSWER — grounded", icon="💬")
        elif rt == "refusal":
            st.warning("REFUSAL — out of scope", icon="🔍")
        elif rt == "error":
            st.error("ERROR — pipeline failure", icon="❌")
        elif rt == "clarify":
            st.warning("CLARIFY — ambiguous", icon="🤔")
        else:
            st.caption("—")

        # Citations
        citations = trace.get("citations", [])
        if citations:
            st.markdown("**Citations attached**")
            for c in citations:
                st.markdown(f"- `{c}`")

        # Debug details
        if st.session_state.get("global_debug_mode", False):
            st.markdown("---")
            st.markdown("**Advanced Statistics**")
            st.write(f"Source: `{trace.get('source', 'N/A')}`")
            st.write(f"Best Dist: `{trace.get('best_distance', 'N/A')}`")
            if trace.get("hits_found") is not None:
                st.write(f"Hits found: `{trace.get('hits_found')}`")
            if trace.get("best_distance") is not None:
                st.write(f"Best Dist: `{trace.get('best_distance'):.4f}`")
            if trace.get("reuse_confidence"):
                st.write(f"Reuse Conf: `{trace['reuse_confidence']:.4f}`")
            if trace.get("hits_found") is not None:
                st.write(f"Hits found: `{trace.get('hits_found')}`")
