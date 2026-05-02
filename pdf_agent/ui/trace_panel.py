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
        st.markdown("## 🧠 Turn Trace History")
        st.caption("Inspect pipeline diagnostics for all turns")
        st.markdown("---")

        traces = st.session_state.get("turn_traces", [])

        if not traces:
            st.info("No turn processed yet.", icon="🔎")
            return

        # ── Highlight Current Turn ─────────────────────────────────
        latest_trace = traces[-1]
        st.markdown("### 🔍 Latest Turn")
        st.markdown(f"**Raw:** `{latest_trace.get('query_raw', '—')}`")
        if latest_trace.get("query_rewritten"):
            st.markdown(f"**Rewritten:** `{latest_trace['query_rewritten']}`")
        st.markdown("---")

        # ── History Container ──────────────────────────────────────
        st.markdown("### 📜 Past Traces (Latest First)")
        
        # Display in reverse chronological order
        for i, trace in enumerate(reversed(traces)):
            turn_num = len(traces) - i
            query_preview = trace.get("query_raw", "Unknown")[:30]
            
            with st.expander(f"Turn {turn_num} — {query_preview}..."):
                # 1. Intent Analysis
                st.markdown("**Intent Analysis**")
                st.write(f"Type: `{trace.get('query_type', 'standalone').upper()}`")
                st.write(f"Dependency: `{trace.get('dependency_type', 'independent')}`")
                
                st.markdown("**Rewritten Query**")
                st.code(trace.get("query_rewritten") or trace.get("query_raw"), language=None)

                st.markdown("---")

                # 2. Grounding Decision
                st.markdown("**Grounding Decision**")
                if trace.get("context_reuse_decision"):
                     st.success("CONTEXT REUSED", icon="♻️")
                else:
                     st.info("FRESH RETRIEVAL", icon="📡")
                st.caption(f"Reason: {trace.get('reuse_reason', 'N/A')}")
                if trace.get("reuse_confidence"):
                    st.write(f"Sem. Conf: `{trace['reuse_confidence']:.4f}`")

                st.markdown("---")

                # 3. Hallucination Gates
                st.markdown("**Hallucination Gates**")
                _gate_badge(trace.get("gate1_passed"), "Gate 1 — retrieval barrier")
                _gate_badge(trace.get("gate2_passed"), "Gate 2 — citation validator")

                st.markdown("---")

                # 4. Response Status
                rt = trace.get("response_type")
                st.markdown("**Response Status**")
                if rt == "answer":
                    st.success("ANSWER — grounded", icon="💬")
                elif rt == "refusal":
                    st.warning("REFUSAL — out of scope", icon="🔍")
                elif rt == "error":
                    st.error("ERROR — internal", icon="❌")
                elif rt == "clarify":
                    st.warning("CLARIFY — ambiguous", icon="🤔")
                else:
                    st.write(f"Status: `{rt or 'N/A'}`")

                # Citations
                citations = trace.get("citations", [])
                if citations:
                    st.markdown("**Citations**")
                    for c in citations:
                        st.write(f"- `{c}`")
                
                # Metadata
                if st.session_state.get("global_debug_mode", False):
                    st.markdown("---")
                    st.markdown("**Advanced Metadata**")
                    st.write(f"Source: `{trace.get('source', 'N/A')}`")
                    if trace.get("best_distance") is not None:
                        st.write(f"Best Dist: `{trace.get('best_distance'):.4f}`")
