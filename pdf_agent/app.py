import streamlit as st
from config import MAX_HISTORY_TURNS, SESSION_RESET_ON_NEW_PDF
from models import ResponseType

st.set_page_config(
    page_title="PDF Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state contract ──────────────────────────────────
# Every key documented here. Panels READ these; app.py OWNS them.
def _init_state():
    defaults = {
        "doc_id":           None,       # str | None
        "doc_name":         None,       # original filename
        "indexed":          False,      # bool — ingestion complete?
        "ingestion_status": None,       # str | None — human-readable status
        "ingestion_error":  None,       # str | None — full error if failed
        "chat_history":     [],         # list[ConversationTurn]
        "last_retrievals":  [],         # list[RetrievalHit] from last turn
        "last_trace":       {},         # dict — per-turn trace events
        "query_input":      "",         # bound to chat input widget
        "processing":       False,      # bool — spinner guard
        "parsed_pages":     [],         # list[ParsedPage] — set after ingestion
        "chunks":           [],         # list[Chunk] — set after ingestion
        "total_chars":      0,
        "scanned_pages":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── layout ──────────────────────────────────────────────────
from ui.upload_panel import render_upload_panel
from ui.chat_panel   import render_chat_panel
from ui.trace_panel  import render_trace_panel
col_upload, col_chat = st.columns([1, 2], gap="large")

with col_upload:
    render_upload_panel()

with col_chat:
    render_chat_panel()

render_trace_panel()          # sidebar — always visible, executes last to capture latest trace
