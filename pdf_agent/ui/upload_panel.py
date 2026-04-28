import streamlit as st
import hashlib, time
from config import MAX_PDF_SIZE_MB, UPLOAD_DIR, SESSION_RESET_ON_NEW_PDF, SCANNED_TEXT_MIN_CHARS_PER_PAGE
from models import ResponseType
from ingestion.pipeline import run_ingestion_pipeline, PipelineResult
from indexing.index_builder import build_index, IndexingResult

def _validate_pdf(file) -> tuple[bool, str]:
    """Returns (is_valid, error_message)."""
    if not file.name.lower().endswith(".pdf"):
        return False, (
            f"'{file.name}' is not a PDF file. "
            "Only .pdf files are accepted."
        )
    size_mb = file.size / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        return False, (
            f"File is {size_mb:.1f} MB — exceeds the {MAX_PDF_SIZE_MB} MB limit. "
            "Please compress or split the PDF and re-upload."
        )
    return True, ""


def render_upload_panel():
    st.subheader("Document")

    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help=f"Max size: {MAX_PDF_SIZE_MB} MB",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        valid, err_msg = _validate_pdf(uploaded)

        if not valid:
            st.error(
                f"**Upload rejected**\n\n{err_msg}",
                icon="🚫",
            )
            return

        # New PDF uploaded — reset session if one was already loaded
        if (
            SESSION_RESET_ON_NEW_PDF
            and st.session_state["doc_id"] is not None
            and st.session_state["doc_name"] != uploaded.name
        ):
            st.warning(
                f"New document detected. Session reset — "
                f"conversation history for **{st.session_state['doc_name']}** cleared.",
                icon="⚠️",
            )
            st.session_state["chat_history"]    = []
            st.session_state["last_retrievals"] = []
            st.session_state["last_trace"]      = []
            st.session_state["indexed"]         = False
            st.session_state["ingestion_error"] = None

        # Only re-ingest if not already indexed under same name
        if not st.session_state["indexed"]:
            st.session_state["ingestion_status"] = "Ingesting..."
            st.session_state["doc_name"]         = uploaded.name

            with st.spinner("Parsing PDF — extracting pages and blocks…"):
                result: PipelineResult = run_ingestion_pipeline(uploaded)

            if not result.success:
                st.session_state["ingestion_error"]  = result.error
                st.session_state["ingestion_status"] = "Failed"
                st.error(
                    f"**Ingestion failed**\n\n"
                    f"**Reason:** {result.error}\n\n"
                    "What to try:\n"
                    "- If password-protected: unlock the PDF first\n"
                    "- If corrupted: re-export from the source app\n"
                    "- If scanned: OCR support is coming in Phase 13",
                    icon="❌",
                )
                return

            # Step 2: Build Index (Phase 5)
            with st.spinner("Building vector index — embedding chunks…"):
                index_result: IndexingResult = build_index(result.chunks)

            if not index_result.success:
                st.session_state["ingestion_error"]  = index_result.error
                st.session_state["ingestion_status"] = "Indexing Failed"
                st.session_state["indexed"]          = False
                st.error(
                    f"**Indexing failed**\n\n"
                    f"**Reason:** {index_result.error}\n\n"
                    "Check your configuration and embedding model availability.",
                    icon="❌",
                )
                return

            # Success — store in session
            st.session_state["doc_id"]      = result.doc_id
            st.session_state["indexed"]     = True
            st.session_state["ingestion_status"] = "Ready"
            st.session_state["ingestion_error"]  = None
            st.session_state["page_count"]  = result.page_count
            st.session_state["chunk_count"] = result.chunk_count # replaces "—"
            st.session_state["parsed_pages"] = result.pages  # list[ParsedPage]
            st.session_state["chunks"]      = result.chunks      # list[Chunk]
            st.session_state["total_chars"]  = result.total_chars
            st.session_state["scanned_pages"] = result.scanned_pages

            st.success(
                f"**Ready** — {uploaded.name} parsed and indexed successfully. "
                f"{result.page_count} pages · "
                f"{result.chunk_count} chunks · "
                f"searchable in Chroma.",
                icon="✅",
            )

            # Warn if scanned pages detected
            if result.scanned_pages > 0:
                st.warning(
                    f"**{result.scanned_pages} page(s) appear to be scanned "
                    f"or image-only** (fewer than "
                    f"{SCANNED_TEXT_MIN_CHARS_PER_PAGE} chars extracted). "
                    "These pages may not be searchable until OCR is added in Phase 13. "
                    "Text-based pages will work normally.",
                    icon="🖼️",
                )

    # ── document status card ──────────────────────────────
    if st.session_state["indexed"]:
        st.markdown("---")
        st.markdown("**Loaded document**")
        st.caption(st.session_state["doc_name"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Pages",   st.session_state.get("page_count",   "—"))
        c2.metric("Chunks",  st.session_state.get("chunk_count",  "—"))
        c3.metric("Scanned", st.session_state.get("scanned_pages","—"))
    elif st.session_state["ingestion_error"]:
        st.markdown("---")
        st.error(
            f"Last error: {st.session_state['ingestion_error']}",
            icon="❌",
        )
    else:
        st.info(
            "No document loaded. Upload a PDF to begin.",
            icon="📄",
        )
