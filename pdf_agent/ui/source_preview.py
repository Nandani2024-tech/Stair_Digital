import streamlit as st

def render_source_preview(chunks: list[dict], turn_index: int = 0) -> None:
    """Renders retrieving chunks grouped by top 3 and others."""
    if not chunks:
        st.info("No source chunks available for this response.")
        return

    top_chunks = chunks[:3]
    other_chunks = chunks[3:]

    # Top 3 chunks
    for i, chunk in enumerate(top_chunks):
        rank = chunk.get('rank', i + 1)
        page = chunk.get('page', 'Unknown')
        score = chunk.get('score', 0.0)
        label = f"Rank {rank} | Page {page} | Score: {score:.2f}"
        
        with st.expander(label):
            st.caption(
                f"Section: {chunk.get('section', 'Unknown Section')} | "
                f"Chunk ID: {chunk.get('chunk_id', 'Unknown')}"
            )
            text = chunk.get('text', '')
            
            show_full = st.checkbox("Show Full Text", key=f"show_full_{turn_index}_{chunk.get('chunk_id', i)}_{i}")
            if show_full:
                display_text = text
            else:
                display_text = text[:200] + "..." if len(text) > 200 else text
            
            st.code(display_text, language=None)

    # Remaining chunks
    if other_chunks:
        with st.expander(f"Show all other sources ({len(other_chunks)} chunks)"):
            for j, chunk in enumerate(other_chunks):
                rank = chunk.get('rank', len(top_chunks) + j + 1)
                st.caption(
                    f"**Rank {rank}** | Page {chunk.get('page', 'Unknown')} | "
                    f"Score: {chunk.get('score', 0.0):.2f} | "
                    f"Section: {chunk.get('section', 'Unknown Section')} | "
                    f"Chunk ID: {chunk.get('chunk_id', 'Unknown')}"
                )
            st.caption("_(full text available in debug mode or full search)_")


