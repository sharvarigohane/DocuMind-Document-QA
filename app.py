"""
RAG App — Streamlit UI
-----------------------
This file is the front-end. It lets the user:
1. Upload a PDF or TXT file
2. Ask questions about it
3. See the AI's answer
"""

import os
import tempfile
import streamlit as st
from rag import build_index, ask

st.set_page_config(page_title="AI Document Assistant", page_icon="📄")

st.title("📄 Ask Your Document")
st.write("Upload a PDF or TXT file, then ask questions about it!")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "txt"],
    help="Upload a PDF or plain text file",
)

if uploaded_file is not None:
    # Only rebuild the index when a NEW file is uploaded.
    # The QA chain lives in session_state (not a global) so each browser tab is independent.
    if "indexed_file" not in st.session_state or st.session_state.indexed_file != uploaded_file.name:

        with st.spinner("📚 Reading and indexing your document... (this takes a moment)"):
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                # build_index() returns the chain; we store it in session_state
                st.session_state.qa_chain = build_index(tmp_path)
            except Exception as e:
                st.error(f"❌ Failed to index document: {e}")
                st.stop()
            finally:
                # Always delete the temp file — even if indexing failed
                os.unlink(tmp_path)

        st.session_state.indexed_file = uploaded_file.name
        st.success(f"✅ '{uploaded_file.name}' indexed successfully!")

    # --- Question input ---
    question = st.text_input(
        "💬 Ask a question about your document",
        placeholder="e.g. What is the main topic of this document?",
    )

    if question:
        with st.spinner("🤔 Thinking..."):
            # Pass the chain from session_state — no global variable needed
            answer = ask(st.session_state.qa_chain, question)

        st.write("### 📝 Answer")
        st.write(answer)
