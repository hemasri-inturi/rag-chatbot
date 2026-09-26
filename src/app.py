"""
RAG Chatbot - Streamlit UI.

Run with: streamlit run src/app.py
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag_engine import RAGEngine  # noqa: E402

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about your documents — answers come with sources.")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")


@st.cache_resource
def get_engine() -> RAGEngine:
    engine = RAGEngine()
    if os.path.exists(INDEX_DIR):
        engine.load_index(INDEX_DIR)
    else:
        with st.spinner("Indexing documents... (first run only)"):
            docs = engine.load_documents(DATA_DIR)
            n_chunks = engine.build_index(docs)
            engine.save_index(INDEX_DIR)
            st.success(f"Indexed {len(docs)} documents → {n_chunks} chunks.")
    return engine


if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ Set your `OPENAI_API_KEY` in a `.env` file to use the chatbot.")
    st.stop()

engine = get_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.caption(f"{s['source']}" + (f" — page {s['page']}" if s.get("page") is not None else ""))

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = engine.ask(question)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📚 Sources"):
                for s in result["sources"]:
                    st.caption(f"{s['source']}" + (f" — page {s['page']}" if s.get("page") is not None else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
