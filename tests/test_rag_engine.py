"""Basic tests for the RAG engine (no API calls)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.documents import Document  # noqa: E402


def test_format_docs():
    from rag_engine import RAGEngine

    # Build engine without hitting the OpenAI API: patch ChatOpenAI init
    import rag_engine as re

    orig = re.ChatOpenAI

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

    re.ChatOpenAI = FakeLLM
    try:
        engine = RAGEngine.__new__(RAGEngine)
        docs = [
            Document(page_content="Hello world", metadata={"source": "a.txt"}),
            Document(page_content="Foo bar", metadata={"source": "b.txt"}),
        ]
        formatted = engine._format_docs(docs)
        assert "Hello world" in formatted
        assert "Foo bar" in formatted
    finally:
        re.ChatOpenAI = orig
