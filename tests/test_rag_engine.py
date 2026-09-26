"""Tests for the advanced RAG pipeline (no API calls)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.documents import Document  # noqa: E402

from hybrid_retriever import reciprocal_rank_fusion  # noqa: E402


def test_rrf_fusion_prefers_top_ranked():
    docs_a = [
        Document(page_content="alpha doc", metadata={"source": "a"}),
        Document(page_content="beta doc", metadata={"source": "b"}),
    ]
    docs_b = [
        Document(page_content="beta doc", metadata={"source": "b"}),
        Document(page_content="gamma doc", metadata={"source": "c"}),
    ]
    fused = reciprocal_rank_fusion([docs_a, docs_b])
    contents = [d.page_content for d in fused]
    # beta appears in both lists -> should rank first
    assert contents[0] == "beta doc"
    assert set(contents) == {"alpha doc", "beta doc", "gamma doc"}


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
