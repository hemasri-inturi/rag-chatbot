"""
Hybrid retriever: combines dense (FAISS) and sparse (BM25) retrieval
using Reciprocal Rank Fusion (RRF).
"""

from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], k: int = 60
) -> List[Document]:
    """Fuse multiple ranked document lists via RRF.

    Score(doc) = sum(1 / (k + rank)) across lists containing the doc.
    """
    fused_scores: dict = {}
    doc_map: dict = {}

    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content  # dedupe key
            doc_map[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    reranked = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [doc_map[key] for key in reranked]


class HybridRetriever:
    """Dense + sparse hybrid retrieval with RRF fusion."""

    def __init__(self, dense_retriever, documents: List[Document]):
        self.dense_retriever = dense_retriever
        self.documents = documents
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def invoke(self, query: str, top_k: int = 10) -> List[Document]:
        dense_docs = self.dense_retriever.invoke(query)

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        sparse_docs = [self.documents[i] for i in ranked_idx]

        fused = reciprocal_rank_fusion([dense_docs, sparse_docs])
        return fused[:top_k]
