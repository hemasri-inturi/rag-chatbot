"""
Cross-encoder reranker: scores (query, chunk) pairs jointly for
higher-precision ranking than bi-encoder similarity alone.
"""

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class Reranker:
    """Rerank retrieved chunks with a cross-encoder model."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, documents: List[Document], top_k: int = 4
    ) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )
        return [doc for doc, _ in ranked[:top_k]]
