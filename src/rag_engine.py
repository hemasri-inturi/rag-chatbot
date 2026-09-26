"""
Advanced RAG Chatbot - Core engine.

Pipeline:
  Query → Expansion → Hybrid retrieval (dense + BM25, RRF fusion)
        → Cross-encoder reranking → Grounded generation → Evaluation
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.evaluation import RAGEvaluator
from src.hybrid_retriever import HybridRetriever
from src.query_expansion import QueryExpander
from src.reranker import Reranker

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question
using ONLY the context provided below. If the answer is not in the context,
say you don't know — do not make up information.

Context:
{context}

Question: {question}

Answer:"""


class AdvancedRAGEngine:
    """Advanced RAG engine: hybrid retrieval + reranking + evaluation."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        retrieval_k: int = 12,
        final_k: int = 4,
        use_query_expansion: bool = True,
        use_reranker: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.final_k = final_k
        self.use_query_expansion = use_query_expansion
        self.use_reranker = use_reranker

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.vectorstore: Optional[FAISS] = None
        self.chunks: List[Document] = []

        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        self.expander = QueryExpander(llm_model=llm_model) if use_query_expansion else None
        self.reranker = Reranker(model_name=reranker_model) if use_reranker else None
        self.evaluator = RAGEvaluator(llm_model=llm_model)

    # ---------- Indexing ----------

    def load_documents(self, data_dir: str) -> List[Document]:
        docs: List[Document] = []
        for path in Path(data_dir).rglob("*"):
            if path.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(path)).load())
            elif path.suffix.lower() == ".txt":
                docs.extend(TextLoader(str(path), encoding="utf-8").load())
        return docs

    def build_index(self, documents: List[Document]) -> int:
        self.chunks = self.splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        return len(self.chunks)

    def save_index(self, index_dir: str) -> None:
        if self.vectorstore is None:
            raise ValueError("No index built yet. Call build_index() first.")
        self.vectorstore.save_local(index_dir)
        # persist chunks for BM25
        import pickle

        with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self, index_dir: str) -> None:
        import pickle

        self.vectorstore = FAISS.load_local(
            index_dir, self.embeddings, allow_dangerous_deserialization=True
        )
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

    # ---------- Retrieval ----------

    def _retrieve(self, question: str) -> List[Document]:
        dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retrieval_k}
        )
        hybrid = HybridRetriever(dense_retriever, self.chunks)

        queries = (
            self.expander.expand(question) if self.expander else [question]
        )
        all_docs: List[Document] = []
        seen = set()
        for q in queries:
            for doc in hybrid.invoke(q, top_k=self.retrieval_k):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_docs.append(doc)

        if self.reranker:
            all_docs = self.reranker.rerank(question, all_docs, top_k=self.final_k)
        else:
            all_docs = all_docs[: self.final_k]
        return all_docs

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    # ---------- Q&A ----------

    def ask(self, question: str, evaluate: bool = False) -> dict:
        if self.vectorstore is None:
            raise ValueError("No index loaded. Build or load an index first.")

        retrieved = self._retrieve(question)

        chain = (
            {
                "context": lambda _: self._format_docs(retrieved),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)

        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None),
            }
            for doc in retrieved
        ]
        result = {"answer": answer, "sources": sources}

        if evaluate:
            chunk_texts = [d.page_content for d in retrieved]
            result["metrics"] = self.evaluator.evaluate(
                question, answer, chunk_texts
            )
        return result
