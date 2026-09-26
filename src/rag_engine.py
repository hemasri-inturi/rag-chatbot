"""
RAG Chatbot - Core RAG engine.

Loads documents, builds a FAISS vector index, and answers
questions using retrieval-augmented generation.
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

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question
using ONLY the context provided below. If the answer is not in the context,
say you don't know — do not make up information.

Context:
{context}

Question: {question}

Answer:"""


class RAGEngine:
    """Retrieval-Augmented Generation engine for document Q&A."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.vectorstore: Optional[FAISS] = None

        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_documents(self, data_dir: str) -> List[Document]:
        """Load PDF and TXT documents from a directory."""
        docs: List[Document] = []
        for path in Path(data_dir).rglob("*"):
            if path.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(path)).load())
            elif path.suffix.lower() == ".txt":
                docs.extend(TextLoader(str(path), encoding="utf-8").load())
        return docs

    def build_index(self, documents: List[Document]) -> int:
        """Chunk documents and build the FAISS vector index.

        Returns the number of chunks indexed.
        """
        chunks = self.splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    def save_index(self, index_dir: str) -> None:
        """Persist the FAISS index to disk."""
        if self.vectorstore is None:
            raise ValueError("No index built yet. Call build_index() first.")
        self.vectorstore.save_local(index_dir)

    def load_index(self, index_dir: str) -> None:
        """Load a persisted FAISS index from disk."""
        self.vectorstore = FAISS.load_local(
            index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, question: str) -> dict:
        """Answer a question with sources.

        Returns:
            dict with 'answer' and 'sources' keys.
        """
        if self.vectorstore is None:
            raise ValueError("No index loaded. Build or load an index first.")

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        retrieved_docs = retriever.invoke(question)

        chain = (
            {
                "context": lambda _: self._format_docs(retrieved_docs),
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
            for doc in retrieved_docs
        ]
        return {"answer": answer, "sources": sources}
