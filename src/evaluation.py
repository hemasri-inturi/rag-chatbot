"""
RAG evaluation metrics (RAGAS-style, lightweight implementation).

- Faithfulness: is the answer grounded in the retrieved context?
- Answer relevancy: does the answer address the question?
- Context precision: fraction of retrieved chunks relevant to the question.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

FAITHFULNESS_PROMPT = """Decide if the answer below is fully supported by the context.
Respond with ONLY "YES" or "NO".

Context:
{context}

Answer:
{answer}"""

RELEVANCY_PROMPT = """Rate how well the answer addresses the question.
Respond with ONLY a number from 0 to 10.

Question: {question}
Answer: {answer}"""

CHUNK_RELEVANCY_PROMPT = """Is this chunk relevant to answering the question?
Respond with ONLY "YES" or "NO".

Question: {question}
Chunk: {chunk}"""


class RAGEvaluator:
    """Lightweight RAGAS-style evaluator using an LLM judge."""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0)
        self.faithfulness_chain = (
            ChatPromptTemplate.from_template(FAITHFULNESS_PROMPT)
            | self.llm
            | StrOutputParser()
        )
        self.relevancy_chain = (
            ChatPromptTemplate.from_template(RELEVANCY_PROMPT)
            | self.llm
            | StrOutputParser()
        )
        self.chunk_chain = (
            ChatPromptTemplate.from_template(CHUNK_RELEVANCY_PROMPT)
            | self.llm
            | StrOutputParser()
        )

    def faithfulness(self, answer: str, context: str) -> float:
        verdict = self.faithfulness_chain.invoke(
            {"answer": answer, "context": context}
        ).strip().upper()
        return 1.0 if verdict.startswith("YES") else 0.0

    def answer_relevancy(self, question: str, answer: str) -> float:
        score = self.relevancy_chain.invoke(
            {"question": question, "answer": answer}
        ).strip()
        try:
            return max(0.0, min(10.0, float(score))) / 10.0
        except ValueError:
            return 0.0

    def context_precision(self, question: str, chunks: list) -> float:
        if not chunks:
            return 0.0
        relevant = sum(
            1
            for chunk in chunks
            if self.chunk_chain.invoke(
                {"question": question, "chunk": chunk}
            ).strip().upper().startswith("YES")
        )
        return relevant / len(chunks)

    def evaluate(self, question: str, answer: str, chunks: list) -> dict:
        context = "\n\n".join(chunks)
        return {
            "faithfulness": self.faithfulness(answer, context),
            "answer_relevancy": self.answer_relevancy(question, answer),
            "context_precision": self.context_precision(question, chunks),
        }
