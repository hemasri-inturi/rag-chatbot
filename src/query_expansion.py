"""
Query expansion: rewrites the user question into multiple
search queries for broader retrieval coverage.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

EXPANSION_PROMPT = """Generate 3 alternative search queries for the question below.
Each query should use different wording to maximize retrieval coverage.
Return one query per line, no numbering, no extra text.

Question: {question}"""


class QueryExpander:
    """LLM-based multi-query expansion."""

    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_template(EXPANSION_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def expand(self, question: str, n: int = 3) -> list:
        result = self.chain.invoke({"question": question})
        queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
        return [question] + queries[:n]
