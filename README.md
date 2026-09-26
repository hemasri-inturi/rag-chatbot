# 🤖 Advanced RAG Chatbot

A production-style **Retrieval-Augmented Generation (RAG)** system with an
advanced retrieval pipeline: **hybrid search**, **cross-encoder reranking**,
**query expansion**, and **RAGAS-style evaluation** — built with LangChain,
FAISS, and OpenAI.

## ✨ Features

### Retrieval pipeline
- 📄 Ingests **PDF** and **TXT** documents
- ✂️ Smart chunking with overlap (RecursiveCharacterTextSplitter)
- 🔀 **Query expansion** — LLM rewrites the question into multiple search queries
- 🔍 **Hybrid retrieval** — dense (FAISS) + sparse (BM25), fused with **Reciprocal Rank Fusion**
- 🎯 **Cross-encoder reranking** (`ms-marco-MiniLM`) for precision on the final top-k

### Generation & trust
- 💬 Grounded answers — the LLM only uses retrieved context
- 📚 Source citations (file + page number) for every answer
- 📊 **Evaluation metrics** — faithfulness, answer relevancy, context precision (RAGAS-style)
- 💾 Persistent FAISS index — builds once, reuses on restart
- 🖥️ Streamlit chat UI with pipeline toggles

## 🏗️ Architecture

```
                        ┌─ Dense (FAISS) ─┐
Query → Expansion ─┤                     ├─► RRF Fusion ─► Reranker ─► Top-k chunks
                        └─ Sparse (BM25) ─┘                                        │
                                                                                   ▼
User question ──► Prompt + Context ──► LLM ──► Answer + Sources + Eval metrics
```

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/hemasri-inturi/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 3. Drop documents into data/ (sample included)

# 4. Run
streamlit run src/app.py
```

## 📁 Project Structure

```
rag-chatbot/
├── src/
│   ├── rag_engine.py       # AdvancedRAGEngine: full pipeline orchestration
│   ├── hybrid_retriever.py # Dense + BM25 with Reciprocal Rank Fusion
│   ├── reranker.py         # Cross-encoder reranking
│   ├── query_expansion.py  # LLM multi-query expansion
│   ├── evaluation.py       # RAGAS-style metrics (LLM judge)
│   └── app.py              # Streamlit chat UI
├── data/                   # Put your PDFs / TXTs here
├── tests/
├── requirements.txt
└── .env.example
```

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Local bi-encoder embeddings |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `llm_model` | `gpt-4o-mini` | Chat model for answers & eval |
| `chunk_size` | `800` | Characters per chunk |
| `chunk_overlap` | `150` | Overlap between chunks |
| `retrieval_k` | `12` | Candidates per retrieval branch |
| `final_k` | `4` | Chunks after reranking |

## 📊 Evaluation

Toggle **"Show evaluation metrics"** in the sidebar to score every answer:

| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer fully supported by the retrieved context? |
| Answer relevancy | Does the answer address the question? (0–1) |
| Context precision | Fraction of retrieved chunks relevant to the question |

## 🧪 Tests

```bash
pytest tests/
```

## 📝 License

MIT
