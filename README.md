# 🤖 RAG Chatbot

A production-style **Retrieval-Augmented Generation (RAG)** chatbot built with
**LangChain**, **FAISS**, and **OpenAI**. Drop in your PDFs or text files, ask
questions, and get grounded answers with cited sources.

## ✨ Features

- 📄 Ingests **PDF** and **TXT** documents
- ✂️ Smart chunking with overlap (RecursiveCharacterTextSplitter)
- 🧠 Free local embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- 🔍 FAISS vector search with top-k retrieval
- 💬 Grounded answers — the LLM only uses retrieved context
- 📚 Source citations (file + page number) for every answer
- 💾 Persistent FAISS index — builds once, reuses on restart
- 🖥️ Clean Streamlit chat UI

## 🏗️ Architecture

```
data/  ──►  Loader (PDF/TXT)  ──►  Chunker  ──►  Embeddings  ──►  FAISS index
                                                                              │
User question ──► Retriever (top-k) ──► Prompt + Context ──► LLM ──► Answer + Sources
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
│   ├── rag_engine.py   # Core RAG logic (load → chunk → embed → retrieve → answer)
│   └── app.py          # Streamlit chat UI
├── data/               # Put your PDFs / TXTs here
├── tests/
│   └── test_rag_engine.py
├── requirements.txt
└── .env.example
```

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `llm_model` | `gpt-4o-mini` | Chat model for answers |
| `chunk_size` | `1000` | Characters per chunk |
| `chunk_overlap` | `200` | Overlap between chunks |
| `top_k` | `4` | Retrieved chunks per query |

## 🧪 Tests

```bash
pytest tests/
```

## 📝 License

MIT
