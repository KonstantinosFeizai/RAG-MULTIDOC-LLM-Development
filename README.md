# Local Multi-Document RAG System

A production-grade, fully local Retrieval-Augmented Generation (RAG) system built from scratch using Python, `uv`, Ollama, ChromaDB, Docling, Hugging Face Cross-Encoders, and Streamlit. This system features multi-format document parsing, structure-aware chunking, conversational query rewriting, two-stage precision reranking, source citation tracking, an interactive web UI, and automated evaluations.

---

## Architecture & Tech Stack

- **Package Management:** `uv` (requires Python 3.10+)
- **Document Parsing:** [Docling](https://docling.ai/) for converting `.pdf` and `.docx` files into clean structured Markdown
- **Local LLM & Embeddings:** [Ollama](https://ollama.com/) (`qwen3:4b-instruct` for query rewriting and generation, `nomic-embed-text` for vector embeddings)
- **Vector Database & Keyword Index:** [ChromaDB](https://www.trychroma.com/) (persistent local vector storage with deterministic `upsert` synchronization) paired with [`rank-bm25`](https://pypi.org/project/rank-bm25/) (`BM25Okapi`) for lexical keyword retrieval
- **Rank Fusion & Reranker:** Reciprocal Rank Fusion (RRF) for scale-agnostic candidate merging, followed by `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for deep semantic candidate reranking
- **Web Interface:** [Streamlit](https://streamlit.io/) for an interactive browser-based chat UI with retrieval diagnostics
- **Chunking Strategy:** Custom Structure-Aware (Header-Aware) Markdown Parsing

---

## Project Structure

```text
rag_01/
├── docs/                      # Source files (.md, .pdf, .docx handbooks, guides, etc.)
│   ├── handbook.md
│   └── tennis_padel_rules_handbook.md
├── chroma_db/                 # Persistent local vector database directory
├── ingest.py                  # Multi-format parser (Docling), header-aware chunker, and indexer
├── rag_engine.py              # Core RAG engine (query rewriting, retrieval, reranking, LLM)
├── app.py                     # Streamlit Web UI frontend with sidebar diagnostics
├── rag.py                     # Terminal CLI chat loop with conversational memory
├── evaluate.py                # Automated RAG testing and scoring script
├── eval_dataset.json          # Test suite dataset with required evaluation keywords
└── pyproject.toml             # Project dependencies and environment configuration
```

## Key Features

1. **Multi-Format Document Ingestion (`Docling`):**
   - Automatically scans `docs/` for `.md`, `.pdf`, and `.docx` files.
   - Uses Docling's local layout models to extract multi-column layouts, tables, and headings into structured Markdown before chunking.
2. **Clean Separation of Concerns:**
   - `ingest.py` handles document conversion, structure tracking, and embedding sync.
   - `rag_engine.py` decouples the core retrieval, reranking, and generation logic so both the Web UI and CLI share the exact same backend.
   - `app.py` provides a visual Streamlit frontend without mixing UI code into the retrieval pipeline.
3. **Deterministic Upserts:**
   - Generates unique, predictable chunk IDs based on filenames and positions (`<filename>_chunk_<index>`). Running the indexer multiple times safely updates modified content or adds new files without duplicating chunks or requiring a manual database wipe.
4. **Header-Aware Chunking:**
   - Automatically detects markdown headers (`#`, `##`, `###`), binds child paragraphs to their parent section, and injects clean breadcrumbs (`Source: file.name > Section`) directly into the chunk text.
5. **Conversational Memory & Query Rewriting:**
   - Automatically rewrites vague follow-up questions using recent chat history into standalone search queries before querying ChromaDB, preventing context loss across multi-turn conversations.
6. **Hybrid Search + Reciprocal Rank Fusion (RRF) + Cross-Encoder Reranking BM25:**
   - **Stage 1:** Broad vector search retrieves top candidate chunks quickly via ChromaDB.
   - **Stage 2:** A local Cross-Encoder model cross-references the question and chunks simultaneously, re-sorting candidates to feed only the highest-precision context to the LLM,Merges vector and BM25 rankings.
7. **Interactive Web UI (`Streamlit`):**
   - Browser chat interface featuring session history, source citations, rewritten standalone query inspection, and expandable reranked context chunks with relevance scores.
8. **Automated Evals Suite:**
   - A built-in testing framework (`evaluate.py`) that scores system accuracy against predefined test cases and keyword criteria.

---

## Quickstart Guide

### 1. Prerequisites

Ensure you have [Ollama](https://ollama.com/) running locally with your models pulled:

```powershell
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
```

### 2. Install Dependencies

Initialize and sync your dependencies using `uv`:

```powershell
uv add chromadb requests sentence-transformers docling streamlit
```

### 3. Ingest Your Documents

Drop your .md, .pdf, or .docx files into the `docs/` folder, then run the indexing script:

```powershell
uv run ingest.py
```

### 4. Run the RAG Chat Loop in terminal

Start interacting with your documents with full source attribution and reranking:

```powershell
uv run rag.py
```

### 5. Launch the Web UI

Start the Streamlit web application in your browser (http://localhost:8501) with full source attribution, query rewriting, and reranking diagnostics:

```powershell
uv run streamlit run app.py
```

### 5. Run Automated Evals

To run the built-in evaluation suite and verify system accuracy:

```powershell
uv run evaluate.py
```
