 HEAD
# PDF Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) application for asking questions about indexed PDF documents. It extracts text, applies OCR fallback to pages with insufficient native text, chunks the content, stores embeddings in ChromaDB, retrieves and reranks relevant passages, and generates answers with a locally hosted Ollama model.

> **Project status:** Development prototype. The current development corpus contains 10 PDFs with 50 pages each (500 pages total). It does **not** yet meet the challenge requirement of at least 10 PDFs with at least 200 pages each.

## Features

- PDF text extraction using PyMuPDF.
- OCR fallback using Tesseract OCR and pytesseract.
- Token-aware chunking with overlap using LangChain.
- Embeddings using `BAAI/bge-small-en-v1.5`.
- Persistent vector storage and semantic retrieval using ChromaDB.
- Candidate reranking using `BAAI/bge-reranker-base`.
- Local answer generation using Ollama `qwen2.5:1.5b`.
- Streamlit chat interface with PDF filenames, page numbers, retrieved passages, and per-query performance metrics.
- SHA-256 file hashing and an ingestion registry to help manage indexing.

## Architecture

```text
PDF files
  -> PyMuPDF extraction
  -> Tesseract OCR fallback for pages with insufficient native text
  -> Page-level text and metadata
  -> Token-aware chunking
  -> BGE-small embeddings
  -> ChromaDB persistent vector index

User question
  -> Semantic retrieval (8 candidates)
  -> BGE cross-encoder reranking (up to 5 chunks)
  -> Context preparation
  -> Ollama Qwen 2.5 1.5B
  -> Answer + retrieved sources + query-specific metrics
  -> Streamlit UI
```

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python |
| PDF extraction | PyMuPDF |
| OCR | Tesseract OCR, pytesseract, Pillow |
| Chunking | LangChain text splitters |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector database | ChromaDB |
| Reranker | `BAAI/bge-reranker-base` |
| Local LLM | Ollama `qwen2.5:1.5b` |
| Frontend | Streamlit |
| Evaluation | Custom retrieval and answer-quality scripts; RAGAS package installed |

## Project structure

```text
navgurukul/
├── data/
│   ├── pdfs/
│   └── processed/
├── vectorstore/
│   └── chroma_db/
├── models/
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   ├── chunking.py
│   │   └── ingest_manager.py
│   ├── retrieval/
│   │   └── reranker.py
│   ├── generation/
│   │   ├── llm.py
│   │   └── rag_pipeline.py
│   └── evaluation/
│       ├── evaluate_rag.py
│       └── review_answer_quality.py
├── frontend/
│   └── app.py
├── tests/
│   ├── verify_index.py
│   └── check_pdf_pages.py
├── logs/
└── README.md
```

## Prerequisites

- Windows 10/11 and Python available on PATH.
- Tesseract OCR installed. The current loader expects `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- Ollama installed and running.
- Adequate disk space for Python packages, downloaded model weights, the Ollama model, and the vector database.

## Setup (Windows CMD)

Open CMD in the project root.

### 1. Create and activate a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

If the project has a `requirements.txt`:

```cmd
pip install -r requirements.txt
```

Otherwise, install the packages used by the current project:

```cmd
pip install pymupdf pytesseract Pillow langchain langchain-text-splitters sentence-transformers chromadb transformers torch accelerate fastapi uvicorn streamlit requests pandas numpy python-multipart ragas
```

### 3. Download the Ollama model

```cmd
ollama pull qwen2.5:1.5b
ollama list
```

The generation module calls the local Ollama API at `http://localhost:11434/api/generate`.

### 4. Configure Tesseract

Install Tesseract OCR. If its executable is not at `C:\Program Files\Tesseract-OCR\tesseract.exe`, update the path in `src/ingestion/pdf_loader.py`.

## Add PDFs and build the index

Place PDF files in `data\pdfs`, then run:

```cmd
python -m src.ingestion.ingest_manager
```

The index is stored in `vectorstore\chroma_db`. The ingestion registry is stored in `data\processed\ingestion_registry.json`.

## Verify the index

```cmd
python tests\verify_index.py
python tests\check_pdf_pages.py
```

A previous development verification reported 941 chunks across 10 unique PDFs. The page-count check reported 50 pages per PDF and no PDFs meeting the 200-page minimum. Results will change if the corpus or chunking configuration changes.

## Run the application

```cmd
streamlit run frontend\app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

For each question, the UI displays the answer, retrieved passages, PDF filenames, page numbers, and metrics for that query. The frontend is intentionally configured without an aggregate evaluation dashboard.

## Query-specific metrics

The current RAG pipeline returns:

- Retrieval time
- Reranking time
- Generation time
- Total pipeline time
- Number of candidates retrieved
- Number of chunks retained after reranking
- Number of sources used
- Average reranker score

These are runtime and retrieval statistics. They are not the same as factual correctness, faithfulness, citation accuracy, Precision@k, Recall@k, or MRR.

## Evaluation scripts

Run retrieval evaluation:

```cmd
python -m src.evaluation.evaluate_rag
```

Run the automated answer-quality proxy:

```cmd
python -m src.evaluation.review_answer_quality
```

The retrieval evaluation uses configured test questions and relevant-page annotations to calculate measures such as Precision@5, Recall@5, MRR, and latency. The reliability of these metrics depends on the test set and ground truth.

The answer-quality script uses embedding similarity between answer sentences and retrieved passages. This is a heuristic proxy, not proof of entailment or factual correctness. Do not report it as validated factual accuracy.

Evaluation outputs are stored under `data\processed`.

## Current configuration and limitations

- Chunk size: 400 tokenizer tokens; overlap: 80 tokens. This is below the challenge's stated 500–1000-token chunk target.
- Retrieval: 8 candidates; reranking returns up to 5.
- Reranker runs on CPU with maximum sequence length 256 and batch size 4.
- Local LLM: Ollama `qwen2.5:1.5b`.
- The development machine has an Intel Core i5-1035G1, 8 GB RAM, and integrated graphics.
- A previously observed query took about 39.4 seconds total, including about 22.7 seconds for generation and 3.6 seconds for reranking. This does not meet the stated 2–5-second target.
- OCR fallback is applied to pages with insufficient native text. Embedded images on otherwise text-rich pages are not separately OCR-processed by the current loader.
- Bounding-box metadata is not currently captured.
- The current development corpus has 10 PDFs of 50 pages each; it does not meet the final 200-pages-per-PDF requirement.
- The small local model can make unsupported inferences. Retrieved sources do not automatically prove every answer claim.

## Troubleshooting

### Progress callback ValueError

The current pipeline calls the progress callback as `(message, percentage)`. The frontend callback must use the same order:

```python
def update_progress(message, percentage):
    progress_bar.progress(min(max(int(percentage), 0), 100))
    status_text.write(message)
```

### Streamlit width deprecation

Use `width="stretch"` instead of `use_container_width=True`:

```python
st.button("Clear Chat History", width="stretch")
```

### Ollama connection issue

Check that Ollama is running and the model is installed:

```cmd
ollama list
```

### ChromaDB index issue

Confirm the persistent path is `vectorstore\chroma_db`. Do not delete the database or ingestion registry unless you intend to rebuild the index. Keep backups until a rebuilt index has been verified.

## Future improvements

- Prepare at least 10 final PDFs with 200+ pages each.
- Align chunk size with the 500–1000-token requirement and validate overlap.
- OCR embedded images on text-rich pages and capture bounding boxes.
- Improve answer grounding and claim-level citation verification.
- Optimize CPU reranking and LLM generation latency.
- Expand the test set and relevant-page ground truth.
- Evaluate retrieval and answer quality on a larger representative dataset.
- Complete FastAPI endpoints if required by the final deployment architecture.

## License

Before public distribution, document the licenses for the code, dependencies, model weights, and PDF dataset. These may have different license terms.
 
# pdf-knowledge-assistant
A local Retrieval-Augmented Generation (RAG) application for querying PDF documents using OCR, semantic search, ChromaDB, BGE embeddings, reranking, and Ollama-powered LLMs.  Copy description
 9bc3a528390778f81ad4665a646377c9bf86831d
