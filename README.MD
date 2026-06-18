# Adaptive RAG

An adaptive Retrieval-Augmented Generation (RAG) system with query classification, confidence-based iterative retrieval, LLM fallback, and asynchronous document ingestion.

## Architecture

```
Client ──► FastAPI ──► QueryClassifier ──► DIRECT_LLM
                         │                    │
                         ▼                    ▼
                    Retriever ──► ConfidenceEvaluator
                         │                    │
                         │        ┌───────────┘
                         │        ▼
                         │   QueryRewriter (loop if low confidence)
                         │        │
                         ▼        ▼
                    RAGService / AdaptiveRAGService ──► LLM (with fallback)
                         │
                         ▼
               ChromaDB (vector store)
```

## Features

- **Adaptive query routing** — classifies queries as DIRECT_LLM (greetings, general) or KNOWLEDGE_QUERY (document-specific)
- **Confidence-based iterative retrieval** — evaluates retrieval quality; rewrites and re-retrieves when confidence is low
- **LLM model fallback** — primary + fallback models with exponential backoff retry on rate limits / API errors
- **Background document ingestion** — upload returns immediately; processing runs async with job status polling
- **SSE streaming** — `POST /query/stream` streams answer tokens via Server-Sent Events
- **Idempotent re-upload** — re-uploading a document replaces its existing vectors (no duplicates)
- **Session lifecycle** — `DELETE /session` clears all data; auto-cleanup on server shutdown
- **RAGAS evaluation** — generates evaluation datasets with real retrieved contexts and system answers
- **Multi-format ingestion** — PDF, DOCX, TXT, XLSX

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload a document (returns `job_id`, processes async) |
| `GET` | `/upload/status/{job_id}` | Poll ingestion job status |
| `POST` | `/query` | Ask a question (returns full JSON response) |
| `POST` | `/query/stream` | Ask a question (SSE token stream) |
| `POST` | `/evaluate` | Run RAGAS evaluation on a dataset |
| `GET` | `/evaluate/results` | Retrieve evaluation results |
| `DELETE` | `/session` | Clear all data (ChromaDB, files, jobs) |

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set your `OPENAI_API_KEY` (or OpenRouter key). Configure models:

```env
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://openrouter.ai/api/v1
LLM_MODEL=gpt-4o-mini
LLM_MODELS=gpt-4o-mini,claude-3-haiku-20240307   # comma-separated fallbacks
EMBEDDING_MODEL=text-embedding-3-small
```

## Run

```powershell
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the Swagger UI.

## Docker

```powershell
docker build -t adaptive-rag .
docker run -p 8000:8000 --env-file .env adaptive-rag
```

## Project Structure

```
app/
├── api/              # FastAPI routers
│   ├── health.py
│   ├── upload.py
│   ├── query.py
│   ├── evaluate.py
│   ├── session.py
│   └── dependencies.py
├── ingestion/        # Document loading & chunking
│   ├── loaders.py
│   ├── chunker.py
│   └── ingestion_service.py
├── retrievers/       # Retrieval pipeline
│   ├── query_classifier.py
│   ├── query_rewriter.py
│   ├── retriever.py
│   └── confidence_evaluator.py
├── services/         # Business logic
│   ├── rag_service.py
│   ├── adaptive_rag_service.py
│   ├── citation_service.py
│   └── ingestion_tracker.py
├── evaluators/       # RAGAS evaluation
│   ├── ragas_evaluator.py
│   ├── eval_dataset_generator.py
│   └── metrics_tracker.py
├── vectorstore/      # ChromaDB client
│   └── chroma_manager.py
├── models/           # Pydantic schemas
│   ├── requests.py
│   └── responses.py
├── utils/            # Config, logging, LLM factory
│   ├── config.py
│   ├── logger.py
│   ├── llm_factory.py
│   └── helpers.py
└── main.py
```
