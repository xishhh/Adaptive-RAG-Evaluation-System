# Adaptive RAG

Phase 1 FastAPI skeleton for an Adaptive Retrieval-Augmented Generation system.

## Current Status

Implemented:

- FastAPI app factory
- Health endpoint: `GET /health`
- Stub endpoints for upload, query, evaluation, and metrics
- Central settings and logging
- Basic tests

Not implemented yet:

- Document ingestion
- Vector storage
- Retrieval
- Answer generation
- RAGAS evaluation

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env