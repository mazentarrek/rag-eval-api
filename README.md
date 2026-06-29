# Production-Grade RAG Evaluation API

A production-style Retrieval-Augmented Generation (RAG) API built with **FastAPI**, **Qdrant**, **Sentence Transformers**, and **Google Gemini**. The project supports document ingestion, semantic retrieval, LLM-powered question answering, and automated evaluation using **RAGAS**.

## Features

* PDF and TXT document ingestion
* Configurable document chunking
* Sentence Transformer embeddings (`all-MiniLM-L6-v2`)
* Vector storage and semantic search with Qdrant
* Context-aware question answering using Google Gemini
* RESTful API built with FastAPI
* RAG evaluation using RAGAS
* Dockerized vector database
* OpenAPI (Swagger) documentation

## Tech Stack

* Python
* FastAPI
* Qdrant
* Sentence Transformers
* Google Gemini API
* RAGAS
* Docker
* PyPDF2

## Project Structure

```text
app/
├── api/
│   └── v1/
├── core/
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── llm.py
│   └── qdrant.py
├── main.py
```

## API Endpoints

| Method | Endpoint           | Description                          |
| ------ | ------------------ | ------------------------------------ |
| POST   | `/api/v1/upload`   | Upload and index PDF/TXT documents   |
| POST   | `/api/v1/chat`     | Ask questions using RAG              |
| POST   | `/api/v1/evaluate` | Evaluate the RAG pipeline with RAGAS |
| GET    | `/health`          | Health check                         |

## RAG Pipeline

```text
Document
    │
    ▼
Text Extraction
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
Qdrant Vector Store
    │
    ▼
Semantic Retrieval
    │
    ▼
Gemini
    │
    ▼
Answer
```

## Evaluation

The project integrates **RAGAS** to evaluate retrieval and generation quality using:

* Faithfulness
* Answer Relevancy
* Context Precision

