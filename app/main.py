import os

from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.qdrant import get_qdrant_client, init_collection


load_dotenv()

APP_NAME = "RAG Evaluation API"
APP_VERSION = "0.1.0"
APP_ENV = os.getenv("APP_ENV", "development")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 Starting {APP_NAME} v{APP_VERSION}")
    print(f"🌍 Environment: {APP_ENV}")

    client = get_qdrant_client()
    init_collection(client)

    yield
    print("👋 Shutting down gracefully...")

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Production-grade RAG API with evaluation, reranking, and observability",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "environment": APP_ENV,
    }

    
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )