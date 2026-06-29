from fastapi import APIRouter, Depends
from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.core.qdrant import get_qdrant_client
from app.core.retrieval import retrieve_context
from app.core.llm import generate_answer

router = APIRouter()


def get_client() -> QdrantClient:
    return get_qdrant_client()


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


@router.post("/chat")
async def chat(
    request: ChatRequest,
    client: QdrantClient = Depends(get_client),
):
    """
    Ask a question using Retrieval-Augmented Generation.
    """

    chunks = retrieve_context(
        client=client,
        query=request.question,
        k=request.top_k,
    )

    answer = generate_answer(
        question=request.question,
        context_chunks=chunks,
    )

    return {
        "question": request.question,
        "answer": answer,
        "sources": chunks,
    }