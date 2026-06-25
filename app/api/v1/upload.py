import io
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from qdrant_client import QdrantClient
from PyPDF2 import PdfReader

from app.core.ingestion import ingest_document
from app.core.qdrant import get_qdrant_client

router = APIRouter()

# ─────────────────────────────────────────────
# Dependency — reuse Qdrant client
# ─────────────────────────────────────────────
def get_client() -> QdrantClient:
    return get_qdrant_client()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode plain text file."""
    return file_bytes.decode("utf-8").strip()

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@router.post("/upload", summary="Upload a document for ingestion")
async def upload_document(
    file: UploadFile = File(...),
    client: QdrantClient = Depends(get_client),
):
    """
    Upload a .txt or .pdf file.
    The file will be:
    1. Parsed into raw text
    2. Chunked into overlapping segments
    3. Embedded using sentence-transformers
    4. Stored in Qdrant (Docker)
    """

    # ── Validate file type ──────────────────
    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"❌ Unsupported file type: {file.content_type}. Use PDF or TXT.",
        )

    # ── Read file bytes ─────────────────────
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="❌ Uploaded file is empty.")

    # ── Extract text ────────────────────────
    if file.content_type == "application/pdf":
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    if not text:
        raise HTTPException(
            status_code=422,
            detail="❌ Could not extract any text from the file.",
        )

    # ── Build metadata ──────────────────────
    doc_id = str(uuid.uuid4())
    metadata = {
        "doc_id": doc_id,
        "filename": file.filename,
        "content_type": file.content_type,
    }

    # ── Ingest ──────────────────────────────
    result = ingest_document(
        client=client,
        text=text,
        metadata=metadata,
    )

    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_stored": result["stored"],
        "word_count": len(text.split()),
    }