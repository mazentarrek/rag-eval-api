import os

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimensions

# Loads once at startup, reused across all requests
print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Embedding model loaded!")

splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    
        chunk_overlap=200,    
        separators=["\n\n", "\n", ".", " ", ""]
    )

# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    
    return splitter.split_text(text)

# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """
    Get embedding vector for a single text string.
    Takes a string → returns a list of 384 numbers

    Args:
        text: Input text

    Returns:
        List of floats (length = VECTOR_SIZE)
    """
    embedding = model.encode(
        text,
        normalize_embeddings=True,  # cosine similarity ready
    )
    return embedding.tolist()

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Get embeddings for multiple chunks in a single efficient call.

    Args:
        texts: List of text chunks

    Returns:
        List of embedding vectors
    """
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,           # process 32 chunks at a time
        show_progress_bar=True,  # helpful for large documents
    )
    return embeddings.tolist()

# ─────────────────────────────────────────────
# Token / word count utility
# ─────────────────────────────────────────────
def count_words(text: str) -> int:
    """
    Returns word count of a string.

    Args:
        text: Input text

    Returns:
        Word count
    """
    return len(text.split())