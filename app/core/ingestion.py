import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.core.embeddings import chunk_text, get_embeddings_batch
from app.core.qdrant import COLLECTION_NAME

def ingest_document(
    client: QdrantClient,
    text: str,
    metadata: dict,
) -> dict:
    """
    Full pipeline:
    1. Chunk the text
    2. Embed all chunks
    3. Store in Qdrant with metadata
    """

    # Step 1: Chunk
    chunks = chunk_text(text)
    print(f"📄 Split into {len(chunks)} chunks")

    # Step 2: Embed
    embeddings = get_embeddings_batch(chunks)
    print(f"🔢 Generated {len(embeddings)} embeddings")

    # Step 3: Store in Qdrant
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                **metadata,
                "chunk_index": i,
                "chunk_text": chunk,
            },
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"✅ Stored {len(points)} points in Qdrant")

    return {
        "chunks": len(chunks),
        "stored": len(points),
        "metadata": metadata,
    }