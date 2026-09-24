from pathlib import Path
from typing import List, Dict, Any
import os

import chromadb


VECTOR_DB_PATH = Path("data/processed/vector_db")
COLLECTION_NAME = os.getenv("FTR_VECTOR_COLLECTION", "ftr_chunks")


_COLLECTION_METADATA = {
    "description": "FTR-RAG physiotherapy guideline chunks",
    "embedding_model": os.getenv("FTR_EMBEDDING_MODEL", "unknown"),
    # Compatibility marker for older Chroma versions.
    "hnsw:space": "cosine",
}


def _collection_create_options(client) -> Dict[str, Any]:
    """Return Chroma-version-compatible options for a cosine HNSW index."""
    try:
        import inspect
        params = inspect.signature(client.get_or_create_collection).parameters
        if "configuration" in params:
            return {
                "metadata": {
                    "description": _COLLECTION_METADATA["description"],
                    "embedding_model": _COLLECTION_METADATA["embedding_model"],
                },
                "configuration": {"hnsw": {"space": "cosine"}},
            }
    except (TypeError, ValueError):
        pass

    # Chroma versions before the configuration API use metadata.
    return {"metadata": dict(_COLLECTION_METADATA)}


def _create_collection(client):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        **_collection_create_options(client),
    )


def get_collection():
    """Open or create the persistent FTR-RAG collection using cosine space."""
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    return _create_collection(client)


def reset_collection():
    """Delete and recreate the active collection with cosine distance."""
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    return _create_collection(client)


def get_indexed_ids() -> List[str]:
    """Return all chunk IDs currently stored in ChromaDB."""
    return list(get_collection().get()["ids"])


def add_chunks(
    collection,
    ids: List[str],
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
):
    """
    Chunk'ları ChromaDB'ye ekler veya mevcut ID varsa günceller.
    """

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def count() -> int:
    """Return the number of indexed chunks in the Chroma collection."""
    return get_collection().count()


def search(
    query_embedding: List[float],
    n_results: int = 10,
):
    """
    Verilen query embedding için ChromaDB'den
    en benzer chunk'ları getirir.
    """

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return results

def get_embeddings_by_ids(ids: List[str]) -> Dict[str, List[float]]:
    """Return already-indexed embeddings for the requested chunk IDs.

    This is intentionally read-only: it reuses the vectors stored in the
    current Chroma collection instead of re-embedding the chunk text.
    """
    if not ids:
        return {}

    collection = get_collection()
    result = collection.get(
        ids=list(ids),
        include=["embeddings"],
    )

    result_ids = result.get("ids", [])
    embeddings = result.get("embeddings", [])

    if embeddings is None:
        embeddings = []

    return {
        str(chunk_id): embedding
        for chunk_id, embedding in zip(result_ids, embeddings)
        if embedding is not None
    }
