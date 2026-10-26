from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from src.retrieval.config import COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_DB_PATH

logger = logging.getLogger(__name__)

_COLLECTION_METADATA = {
    "description": "FTR-RAG physiotherapy guideline chunks",
    "embedding_model": EMBEDDING_MODEL,
    # Compatibility marker for older Chroma versions.
    "hnsw:space": "cosine",
}

_CLIENT = None
_COLLECTION = None
_CACHE_KEY: Optional[Tuple[str, str]] = None


def _current_cache_key() -> Tuple[str, str]:
    return str(VECTOR_DB_PATH.resolve()), COLLECTION_NAME


def _get_client():
    global _CLIENT, _COLLECTION, _CACHE_KEY
    cache_key = _current_cache_key()
    if _CLIENT is None or _CACHE_KEY != cache_key:
        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        _COLLECTION = None
        _CACHE_KEY = cache_key
    return _CLIENT


def _collection_create_options(client) -> Dict[str, Any]:
    """Return Chroma-version-compatible options for a cosine HNSW index."""
    try:
        parameters = inspect.signature(client.get_or_create_collection).parameters
        if "configuration" in parameters:
            return {
                "metadata": {
                    "description": _COLLECTION_METADATA["description"],
                    "embedding_model": _COLLECTION_METADATA["embedding_model"],
                },
                "configuration": {"hnsw": {"space": "cosine"}},
            }
    except (TypeError, ValueError):
        pass
    return {"metadata": dict(_COLLECTION_METADATA)}


def _create_collection(client):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        **_collection_create_options(client),
    )


def get_collection():
    """Open or create the persistent collection; intended for index building."""
    global _COLLECTION
    client = _get_client()
    if _COLLECTION is None:
        _COLLECTION = _create_collection(client)
    return _COLLECTION


def open_collection():
    """Open an existing non-empty collection without creating one."""
    global _COLLECTION
    client = _get_client()
    if _COLLECTION is None:
        try:
            _COLLECTION = client.get_collection(name=COLLECTION_NAME)
        except chromadb.errors.NotFoundError as exc:
            raise RuntimeError(
                f"Chroma collection '{COLLECTION_NAME}' was not found at "
                f"'{VECTOR_DB_PATH}'. Build the dense index first."
            ) from exc

    try:
        collection_count = _COLLECTION.count()
    except Exception:
        _COLLECTION = None
        raise
    if collection_count == 0:
        raise RuntimeError(
            f"Chroma collection '{COLLECTION_NAME}' exists but is empty at "
            f"'{VECTOR_DB_PATH}'. Build the dense index first."
        )
    return _COLLECTION


def reset_collection():
    """Delete and recreate the active collection for index building."""
    global _CLIENT, _COLLECTION, _CACHE_KEY
    _CLIENT = None
    _COLLECTION = None
    _CACHE_KEY = None

    client = _get_client()
    try:
        client.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass
    else:
        client.delete_collection(name=COLLECTION_NAME)

    _COLLECTION = _create_collection(client)
    return _COLLECTION


def get_indexed_ids() -> List[str]:
    """Return all indexed chunk IDs without loading documents or metadata."""
    return list(open_collection().get(include=[])["ids"])


def add_chunks(
    collection,
    ids: List[str],
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
):
    """Insert chunks into Chroma, updating existing IDs."""
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def count() -> int:
    """Return the number of indexed chunks; fail if no usable index exists."""
    return open_collection().count()


def search(
    query_embedding: List[float],
    n_results: int = 10,
):
    """Search the existing collection by query embedding."""
    return open_collection().query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def get_embeddings_by_ids(ids: List[str]) -> Dict[str, List[float]]:
    """Return already-indexed embeddings for the requested IDs."""
    if not ids:
        return {}

    result = open_collection().get(ids=list(ids), include=["embeddings"])
    result_ids = result.get("ids", [])
    embeddings = result.get("embeddings")
    if embeddings is None:
        embeddings = []
    return {
        str(chunk_id): embedding
        for chunk_id, embedding in zip(result_ids, embeddings)
        if embedding is not None
    }
