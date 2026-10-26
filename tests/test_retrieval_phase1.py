from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.retrieval import hybrid_retriever, vector_store
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.config import REFERENCE_PENALTY
from src.retrieval.indexer import load_chunks
from src.retrieval.retrieval_utils import normalize_chunk_id


def _reset_vector_cache(monkeypatch):
    monkeypatch.setattr(vector_store, "_CLIENT", None)
    monkeypatch.setattr(vector_store, "_COLLECTION", None)
    monkeypatch.setattr(vector_store, "_CACHE_KEY", None)


def test_normalize_chunk_id():
    assert normalize_chunk_id("  chunk-1  ") == "chunk-1"
    assert normalize_chunk_id(123) == "123"


def test_indexer_normalizes_and_warns_for_chunk_ids(tmp_path, monkeypatch, caplog):
    chunk_file = tmp_path / "sample.json"
    chunk_file.write_text(
        json.dumps({"chunks": [
            {"chunk_id": " id-1 ", "text": "first", "metadata": {}},
            {"chunk_id": "id-1", "text": "duplicate", "metadata": {}},
        ]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.retrieval.indexer.CHUNKS_DIR", tmp_path)

    chunks = load_chunks()

    assert [chunk["id"] for chunk in chunks] == ["id-1"]
    assert "Normalized chunk ID" in caplog.text


def test_bm25_normalizes_loaded_and_debug_ids(tmp_path):
    (tmp_path / "sample.json").write_text(
        json.dumps({"chunks": [
            {"chunk_id": " id-1 ", "text": "knee exercise", "metadata": {}},
            {"chunk_id": "id-2", "text": "neck mobility", "metadata": {}},
        ]}),
        encoding="utf-8",
    )
    retriever = BM25Retriever(chunks_dir=tmp_path)

    assert retriever.ids == ["id-1", "id-2"]
    assert len(retriever.ids) == len(set(retriever.ids))
    assert retriever.debug_rank("knee", {" id-1 "})["id-1"]["found"] is True
    assert retriever.reference_penalty == REFERENCE_PENALTY


def test_open_collection_does_not_create_missing_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "VECTOR_DB_PATH", tmp_path / "db")
    monkeypatch.setattr(vector_store, "COLLECTION_NAME", "missing_collection")
    _reset_vector_cache(monkeypatch)

    with pytest.raises(RuntimeError, match="was not found"):
        vector_store.open_collection()

    assert vector_store._CLIENT.list_collections() == []


def test_open_collection_rejects_empty_collection_and_reset_invalidates_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vector_store, "VECTOR_DB_PATH", tmp_path / "db")
    monkeypatch.setattr(vector_store, "COLLECTION_NAME", "empty_collection")
    _reset_vector_cache(monkeypatch)
    collection = vector_store.get_collection()

    with pytest.raises(RuntimeError, match="exists but is empty"):
        vector_store.open_collection()

    old_client = vector_store._CLIENT
    reset = vector_store.reset_collection()
    assert reset.count() == 0
    assert vector_store._CLIENT is not old_client
    with pytest.raises(RuntimeError, match="exists but is empty"):
        vector_store.open_collection()


def test_indexed_id_read_does_not_load_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "VECTOR_DB_PATH", tmp_path / "db")
    monkeypatch.setattr(vector_store, "COLLECTION_NAME", "indexed_collection")
    _reset_vector_cache(monkeypatch)
    collection = vector_store.get_collection()
    collection.upsert(
        ids=["chunk-1"],
        documents=["retrieval text"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"title": "Example"}],
    )

    assert vector_store.get_indexed_ids() == ["chunk-1"]
    assert vector_store.count() == 1
    vectors = vector_store.get_embeddings_by_ids(["chunk-1"])
    assert len(vectors["chunk-1"]) == 2


def test_hybrid_raises_with_id_mismatch_counts(monkeypatch):
    monkeypatch.setattr(hybrid_retriever, "get_indexed_ids", lambda: ["dense-id"])
    monkeypatch.setattr(
        hybrid_retriever,
        "open_collection",
        lambda: SimpleNamespace(metadata={"embedding_model": "unknown"}),
    )

    with pytest.raises(RuntimeError, match=r"dense-only=1.*BM25-only=1"):
        hybrid_retriever.HybridRetriever(
            bm25_retriever=SimpleNamespace(ids=["bm25-id"])
        )


def test_hybrid_rejects_embedding_model_mismatch(monkeypatch):
    monkeypatch.setattr(hybrid_retriever, "get_indexed_ids", lambda: ["same-id"])
    monkeypatch.setattr(
        hybrid_retriever,
        "open_collection",
        lambda: SimpleNamespace(metadata={"embedding_model": "other-model"}),
    )

    with pytest.raises(RuntimeError, match="embedding model mismatch"):
        hybrid_retriever.HybridRetriever(
            bm25_retriever=SimpleNamespace(ids=["same-id"])
        )


def test_hybrid_skips_unknown_embedding_metadata(monkeypatch):
    monkeypatch.setattr(hybrid_retriever, "get_indexed_ids", lambda: ["same-id"])
    monkeypatch.setattr(
        hybrid_retriever,
        "open_collection",
        lambda: SimpleNamespace(metadata={"embedding_model": "unknown"}),
    )

    instance = hybrid_retriever.HybridRetriever(
        bm25_retriever=SimpleNamespace(ids=["same-id"])
    )
    assert instance.bm25.ids == ["same-id"]


def test_fusion_defaults_preserve_weighted_rrf_scores():
    instance = object.__new__(hybrid_retriever.HybridRetriever)
    instance.rrf_k = 60
    instance.dense_weight = 1.0
    instance.bm25_weight = 0.85
    rows = instance._rrf_fusion(
        [{"id": " id-1 ", "rank": 1, "score": 0.2, "document": "d", "metadata": {}}],
        [{"id": "id-1", "rank": 1, "score": 2.0, "document": "b", "metadata": {}}],
    )

    assert [row["id"] for row in rows] == ["id-1"]
    assert rows[0]["rrf_score"] == pytest.approx(1 / 61 + 0.85 / 61)
