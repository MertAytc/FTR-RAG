"""Shared configuration for retrieval modules."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


CHUNKS_DIR = _configured_path(
    "FTR_CHUNKS_DIR",
    PROJECT_ROOT / "data" / "processed" / "chunks",
)
VECTOR_DB_PATH = _configured_path(
    "FTR_VECTOR_DB_DIR",
    PROJECT_ROOT / "data" / "processed" / "vector_db",
)
COLLECTION_NAME = os.getenv("FTR_VECTOR_COLLECTION", "ftr_chunks").strip() or "ftr_chunks"
EMBEDDING_MODEL = os.getenv("FTR_EMBEDDING_MODEL", "qwen3-embedding:0.6b").strip()
BENCHMARK_PATH = _configured_path(
    "FTR_BENCHMARK_PATH",
    PROJECT_ROOT / "ftr_rag_20_single_gold_chunk_test_v3_4.json",
)
REFERENCE_PENALTY = 0.25
