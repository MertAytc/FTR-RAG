from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from src.retrieval.config import (
    BENCHMARK_PATH,
    CHUNKS_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from src.retrieval.embedding import embed_texts
from src.retrieval.retrieval_utils import normalize_chunk_id
from src.retrieval.vector_store import add_chunks, reset_collection

logger = logging.getLogger(__name__)
BATCH_SIZE = int(os.getenv("FTR_INDEXER_BATCH_SIZE", "32"))

METADATA_FIELDS = [
    "document_id",
    "title",
    "organization",
    "year",
    "domain",
    "condition",
    "population",
    "section",
    "subsection",
    "page",
    "source_type",
]


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Keep Chroma-compatible values for the metadata used by retrieval."""
    cleaned = {}
    for field in METADATA_FIELDS:
        value = metadata.get(field)
        if value is None:
            value = ""
        if isinstance(value, (str, int, float, bool)):
            cleaned[field] = value
        else:
            cleaned[field] = str(value)
    return cleaned


def load_chunks() -> List[Dict[str, Any]]:
    """Load chunk records from the configured processed-chunks directory."""
    all_chunks = []
    seen_ids = set()
    json_files = sorted(CHUNKS_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No chunk JSON files found in {CHUNKS_DIR}")

    duplicate_count = 0
    for json_file in json_files:
        if json_file.name == "_quality_report.json":
            continue

        logger.info("Reading chunks: %s", json_file.name)
        with json_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            chunks = data.get("chunks", [])
        elif isinstance(data, list):
            chunks = data
        else:
            logger.warning("Unexpected JSON format; skipping %s", json_file.name)
            continue

        if not isinstance(chunks, list):
            logger.warning("The 'chunks' field is not a list; skipping %s", json_file.name)
            continue

        logger.info("Chunk records in %s: %d", json_file.name, len(chunks))
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            raw_chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            if not raw_chunk_id or not text:
                continue

            chunk_id = normalize_chunk_id(raw_chunk_id)
            if chunk_id != raw_chunk_id:
                logger.warning(
                    "Normalized chunk ID in %s: %r -> %r",
                    json_file.name,
                    raw_chunk_id,
                    chunk_id,
                )

            if chunk_id in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(chunk_id)

            all_chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": clean_metadata(chunk.get("metadata") or {}),
            })

    logger.info("Unique chunks loaded: %d", len(all_chunks))
    logger.info("Duplicate chunk IDs skipped: %d", duplicate_count)
    return all_chunks


def verify_index(collection, expected_chunks: List[Dict[str, Any]]) -> None:
    """Verify that the dense index exactly matches the processed chunk IDs."""
    expected_ids = {normalize_chunk_id(chunk["id"]) for chunk in expected_chunks}
    indexed_ids = {
        normalize_chunk_id(chunk_id)
        for chunk_id in collection.get(include=[])["ids"]
    }
    missing_ids = sorted(expected_ids - indexed_ids)
    extra_ids = sorted(indexed_ids - expected_ids)

    logger.info(
        "Dense index verification: corpus=%d index=%d missing=%d extra=%d",
        len(expected_ids), len(indexed_ids), len(missing_ids), len(extra_ids),
    )
    if missing_ids or extra_ids:
        if missing_ids:
            logger.error("First missing IDs: %s", missing_ids[:10])
        if extra_ids:
            logger.error("First extra IDs: %s", extra_ids[:10])
        raise RuntimeError("Dense index IDs do not match the processed chunk corpus.")

    if BENCHMARK_PATH.exists():
        with BENCHMARK_PATH.open("r", encoding="utf-8-sig") as file:
            benchmark_data = json.load(file)
        questions = (
            benchmark_data.get("questions", [])
            if isinstance(benchmark_data, dict)
            else benchmark_data
        )

        gold_ids = []
        for question in questions:
            if not isinstance(question, dict):
                continue
            gold_id = question.get("gold_chunk_id")
            if gold_id:
                gold_ids.append(normalize_chunk_id(gold_id))
            multi_gold_ids = question.get("gold_chunk_ids", [])
            if isinstance(multi_gold_ids, list):
                gold_ids.extend(
                    normalize_chunk_id(value)
                    for value in multi_gold_ids
                    if value
                )

        gold_ids = list(dict.fromkeys(gold_ids))
        missing_gold = [chunk_id for chunk_id in gold_ids if chunk_id not in indexed_ids]
        logger.info(
            "Benchmark gold ID verification (%s): total=%d found=%d missing=%d",
            BENCHMARK_PATH,
            len(gold_ids),
            len(gold_ids) - len(missing_gold),
            len(missing_gold),
        )
        if missing_gold:
            logger.error("Missing benchmark gold IDs: %s", missing_gold[:20])
            raise RuntimeError("Some benchmark gold chunk IDs are missing from the dense index.")
    else:
        logger.info("Benchmark file not found; skipping gold ID verification: %s", BENCHMARK_PATH)

    logger.info("Dense index verification passed.")


def index_chunks() -> None:
    """Embed all unique processed chunks and store them in ChromaDB."""
    logger.info("Starting FTR-RAG dense indexing.")
    chunks = load_chunks()
    if not chunks:
        logger.warning("No chunks found to index.")
        return
    if BATCH_SIZE <= 0:
        raise ValueError("FTR_INDEXER_BATCH_SIZE must be greater than zero.")

    logger.info("Chroma collection: %s", COLLECTION_NAME)
    logger.info("Chroma distance space: cosine (fresh collection)")
    collection = reset_collection()
    total = len(chunks)
    logger.info("Cleared previous collection; chunks to index: %d", total)
    logger.info("Embedding model: %s", EMBEDDING_MODEL)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]
        ids = [normalize_chunk_id(chunk["id"]) for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]
        end = min(start + BATCH_SIZE, total)
        logger.info("Embedding batch %d-%d of %d", start + 1, end, total)

        embeddings = embed_texts(texts, input_type="document")
        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding count does not match the chunk batch size.")
        add_chunks(collection, ids, texts, embeddings, metadatas)

    logger.info("Indexing completed. Chroma records: %d", collection.count())
    verify_index(collection, chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    index_chunks()
