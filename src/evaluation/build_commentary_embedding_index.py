"""Clone the inline-table Chroma index and re-embed only commented table chunks."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.embedding import embed_texts


LOG = logging.getLogger("commentary_embedding_index")


def _load_commentary_chunks(chunks_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(chunks_dir.glob("*.chunks.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for chunk in data.get("chunks", []):
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            text = str(chunk.get("text") or "")
            commentary = str(chunk.get("commented_text") or "").strip()
            if chunk_id and text.strip() and commentary:
                rows.append({
                    "id": chunk_id,
                    "embedding_input": f"{text}\n\n{commentary}",
                })
    return rows


def build_index(
    source_db: Path,
    source_collection_name: str,
    target_db: Path,
    target_collection_name: str,
    chunks_dir: Path,
    batch_size: int = 128,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    source_client = chromadb.PersistentClient(path=str(source_db))
    source = source_client.get_collection(source_collection_name)
    source_count = int(source.count())
    if source_count <= 0:
        raise RuntimeError(f"Source collection is empty: {source_collection_name}")

    target_db.mkdir(parents=True, exist_ok=True)
    target_client = chromadb.PersistentClient(path=str(target_db))
    target = target_client.get_or_create_collection(
        name=target_collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": (source.metadata or {}).get("embedding_model", "unknown"),
            "embedding_variant": "raw_text_plus_commented_text_for_commented_chunks",
            "source_collection": source_collection_name,
        },
    )
    if target.count() != 0:
        raise RuntimeError(
            f"Target collection already contains {target.count()} records: "
            f"{target_collection_name}. Choose a new collection name."
        )

    for offset in range(0, source_count, batch_size):
        batch = source.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = list(batch.get("ids") or [])
        documents = list(batch.get("documents") or [])
        metadatas = list(batch.get("metadatas") or [])
        raw_embeddings = batch.get("embeddings")
        embeddings = raw_embeddings.tolist() if hasattr(raw_embeddings, "tolist") else list(raw_embeddings or [])
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise RuntimeError(f"Source batch at offset {offset} has inconsistent lengths.")
        target.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        LOG.info("Copied dense index records %d-%d/%d", offset + 1, offset + len(ids), source_count)

    commentary_rows = _load_commentary_chunks(chunks_dir)
    if not commentary_rows:
        raise RuntimeError(f"No commented table chunks found in {chunks_dir}")
    target_ids = set(target.get(include=[]) ["ids"])
    missing = sorted({row["id"] for row in commentary_rows} - target_ids)
    if missing:
        raise RuntimeError(f"Commented chunks missing from source index: {missing[:10]}")

    for offset in range(0, len(commentary_rows), batch_size):
        rows = commentary_rows[offset:offset + batch_size]
        embeddings = embed_texts(
            [row["embedding_input"] for row in rows],
            input_type="document",
        )
        if len(embeddings) != len(rows):
            raise RuntimeError("Embedding count does not match the commentary batch.")
        target.update(
            ids=[row["id"] for row in rows],
            embeddings=embeddings,
        )
        LOG.info(
            "Re-embedded commented table chunks %d-%d/%d",
            offset + 1,
            offset + len(rows),
            len(commentary_rows),
        )

    if target.count() != source_count:
        raise RuntimeError(
            f"Cloned index count differs: source={source_count}, target={target.count()}"
        )

    manifest = {
        "source_db": str(source_db),
        "source_collection": source_collection_name,
        "target_db": str(target_db),
        "target_collection": target_collection_name,
        "embedding_model": (source.metadata or {}).get("embedding_model", "unknown"),
        "record_count": target.count(),
        "commented_chunks_reembedded": len(commentary_rows),
        "embedding_input": "chunk.text + two newlines + chunk.commented_text",
        "stored_documents": "unchanged original chunk.text",
    }
    manifest_path = target_db / "embedding_variant_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--source-collection", required=True)
    parser.add_argument("--target-db", type=Path, required=True)
    parser.add_argument("--target-collection", required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s | %(message)s")
    manifest = build_index(
        args.source_db,
        args.source_collection,
        args.target_db,
        args.target_collection,
        args.chunks_dir,
        args.batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
