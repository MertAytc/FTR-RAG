import json
from pathlib import Path
from typing import Dict, Any, List

from src.retrieval.embedding import embed_texts, EMBEDDING_MODEL
from src.retrieval.vector_store import get_collection, add_chunks


CHUNKS_DIR = Path("data/processed/chunks")

BATCH_SIZE = 32


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
    """
    ChromaDB metadata'sında None değerleri bırakmamak için
    metadata'yı temizler.
    """

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
    """
    data/processed/chunks altındaki JSON dosyalarından
    chunk'ları yükler.

    Chunker çıktısı:
    {
        ...
        "chunks": [
            {
                "text": "...",
                "metadata": {...},
                "chunk_id": "..."
            }
        ]
    }
    """

    all_chunks = []
    seen_ids = set()

    json_files = sorted(CHUNKS_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"Chunk JSON bulunamadı: {CHUNKS_DIR}"
        )

    duplicate_count = 0

    for json_file in json_files:

        # Quality report'u chunk dosyası olarak okuma
        if json_file.name == "_quality_report.json":
            continue

        print(f"Okunuyor: {json_file.name}")

        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Normal chunker çıktısı
        if isinstance(data, dict):
            chunks = data.get("chunks", [])

        # Liste formatını da destekle
        elif isinstance(data, list):
            chunks = data

        else:
            print(
                "  UYARI: Beklenmeyen JSON formatı, atlandı."
            )
            continue

        if not isinstance(chunks, list):
            print(
                "  UYARI: 'chunks' alanı liste değil, atlandı."
            )
            continue

        print(f"  Chunk sayısı: {len(chunks)}")

        for chunk in chunks:

            if not isinstance(chunk, dict):
                continue

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")

            if not chunk_id or not text:
                continue

            # Duplicate chunk ID kontrolü
            if chunk_id in seen_ids:
                duplicate_count += 1
                continue

            seen_ids.add(chunk_id)

            metadata = clean_metadata(
                chunk.get("metadata", {})
            )

            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                }
            )

    print()
    print(f"Toplam benzersiz chunk: {len(all_chunks)}")
    print(f"Atlanan duplicate chunk: {duplicate_count}")

    return all_chunks


def index_chunks():
    """
    Tüm benzersiz chunk'ları embedding'e çevirir
    ve ChromaDB'ye kaydeder.
    """

    print("=" * 60)
    print("FTR-RAG DENSE INDEXING")
    print("=" * 60)

    chunks = load_chunks()

    if not chunks:
        print()
        print("UYARI: Indexlenecek chunk bulunamadı.")
        return

    collection = get_collection()

    total = len(chunks)

    print()
    print(f"ChromaDB mevcut kayıt: {collection.count()}")
    print(f"Indexlenecek chunk: {total}")
    print(f"Embedding modeli: {EMBEDDING_MODEL}")
    print()

    for start in range(0, total, BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        ids = [
            chunk["id"]
            for chunk in batch
        ]

        metadatas = [
            chunk["metadata"]
            for chunk in batch
        ]

        end = min(start + BATCH_SIZE, total)

        print(
            f"Embedding: {start + 1}-{end}/{total}"
        )

        embeddings = embed_texts(texts)

        if len(embeddings) != len(batch):
            raise RuntimeError(
                "Embedding sayısı ile chunk sayısı eşleşmiyor."
            )

        add_chunks(
            collection=collection,
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print()
    print("=" * 60)
    print("INDEXLEME TAMAMLANDI")
    print("=" * 60)
    print(f"ChromaDB kayıt sayısı: {collection.count()}")


if __name__ == "__main__":
    index_chunks()