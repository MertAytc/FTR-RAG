import json
from pathlib import Path
from typing import Dict, Any, List

from src.retrieval.embedding import embed_texts, EMBEDDING_MODEL
from src.retrieval.vector_store import reset_collection, add_chunks


CHUNKS_DIR = Path("data/processed/chunks")

BATCH_SIZE = int(__import__("os").getenv("FTR_INDEXER_BATCH_SIZE", "32"))


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


def verify_index(collection, expected_chunks: List[Dict[str, Any]]):
    """Verify that the dense index exactly matches the processed chunk corpus."""
    expected_ids = {chunk["id"] for chunk in expected_chunks}
    indexed_ids = set(collection.get()["ids"])
    missing_ids = sorted(expected_ids - indexed_ids)
    extra_ids = sorted(indexed_ids - expected_ids)

    print()
    print("=" * 60)
    print("DENSE INDEX DOĞRULAMA")
    print("=" * 60)
    print(f"Chunk corpus: {len(expected_ids)}")
    print(f"Dense index: {len(indexed_ids)}")
    print(f"Missing IDs: {len(missing_ids)}")
    print(f"Extra IDs: {len(extra_ids)}")

    if missing_ids or extra_ids:
        if missing_ids:
            print(f"İlk eksik ID'ler: {missing_ids[:10]}")
        if extra_ids:
            print(f"İlk fazla ID'ler: {extra_ids[:10]}")
        raise RuntimeError("Dense index ile chunk corpus birebir eşleşmiyor.")

    # v3.4 single-gold benchmark gold IDs are checked as a separate
    # integrity test. This prevents running the benchmark against a stale
    # or partial dense index.
    benchmark_path = Path("ftr_rag_20_single_gold_chunk_test_v3_4.json")
    if benchmark_path.exists():
        with benchmark_path.open("r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

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
                gold_ids.append(str(gold_id).strip())

        gold_ids = list(dict.fromkeys(gold_ids))
        found_gold = [gid for gid in gold_ids if gid in indexed_ids]
        missing_gold = [gid for gid in gold_ids if gid not in indexed_ids]

        print()
        print("BENCHMARK GOLD ID DOĞRULAMA")
        print(f"Gold IDs: {len(gold_ids)}")
        print(f"Found: {len(found_gold)}")
        print(f"Missing: {len(missing_gold)}")

        if missing_gold:
            print(f"Eksik gold ID'ler: {missing_gold[:20]}")
            raise RuntimeError(
                "v3.4 benchmark gold chunk'larının tamamı dense indexte bulunmuyor."
            )

        print("Gold doğrulama: BAŞARILI")

    print("Doğrulama: BAŞARILI")


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

    # Rebuild only the collection selected by FTR_VECTOR_COLLECTION.
    # Default remains ftr_chunks; for E5 use a separate collection.
    # vector_store.reset_collection() recreates the production collection
    # used by HybridRetriever.
    print("Chroma collection: ftr_chunks")
    print("Chroma distance space: cosine (fresh collection)")
    collection = reset_collection()

    total = len(chunks)

    print()
    print("ChromaDB eski collection temizlendi.")
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

        embeddings = embed_texts(texts, input_type="document")

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
    verify_index(collection, chunks)


if __name__ == "__main__":
    index_chunks()