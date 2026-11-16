# C3 experiment freeze

Configuration: Workbook metadata configuration; commentary-augmented dense embedding and generation context.

BM25: All semantic metadata; commented_text is present in the shared BM25 index.

Reranker: 14 selected metadata fields; commentary is not a direct reranker field.

Generation model: `qwen3:8b`. The committed run reports are the authoritative settings and results.

Included: the matching chunk JSON corpus, the exact 50-question benchmark, saved retrieval and generation JSONL outputs, run reports, available RAGAS score artifacts, the Top-5 expert review workbook, source code, and requirements files.

Not included: source PDFs, local Python environments, or Chroma SQLite vector databases. Saved retrieval outputs are included; reproducing a new dense retrieval run requires rebuilding the vector index from the committed chunk corpus.

C3 uses `chunk.text + commented_text` for dense embeddings on commented chunks and includes commentary in generation context. The shared BM25 index uses `metadata_mode=all`, which also indexes `commented_text`; the reranker does not receive commentary directly.

Snapshot assembled from workspace state on 2026-11-16; parent commit: `99c947c5ce535de73661bf0a5188d8592c3987d2`.
