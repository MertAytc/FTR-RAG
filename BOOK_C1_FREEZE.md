# C1 experiment freeze

Configuration: Baseline metadata configuration; raw-text embedding.

BM25: All searchable values from the intersection of baseline and latest metadata keys.

Reranker: 25 metadata fields.

Generation model: `qwen3:8b`. The committed run reports are the authoritative settings and results.

Included: the source context-table corpus and the matching filtered chunk JSON corpus, the exact 50-question benchmark, saved retrieval and generation JSONL outputs, run reports, available RAGAS score artifacts, the Top-5 expert review workbook, source code, and requirements files.

Not included: source PDFs, local Python environments, or Chroma SQLite vector databases. Saved retrieval outputs are included; reproducing a new dense retrieval run requires rebuilding the vector index from the committed chunk corpus.

Snapshot assembled from workspace state on 2026-11-16; parent commit: `99c947c5ce535de73661bf0a5188d8592c3987d2`.
