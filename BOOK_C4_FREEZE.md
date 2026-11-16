# C4 experiment freeze

Configuration: Baseline plus expert metadata configuration; raw-text embedding.

BM25: All searchable values from the union of baseline common fields and expert fields.

Reranker: 38 metadata fields.

Generation model: `qwen3:8b`. The committed run reports are the authoritative settings and results.

Included: the source context-table corpus and the matching filtered chunk JSON corpus, the exact 50-question benchmark, saved retrieval and generation JSONL outputs, run reports, available RAGAS score artifacts, the Top-5 expert review workbook, source code, and requirements files.

Not included: source PDFs, local Python environments, or Chroma SQLite vector databases. Saved retrieval outputs are included; reproducing a new dense retrieval run requires rebuilding the vector index from the committed chunk corpus.

Review note: the workbook filename says Top-5, and each row has five chunk IDs, but its column header currently says Top 3 and its freeze pane is at A25. This workbook was preserved as supplied.

Snapshot assembled from workspace state on 2026-11-16; parent commit: `99c947c5ce535de73661bf0a5188d8592c3987d2`.
