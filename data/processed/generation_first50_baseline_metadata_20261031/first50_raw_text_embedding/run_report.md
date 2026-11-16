# raw_text_embedding

RAGAS scores are not computed in this run. `ragas_input_generation.jsonl` contains the required inputs and null metric columns.

- Questions generated: 50/50
- Retrieval-metric eligible questions: 39
- Chunk corpus: `data\processed\chunks_latest_baseline_metadata_20261031` (13,895 unique chunks)
- Dense collection: `ftr_chunks_inline_tables_v1_20261030` (13,895 chunks)
- BM25: `all` semantic metadata; Dense: `raw_text_embedding`; reranker metadata fields: 25
- Retrieval elapsed: 75.7s; generation elapsed: 2001.7s

## Retrieval metrics (macro group recall, eligible retrieval-track questions)

| Stage | Relevance | n | Hit@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@10 | MRR@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DENSE | strict_canonical | 39 | 0.231 | 0.436 | 0.551 | 0.692 | 0.435 | 0.369 |
| DENSE | acceptable_full | 39 | 0.410 | 0.564 | 0.731 | 0.846 | 0.603 | 0.547 |
| DENSE | acceptable_plus_partial | 39 | 0.538 | 0.769 | 0.910 | 0.936 | 0.589 | 0.686 |
| BM25 | strict_canonical | 39 | 0.205 | 0.333 | 0.538 | 0.641 | 0.392 | 0.333 |
| BM25 | acceptable_full | 39 | 0.333 | 0.551 | 0.654 | 0.769 | 0.540 | 0.493 |
| BM25 | acceptable_plus_partial | 39 | 0.590 | 0.756 | 0.821 | 0.897 | 0.582 | 0.719 |
| RRF | strict_canonical | 39 | 0.205 | 0.474 | 0.590 | 0.731 | 0.462 | 0.399 |
| RRF | acceptable_full | 39 | 0.513 | 0.679 | 0.769 | 0.885 | 0.677 | 0.637 |
| RRF | acceptable_plus_partial | 39 | 0.718 | 0.872 | 0.923 | 0.962 | 0.697 | 0.831 |
| RERANKER | strict_canonical | 39 | 0.359 | 0.551 | 0.641 | 0.769 | 0.555 | 0.499 |
| RERANKER | acceptable_full | 39 | 0.564 | 0.731 | 0.769 | 0.897 | 0.719 | 0.678 |
| RERANKER | acceptable_plus_partial | 39 | 0.923 | 0.974 | 0.974 | 0.987 | 0.762 | 0.962 |

## Generation timing

- Mean TTFT: 5.547723472001962
- Median TTFT: 5.248951699992176
- Mean generation response time: 39.93267389000743
- Median generation response time: 38.50462750002043

## Files

- RAGAS input: `data\processed\generation_first50_baseline_metadata_20261031\first50_raw_text_embedding\ragas_input_generation.jsonl`
- Per-question retrieval: `data\processed\generation_first50_baseline_metadata_20261031\first50_raw_text_embedding\retrieval_per_question.jsonl`
- Full report: `data\processed\generation_first50_baseline_metadata_20261031\first50_raw_text_embedding\run_report.json`
