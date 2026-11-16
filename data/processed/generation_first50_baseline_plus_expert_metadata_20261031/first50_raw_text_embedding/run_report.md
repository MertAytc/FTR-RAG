# raw_text_embedding

RAGAS scores are not computed in this run. `ragas_input_generation.jsonl` contains the required inputs and null metric columns.

- Questions generated: 50/50
- Retrieval-metric eligible questions: 39
- Chunk corpus: `data\processed\chunks_latest_baseline_plus_expert_metadata_20261031` (13,895 unique chunks)
- Dense collection: `ftr_chunks_inline_tables_v1_20261030` (13,895 chunks)
- BM25: `all` semantic metadata; Dense: `raw_text_embedding`; reranker metadata fields: 38
- Retrieval elapsed: 79.5s; generation elapsed: 1915.6s

## Retrieval metrics (macro group recall, eligible retrieval-track questions)

| Stage | Relevance | n | Hit@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@10 | MRR@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DENSE | strict_canonical | 39 | 0.231 | 0.436 | 0.551 | 0.692 | 0.435 | 0.369 |
| DENSE | acceptable_full | 39 | 0.410 | 0.564 | 0.731 | 0.846 | 0.603 | 0.547 |
| DENSE | acceptable_plus_partial | 39 | 0.538 | 0.769 | 0.910 | 0.936 | 0.589 | 0.686 |
| BM25 | strict_canonical | 39 | 0.179 | 0.333 | 0.500 | 0.654 | 0.387 | 0.322 |
| BM25 | acceptable_full | 39 | 0.333 | 0.551 | 0.641 | 0.782 | 0.546 | 0.494 |
| BM25 | acceptable_plus_partial | 39 | 0.615 | 0.782 | 0.833 | 0.910 | 0.593 | 0.733 |
| RRF | strict_canonical | 39 | 0.282 | 0.538 | 0.603 | 0.744 | 0.503 | 0.452 |
| RRF | acceptable_full | 39 | 0.513 | 0.718 | 0.782 | 0.846 | 0.672 | 0.647 |
| RRF | acceptable_plus_partial | 39 | 0.718 | 0.859 | 0.923 | 0.936 | 0.697 | 0.821 |
| RERANKER | strict_canonical | 39 | 0.385 | 0.513 | 0.603 | 0.679 | 0.522 | 0.497 |
| RERANKER | acceptable_full | 39 | 0.641 | 0.731 | 0.731 | 0.833 | 0.720 | 0.716 |
| RERANKER | acceptable_plus_partial | 39 | 0.949 | 0.936 | 0.962 | 0.962 | 0.778 | 0.962 |

## Generation timing

- Mean TTFT: 5.390264234001515
- Median TTFT: 4.813495999958832
- Mean generation response time: 38.259288984001614
- Median generation response time: 37.98149419995025

## Files

- RAGAS input: `data\processed\generation_first50_baseline_plus_expert_metadata_20261031\first50_raw_text_embedding\ragas_input_generation.jsonl`
- Per-question retrieval: `data\processed\generation_first50_baseline_plus_expert_metadata_20261031\first50_raw_text_embedding\retrieval_per_question.jsonl`
- Full report: `data\processed\generation_first50_baseline_plus_expert_metadata_20261031\first50_raw_text_embedding\run_report.json`
