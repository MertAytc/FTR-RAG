# commented_text_embedding

RAGAS scores are not computed in this run. `ragas_input_generation.jsonl` contains the required inputs and null metric columns.

- Questions generated: 50/50
- Retrieval-metric eligible questions: 39
- Chunk corpus: `data\processed\chunks_context_tables_v1_workbook_metadata` (13,895 unique chunks)
- Dense collection: `ftr_chunks_inline_tables_commentary_embed_v1_20261030` (13,895 chunks)
- BM25: `all` semantic metadata; Dense: `commented_text_embedding`; reranker metadata fields: 14
- Retrieval elapsed: 959.3s; generation elapsed: 1941.2s

## Retrieval metrics (macro group recall, eligible retrieval-track questions)

| Stage | Relevance | n | Hit@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@10 | MRR@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DENSE | strict_canonical | 39 | 0.231 | 0.436 | 0.551 | 0.692 | 0.435 | 0.369 |
| DENSE | acceptable_full | 39 | 0.410 | 0.564 | 0.731 | 0.846 | 0.603 | 0.546 |
| DENSE | acceptable_plus_partial | 39 | 0.538 | 0.769 | 0.910 | 0.936 | 0.589 | 0.686 |
| BM25 | strict_canonical | 39 | 0.231 | 0.397 | 0.526 | 0.615 | 0.409 | 0.360 |
| BM25 | acceptable_full | 39 | 0.385 | 0.538 | 0.679 | 0.744 | 0.552 | 0.513 |
| BM25 | acceptable_plus_partial | 39 | 0.667 | 0.821 | 0.846 | 0.885 | 0.615 | 0.772 |
| RRF | strict_canonical | 39 | 0.308 | 0.551 | 0.577 | 0.744 | 0.519 | 0.469 |
| RRF | acceptable_full | 39 | 0.462 | 0.756 | 0.782 | 0.872 | 0.676 | 0.633 |
| RRF | acceptable_plus_partial | 39 | 0.692 | 0.897 | 0.910 | 0.962 | 0.699 | 0.822 |
| RERANKER | strict_canonical | 39 | 0.410 | 0.551 | 0.641 | 0.744 | 0.565 | 0.520 |
| RERANKER | acceptable_full | 39 | 0.667 | 0.769 | 0.795 | 0.872 | 0.758 | 0.741 |
| RERANKER | acceptable_plus_partial | 39 | 0.923 | 0.949 | 0.974 | 0.987 | 0.809 | 0.962 |

## Generation timing

- Mean TTFT: 5.876075950000668
- Median TTFT: 5.3453446999774314
- Mean generation response time: 38.749054020001786
- Median generation response time: 38.07463289995212

## Files

- RAGAS input: `data\processed\generation_first50_workbook_metadata_20261030\first50_commented_text_embedding\ragas_input_generation.jsonl`
- Per-question retrieval: `data\processed\generation_first50_workbook_metadata_20261030\first50_commented_text_embedding\retrieval_per_question.jsonl`
- Full report: `data\processed\generation_first50_workbook_metadata_20261030\first50_commented_text_embedding\run_report.json`
