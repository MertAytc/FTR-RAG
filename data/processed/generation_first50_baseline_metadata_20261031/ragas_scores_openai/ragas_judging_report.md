# RAGAS evaluation report

- Judge: gpt-6-luna via Responses API; reasoning effort low.
- Embeddings: OpenAI Embeddings API `text-embedding-3-small`.
- Bootstrap: 10,000 resamples; seed 20261031.
- Judge calls recorded: 1641; tokens in/out: 2,097,103/412,500; estimated cost: $0.415960.
- Embedding calls recorded: 200; input tokens: 12,911; estimated cost: $0.000258.

## ragas_input_generation.jsonl

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 50 | 0.8366 | [0.7769, 0.8911] |
| answer_relevancy | 50 | 0.7108 | [0.6270, 0.7848] |
| llm_context_precision_with_reference | 50 | 0.7476 | [0.6453, 0.8421] |
| context_recall | 50 | 0.6733 | [0.5633, 0.7800] |
| context_entity_recall | 50 | 0.1707 | [0.0959, 0.2550] |
| factual_correctness_f1 | 50 | 0.3300 | [0.2554, 0.4070] |
| semantic_similarity | 50 | 0.7148 | [0.6666, 0.7565] |
| noise_sensitivity_relevant | 50 | 0.3339 | [0.2448, 0.4237] |

### By evaluation track

**excluded_invalid_gold**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 1 | 1.0000 | [1.0000, 1.0000] |
| answer_relevancy | 1 | 0.0000 | [0.0000, 0.0000] |
| llm_context_precision_with_reference | 1 | 0.0000 | [0.0000, 0.0000] |
| context_recall | 1 | 0.0000 | [0.0000, 0.0000] |
| context_entity_recall | 1 | 0.3333 | [0.3333, 0.3333] |
| factual_correctness_f1 | 1 | 0.0000 | [0.0000, 0.0000] |
| semantic_similarity | 1 | 0.7001 | [0.7001, 0.7001] |
| noise_sensitivity_relevant | 1 | 0.0000 | [0.0000, 0.0000] |

**metadata_attribution**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 5 | 0.6729 | [0.4900, 0.8500] |
| answer_relevancy | 5 | 0.8163 | [0.6608, 0.9261] |
| llm_context_precision_with_reference | 5 | 0.3037 | [0.1694, 0.4438] |
| context_recall | 5 | 0.2333 | [0.0667, 0.4000] |
| context_entity_recall | 5 | 0.4948 | [0.1967, 0.8114] |
| factual_correctness_f1 | 5 | 0.3460 | [0.2180, 0.5440] |
| semantic_similarity | 5 | 0.5941 | [0.3471, 0.7865] |
| noise_sensitivity_relevant | 5 | 0.2257 | [0.0400, 0.4114] |

**pending_gold**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 5 | 0.8538 | [0.6538, 1.0000] |
| answer_relevancy | 5 | 0.2266 | [0.0000, 0.4917] |
| llm_context_precision_with_reference | 5 | 0.2667 | [0.0000, 0.6667] |
| context_recall | 5 | 0.2000 | [0.0000, 0.4000] |
| context_entity_recall | 5 | 0.1000 | [0.0000, 0.3000] |
| factual_correctness_f1 | 5 | 0.3440 | [0.1500, 0.5080] |
| semantic_similarity | 5 | 0.6833 | [0.5712, 0.7811] |
| noise_sensitivity_relevant | 5 | 0.0000 | [0.0000, 0.0000] |

**retrieval**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 39 | 0.8513 | [0.7833, 0.9111] |
| answer_relevancy | 39 | 0.7775 | [0.7101, 0.8344] |
| llm_context_precision_with_reference | 39 | 0.8853 | [0.8080, 0.9481] |
| context_recall | 39 | 0.8077 | [0.7009, 0.9017] |
| context_entity_recall | 39 | 0.1340 | [0.0577, 0.2237] |
| factual_correctness_f1 | 39 | 0.3346 | [0.2462, 0.4267] |
| semantic_similarity | 39 | 0.7347 | [0.6881, 0.7765] |
| noise_sensitivity_relevant | 39 | 0.3991 | [0.2967, 0.5018] |

