# RAGAS evaluation report

- Judge: gpt-6-luna via Responses API; reasoning effort low.
- Embeddings: OpenAI Embeddings API `text-embedding-3-small`.
- Bootstrap: 10,000 resamples; seed 20261030.
- Judge calls recorded: 3475; tokens in/out: 4,512,854/897,331; estimated cost: $0.862385.
- Embedding calls recorded: 404; input tokens: 25,796; estimated cost: $0.000516.

## ragas_input_generation.jsonl

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 50 | 0.8410 | [0.7740, 0.8991] |
| answer_relevancy | 50 | 0.6703 | [0.5750, 0.7581] |
| llm_context_precision_with_reference | 50 | 0.7095 | [0.6152, 0.8002] |
| context_recall | 50 | 0.6500 | [0.5366, 0.7600] |
| context_entity_recall | 50 | 0.1448 | [0.0750, 0.2233] |
| factual_correctness_f1 | 50 | 0.3284 | [0.2392, 0.4202] |
| semantic_similarity | 50 | 0.7214 | [0.6823, 0.7560] |
| noise_sensitivity_relevant | 50 | 0.3757 | [0.2843, 0.4724] |

### By evaluation track

**excluded_invalid_gold**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 1 | 1.0000 | [1.0000, 1.0000] |
| answer_relevancy | 1 | 0.0000 | [0.0000, 0.0000] |
| llm_context_precision_with_reference | 1 | 1.0000 | [1.0000, 1.0000] |
| context_recall | 1 | 0.0000 | [0.0000, 0.0000] |
| context_entity_recall | 1 | 0.6667 | [0.6667, 0.6667] |
| factual_correctness_f1 | 1 | 0.0000 | [0.0000, 0.0000] |
| semantic_similarity | 1 | 0.6711 | [0.6711, 0.6711] |
| noise_sensitivity_relevant | 1 | 0.0000 | [0.0000, 0.0000] |

**metadata_attribution**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 5 | 0.7778 | [0.5333, 0.9778] |
| answer_relevancy | 5 | 0.4112 | [0.1583, 0.6641] |
| llm_context_precision_with_reference | 5 | 0.3411 | [0.1293, 0.5528] |
| context_recall | 5 | 0.3667 | [0.1667, 0.5333] |
| context_entity_recall | 5 | 0.4948 | [0.2000, 0.8048] |
| factual_correctness_f1 | 5 | 0.1600 | [0.0000, 0.4000] |
| semantic_similarity | 5 | 0.5013 | [0.3285, 0.6185] |
| noise_sensitivity_relevant | 5 | 0.4556 | [0.1500, 0.7611] |

**pending_gold**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 5 | 0.6648 | [0.3200, 0.9048] |
| answer_relevancy | 5 | 0.2792 | [0.0000, 0.6442] |
| llm_context_precision_with_reference | 5 | 0.1167 | [0.0000, 0.2500] |
| context_recall | 5 | 0.1000 | [0.0000, 0.3000] |
| context_entity_recall | 5 | 0.0000 | [0.0000, 0.0000] |
| factual_correctness_f1 | 5 | 0.4520 | [0.1800, 0.7600] |
| semantic_similarity | 5 | 0.6926 | [0.5758, 0.7971] |
| noise_sensitivity_relevant | 5 | 0.0000 | [0.0000, 0.0000] |

**retrieval**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 39 | 0.8676 | [0.8048, 0.9229] |
| answer_relevancy | 39 | 0.7708 | [0.6852, 0.8434] |
| llm_context_precision_with_reference | 39 | 0.8253 | [0.7454, 0.8960] |
| context_recall | 39 | 0.7735 | [0.6538, 0.8761] |
| context_entity_recall | 39 | 0.1051 | [0.0410, 0.1821] |
| factual_correctness_f1 | 39 | 0.3426 | [0.2382, 0.4510] |
| semantic_similarity | 39 | 0.7546 | [0.7240, 0.7850] |
| noise_sensitivity_relevant | 39 | 0.4233 | [0.3252, 0.5240] |


## ragas_input_generation.jsonl

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 50 | 0.8189 | [0.7499, 0.8834] |
| answer_relevancy | 50 | 0.6427 | [0.5443, 0.7333] |
| llm_context_precision_with_reference | 50 | 0.7066 | [0.6006, 0.8046] |
| context_recall | 50 | 0.6800 | [0.5667, 0.7833] |
| context_entity_recall | 50 | 0.1903 | [0.1117, 0.2787] |
| factual_correctness_f1 | 50 | 0.3258 | [0.2448, 0.4106] |
| semantic_similarity | 50 | 0.7185 | [0.6795, 0.7530] |
| noise_sensitivity_relevant | 50 | 0.4182 | [0.3220, 0.5154] |

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
| semantic_similarity | 1 | 0.6711 | [0.6711, 0.6711] |
| noise_sensitivity_relevant | 1 | 0.0000 | [0.0000, 0.0000] |

**metadata_attribution**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 5 | 0.5917 | [0.3583, 0.8250] |
| answer_relevancy | 5 | 0.4129 | [0.1568, 0.6689] |
| llm_context_precision_with_reference | 5 | 0.4254 | [0.2008, 0.6017] |
| context_recall | 5 | 0.4000 | [0.2000, 0.5667] |
| context_entity_recall | 5 | 0.4067 | [0.2300, 0.7067] |
| factual_correctness_f1 | 5 | 0.1740 | [0.0000, 0.4420] |
| semantic_similarity | 5 | 0.5013 | [0.3286, 0.6185] |
| noise_sensitivity_relevant | 5 | 0.5167 | [0.2000, 0.8333] |

**pending_gold**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 5 | 0.9714 | [0.9143, 1.0000] |
| answer_relevancy | 5 | 0.2732 | [0.0000, 0.6381] |
| llm_context_precision_with_reference | 5 | 0.1167 | [0.0000, 0.3500] |
| context_recall | 5 | 0.3000 | [0.1000, 0.5000] |
| context_entity_recall | 5 | 0.0000 | [0.0000, 0.0000] |
| factual_correctness_f1 | 5 | 0.3220 | [0.1500, 0.4720] |
| semantic_similarity | 5 | 0.6926 | [0.5757, 0.7971] |
| noise_sensitivity_relevant | 5 | 0.0857 | [0.0000, 0.2571] |

**retrieval**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 39 | 0.8238 | [0.7449, 0.8923] |
| answer_relevancy | 39 | 0.7361 | [0.6425, 0.8184] |
| llm_context_precision_with_reference | 39 | 0.8364 | [0.7428, 0.9156] |
| context_recall | 39 | 0.7821 | [0.6624, 0.8889] |
| context_entity_recall | 39 | 0.1833 | [0.0927, 0.2846] |
| factual_correctness_f1 | 39 | 0.3541 | [0.2567, 0.4523] |
| semantic_similarity | 39 | 0.7509 | [0.7195, 0.7816] |
| noise_sensitivity_relevant | 39 | 0.4590 | [0.3539, 0.5625] |


## Paired commentary effect (commented arm minus raw arm)

| Metric | n | Mean difference | 95% CI |
|---|---:|---:|---:|
| faithfulness | 50 | -0.0221 | [-0.0817, 0.0437] |
| answer_relevancy | 50 | -0.0276 | [-0.0660, -0.0026] |
| llm_context_precision_with_reference | 50 | -0.0029 | [-0.0644, 0.0482] |
| context_recall | 50 | 0.0300 | [-0.0133, 0.0767] |
| context_entity_recall | 50 | 0.0455 | [-0.0176, 0.1172] |
| factual_correctness_f1 | 50 | -0.0026 | [-0.0436, 0.0388] |
| semantic_similarity | 50 | -0.0029 | [-0.0066, 0.0004] |
| noise_sensitivity_relevant | 50 | 0.0425 | [-0.0211, 0.1070] |
