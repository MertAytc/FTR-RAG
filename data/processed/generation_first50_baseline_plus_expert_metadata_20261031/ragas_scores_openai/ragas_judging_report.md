# RAGAS evaluation report

- Judge: gpt-6-luna via Responses API; reasoning effort low.
- Embeddings: OpenAI Embeddings API `text-embedding-3-small`.
- Bootstrap: 10,000 resamples; seed 20261031.
- Judge calls recorded: 1599; tokens in/out: 1,984,340/390,126; estimated cost: $0.393497.
- Embedding calls recorded: 200; input tokens: 13,038; estimated cost: $0.000261.

## ragas_input_generation.jsonl

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 50 | 0.8165 | [0.7578, 0.8725] |
| answer_relevancy | 50 | 0.7216 | [0.6341, 0.7985] |
| llm_context_precision_with_reference | 50 | 0.7148 | [0.6117, 0.8120] |
| context_recall | 50 | 0.6667 | [0.5583, 0.7700] |
| context_entity_recall | 50 | 0.1560 | [0.0823, 0.2407] |
| factual_correctness_f1 | 50 | 0.2616 | [0.1924, 0.3338] |
| semantic_similarity | 50 | 0.7378 | [0.7072, 0.7669] |
| noise_sensitivity_relevant | 50 | 0.4061 | [0.2998, 0.5099] |

### By evaluation track

**excluded_invalid_gold**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 1 | 1.0000 | [1.0000, 1.0000] |
| answer_relevancy | 1 | 0.0000 | [0.0000, 0.0000] |
| llm_context_precision_with_reference | 1 | 0.0000 | [0.0000, 0.0000] |
| context_recall | 1 | 0.0000 | [0.0000, 0.0000] |
| context_entity_recall | 1 | 0.0000 | [0.0000, 0.0000] |
| factual_correctness_f1 | 1 | 0.0000 | [0.0000, 0.0000] |
| semantic_similarity | 1 | 0.6849 | [0.6849, 0.6849] |
| noise_sensitivity_relevant | 1 | 0.0000 | [0.0000, 0.0000] |

**metadata_attribution**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 5 | 0.7683 | [0.6200, 0.9100] |
| answer_relevancy | 5 | 0.4198 | [0.1284, 0.7113] |
| llm_context_precision_with_reference | 5 | 0.2173 | [0.0459, 0.3992] |
| context_recall | 5 | 0.2167 | [0.0000, 0.5167] |
| context_entity_recall | 5 | 0.4233 | [0.1967, 0.7333] |
| factual_correctness_f1 | 5 | 0.2200 | [0.0900, 0.3500] |
| semantic_similarity | 5 | 0.6604 | [0.5685, 0.7810] |
| noise_sensitivity_relevant | 5 | 0.3962 | [0.0667, 0.7257] |

**pending_gold**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 5 | 0.9778 | [0.9333, 1.0000] |
| answer_relevancy | 5 | 0.2770 | [0.0000, 0.6427] |
| llm_context_precision_with_reference | 5 | 0.3000 | [0.1000, 0.5000] |
| context_recall | 5 | 0.3000 | [0.1000, 0.5000] |
| context_entity_recall | 5 | 0.1000 | [0.0000, 0.3000] |
| factual_correctness_f1 | 5 | 0.2840 | [0.1240, 0.4440] |
| semantic_similarity | 5 | 0.6775 | [0.5377, 0.7900] |
| noise_sensitivity_relevant | 5 | 0.0667 | [0.0000, 0.2000] |

**retrieval**

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| faithfulness | 39 | 0.7973 | [0.7267, 0.8642] |
| answer_relevancy | 39 | 0.8358 | [0.7951, 0.8731] |
| llm_context_precision_with_reference | 39 | 0.8501 | [0.7621, 0.9254] |
| context_recall | 39 | 0.7885 | [0.6838, 0.8825] |
| context_entity_recall | 39 | 0.1329 | [0.0543, 0.2248] |
| factual_correctness_f1 | 39 | 0.2708 | [0.1913, 0.3562] |
| semantic_similarity | 39 | 0.7569 | [0.7267, 0.7854] |
| noise_sensitivity_relevant | 39 | 0.4613 | [0.3441, 0.5790] |

