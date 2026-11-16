from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


RERANKER_MODEL = os.getenv(
    "FTR_RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3",
)

RERANKER_METADATA_FIELDS = (
    "doc_name",
    "guideline_title",
    "guideline_source_authority",
    "guideline_filename_year",
    "guideline_version",
    "guideline_publication_date",
    "guideline_last_updated",
    "guideline_diseases_mentioned",
    "guideline_diseases_covered",
    "guideline_conditions_covered",
    "guideline_target_population",
    "guideline_age_group",
    "guideline_clinical_phase",
    "guideline_treatment_context",
    "guideline_injury_context",
    "guideline_summary",
    "chunk_diseases",
    "chunk_conditions",
    "chunk_interventions",
    "chunk_population",
    "chunk_clinical_phase",
    "chunk_recommendation_status",
    "recommendation_strength",
    "evidence_level",
    "chunk_contraindications",
    "chunk_adverse_effects",
    "chunk_clinical_warnings",
)

_MAX_METADATA_CONTEXT_CHARS = 1600
_MAX_METADATA_VALUE_CHARS = 260


def _metadata_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        value = "; ".join(str(item).strip() for item in value if str(item).strip())
    elif isinstance(value, dict):
        value = "; ".join(f"{key}: {item}" for key, item in value.items())
    else:
        value = str(value).strip()
    if value.casefold() in {"unknown", "not specified", "unspecified", "n/a", "none"}:
        return ""
    return value[:_MAX_METADATA_VALUE_CHARS]


def _reranker_document(item: Dict[str, Any], include_metadata_context: bool) -> str:
    """Build reranker text; metadata is an opt-in experimental prefix."""
    document = str(item.get("document", ""))
    if not include_metadata_context:
        return document
    metadata = item.get("metadata") or {}
    fields: List[str] = []
    for field in RERANKER_METADATA_FIELDS:
        value = _metadata_value(metadata.get(field))
        if value:
            label = field.removeprefix("guideline_").replace("_", " ")
            fields.append(f"{label}: {value}")
    if not fields:
        return document
    prefix = "GUIDELINE METADATA\n" + "\n".join(fields)
    prefix = prefix[:_MAX_METADATA_CONTEXT_CHARS]
    return prefix + "\n\nCHUNK TEXT\n" + document


class CrossEncoderReranker:
    """Cross-encoder reranker for an already-retrieved candidate pool.

    Query ve chunk birlikte değerlendirilir. Reranker retrieval yapmaz;
    RRF/Dense/BM25 tarafından oluşturulan adayları yeniden sıralar.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        if CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers kurulu değil. "
                "Kurulum: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        include_metadata_context: bool = False,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        top_k = max(1, min(top_k, len(candidates)))

        pairs = [
            (query, _reranker_document(item, include_metadata_context))
            for item in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked = []
        for original_rank, (item, score) in enumerate(
            zip(candidates, scores),
            start=1,
        ):
            result = dict(item)
            result["reranker_score"] = float(score)
            result["pre_rerank_rank"] = original_rank
            result["reranker_used_metadata_context"] = bool(include_metadata_context)
            ranked.append(result)

        ranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        for rank, item in enumerate(ranked, start=1):
            item["reranker_rank"] = rank

        return ranked[:top_k]

    def rerank_hybrid(
        self,
        query: str,
        *,
        rrf_top_k: int = 20,
        top_k: int = 5,
        hybrid_retriever=None,
        include_metadata_context: bool = False,
    ) -> Dict[str, Any]:
        """Fetch RRF Top-20 candidates and return the reranked Top-5 by default."""
        if rrf_top_k < 1:
            raise ValueError("rrf_top_k en az 1 olmalidir.")
        if top_k < 1:
            raise ValueError("top_k en az 1 olmalidir.")

        if hybrid_retriever is None:
            from src.retrieval.hybrid_retriever import HybridRetriever

            hybrid_retriever = HybridRetriever()

        retrieval = hybrid_retriever.search(query, n_results=rrf_top_k)
        candidates = retrieval.get("results", [])
        processed_query = retrieval.get("processed_query") or {}
        reranked = self.rerank(
            processed_query.get("rewritten_query") or query,
            candidates,
            top_k=top_k,
            include_metadata_context=include_metadata_context,
        )

        return {
            "retrieval": retrieval,
            "metadata_results": retrieval.get("metadata_results", []),
            "rrf_candidates": candidates,
            "reranked_results": reranked,
            "rrf_top_k": rrf_top_k,
            "top_k": top_k,
            "include_metadata_context": include_metadata_context,
        }
