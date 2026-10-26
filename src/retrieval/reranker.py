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
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        top_k = max(1, min(top_k, len(candidates)))

        pairs = [
            (query, str(item.get("document", "")))
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
        )

        return {
            "retrieval": retrieval,
            "rrf_candidates": candidates,
            "reranked_results": reranked,
            "rrf_top_k": rrf_top_k,
            "top_k": top_k,
        }
