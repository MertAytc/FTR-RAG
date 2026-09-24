from __future__ import annotations

from collections import Counter
from typing import Dict, List, Any, Set
import math

from src.retrieval.embedding import embed_texts
from src.retrieval.hybrid_retriever import HybridRetriever


def _cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []

    low = min(values)
    high = max(values)

    if math.isclose(low, high):
        return [1.0] * len(values)

    return [(value - low) / (high - low) for value in values]


class MMRRetriever:
    """Hybrid + quality-aware RRF + document-aware MMR.

    Improvements over the previous implementation:
    1. MMR semantic similarity uses the English clinical rewrite, not the
       original Turkish query against an English corpus.
    2. Candidate pool is larger so relevant chunks have room to survive.
    3. A per-document cap prevents bibliography/reference or single-document
       clusters from consuming the entire final context.
    4. Reference-heavy chunks already down-ranked by HybridRetriever remain
       disadvantaged.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        candidate_k: int = 40,
        final_k: int = 6,
        lambda_mult: float = 0.80,
        rrf_weight: float = 0.85,
        semantic_weight: float = 0.15,
        max_chunks_per_document: int = 2,
    ):
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult 0 ile 1 arasında olmalıdır.")
        if not 0.0 <= rrf_weight <= 1.0:
            raise ValueError("rrf_weight 0 ile 1 arasında olmalıdır.")
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight 0 ile 1 arasında olmalıdır.")
        if not math.isclose(rrf_weight + semantic_weight, 1.0):
            raise ValueError("rrf_weight + semantic_weight = 1 olmalıdır.")
        if max_chunks_per_document < 1:
            raise ValueError("max_chunks_per_document en az 1 olmalıdır.")

        self.hybrid = hybrid_retriever or HybridRetriever()
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.lambda_mult = lambda_mult
        self.rrf_weight = rrf_weight
        self.semantic_weight = semantic_weight
        self.max_chunks_per_document = max_chunks_per_document

    @staticmethod
    def _document_id(item: Dict[str, Any]) -> str:
        metadata = item.get("metadata") or {}
        return str(
            metadata.get("document_id")
            or item.get("id", "").split("_chunk_")[0]
        )

    def _prepare_candidates(
        self,
        semantic_query: str,
        candidates: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[List[float]]]:
        candidates = [
            dict(item)
            for item in candidates[:self.candidate_k]
        ]

        if not candidates:
            return [], []

        rrf_scores = [
            float(item["rrf_score"])
            for item in candidates
        ]
        normalized_rrf = _minmax(rrf_scores)

        # IMPORTANT: this is the English clinical rewrite. The previous
        # implementation embedded the original Turkish query here.
        query_embedding = embed_texts([semantic_query])[0]
        candidate_embeddings = embed_texts(
            [item["document"] for item in candidates]
        )

        semantic_scores = [
            _cosine_similarity(query_embedding, embedding)
            for embedding in candidate_embeddings
        ]
        normalized_semantic = _minmax(semantic_scores)

        for item, rrf_rel, semantic_raw, semantic_rel in zip(
            candidates,
            normalized_rrf,
            semantic_scores,
            normalized_semantic,
        ):
            item["normalized_rrf"] = rrf_rel
            item["semantic_similarity"] = semantic_raw
            item["normalized_semantic"] = semantic_rel
            item["relevance"] = (
                self.rrf_weight * rrf_rel
                + self.semantic_weight * semantic_rel
            )

        return candidates, candidate_embeddings

    def select(
        self,
        semantic_query: str,
        candidates: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        candidates, embeddings = self._prepare_candidates(
            semantic_query,
            candidates,
        )

        if not candidates:
            return [], []

        remaining = list(range(len(candidates)))
        selected: List[int] = []
        selection_trace: List[Dict[str, Any]] = []
        document_counts: Counter[str] = Counter()

        def allowed(index: int, strict_cap: bool = True) -> bool:
            if not strict_cap:
                return True
            doc_id = self._document_id(candidates[index])
            return document_counts[doc_id] < self.max_chunks_per_document

        def score_candidate(
            candidate_index: int,
        ) -> tuple[float, float]:
            relevance = float(candidates[candidate_index]["relevance"])

            if not selected:
                return relevance, 0.0

            redundancy = max(
                _cosine_similarity(
                    embeddings[candidate_index],
                    embeddings[selected_index],
                )
                for selected_index in selected
            )

            # Cosine [-1,1] -> [0,1]
            redundancy = (redundancy + 1.0) / 2.0

            mmr_score = (
                self.lambda_mult * relevance
                - (1.0 - self.lambda_mult) * redundancy
            )
            return mmr_score, redundancy

        while remaining and len(selected) < self.final_k:
            eligible = [
                i for i in remaining
                if allowed(i, strict_cap=True)
            ]

            # If the corpus has fewer than final_k distinct documents, do not
            # fail retrieval just because the diversity cap is unreachable.
            if not eligible:
                eligible = remaining

            best_index = None
            best_score = -math.inf
            best_redundancy = 0.0

            for candidate_index in eligible:
                mmr_score, redundancy = score_candidate(candidate_index)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = candidate_index
                    best_redundancy = redundancy

            assert best_index is not None

            item = candidates[best_index]
            item["mmr_score"] = best_score
            item["redundancy"] = best_redundancy
            item["selection_rank"] = len(selected) + 1

            selected.append(best_index)
            remaining.remove(best_index)

            doc_id = self._document_id(item)
            document_counts[doc_id] += 1

            selection_trace.append({
                "selection_rank": len(selected),
                "id": item["id"],
                "document_id": doc_id,
                "rrf_rank": item.get("rrf_rank"),
                "rrf_score": item.get("rrf_score"),
                "relevance": item.get("relevance"),
                "semantic_similarity": item.get("semantic_similarity"),
                "mmr_score": best_score,
                "redundancy": best_redundancy,
            })

        return [candidates[i] for i in selected], selection_trace

    def search(
        self,
        query: str,
        debug_gold_ids: Set[str] | None = None,
    ) -> Dict[str, Any]:
        hybrid_result = self.hybrid.search(
            query,
            n_results=self.candidate_k,
            debug_gold_ids=debug_gold_ids,
        )

        # Use the same processed query that drove dense retrieval.
        semantic_query = hybrid_result["processed_query"]["rewritten_query"]

        selected, selection_trace = self.select(
            semantic_query,
            hybrid_result["results"],
        )

        document_counts = Counter(
            self._document_id(item)
            for item in selected
        )

        return {
            "query": query,
            "processed_query": hybrid_result["processed_query"],
            "hybrid_results": hybrid_result["results"],
            "debug": hybrid_result.get("debug"),
            "mmr_debug": {
                "candidate_k": self.candidate_k,
                "final_k": self.final_k,
                "lambda_mult": self.lambda_mult,
                "rrf_weight": self.rrf_weight,
                "semantic_weight": self.semantic_weight,
                "max_chunks_per_document": self.max_chunks_per_document,
                "selection_trace": selection_trace,
                "selected_document_counts": dict(document_counts),
                "semantic_query": semantic_query,
            },
            "results": selected,
        }
