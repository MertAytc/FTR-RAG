from __future__ import annotations

from typing import Any, Dict, List, Set

from src.retrieval import embedding
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding import embed_texts
from src.retrieval.query_processor import process_query
from src.retrieval.retrieval_utils import annotate_result, normalize_chunk_id
from src.retrieval.vector_store import (
    count as dense_count,
    get_indexed_ids,
    open_collection,
    search as dense_search,
)


class HybridRetriever:
    """Dense + BM25 hybrid retrieval with weighted, quality-aware RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        dense_top_n: int = 40,
        bm25_top_n: int = 40,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 0.85,
        bm25_apply_reference_penalty: bool = True,
        rrf_quality_penalty: bool = True,
    ):
        if dense_weight <= 0 or bm25_weight <= 0:
            raise ValueError("RRF weights must be positive.")

        self.bm25 = bm25_retriever or BM25Retriever()
        self.dense_top_n = dense_top_n
        self.bm25_top_n = bm25_top_n
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.bm25_apply_reference_penalty = bm25_apply_reference_penalty
        self.rrf_quality_penalty = rrf_quality_penalty

        dense_ids = {normalize_chunk_id(value) for value in get_indexed_ids()}
        bm25_ids = {normalize_chunk_id(value) for value in self.bm25.ids}
        dense_only = sorted(dense_ids - bm25_ids)
        bm25_only = sorted(bm25_ids - dense_ids)
        if dense_only or bm25_only:
            raise RuntimeError(
                "Dense and BM25 indexes contain different chunk IDs: "
                f"dense-only={len(dense_only)} (examples: {dense_only[:5]}), "
                f"BM25-only={len(bm25_only)} (examples: {bm25_only[:5]})."
            )

        collection_metadata = open_collection().metadata or {}
        indexed_model = collection_metadata.get("embedding_model")
        if indexed_model != "unknown" and indexed_model != embedding.EMBEDDING_MODEL:
            raise RuntimeError(
                "Dense index embedding model mismatch: "
                f"collection={indexed_model!r}, configured={embedding.EMBEDDING_MODEL!r}."
            )

    def _dense_retrieve(
        self,
        query: str,
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = embed_texts([query], input_type="query")[0]
        results = dense_search(
            query_embedding=query_embedding,
            n_results=top_n or self.dense_top_n,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        dense_results = []
        for rank, chunk_id in enumerate(ids, start=1):
            item = {
                "id": normalize_chunk_id(chunk_id),
                "rank": rank,
                "score": float(distances[rank - 1]),
                "document": documents[rank - 1],
                "metadata": metadatas[rank - 1],
            }
            dense_results.append(annotate_result(item))
        return dense_results

    def _bm25_retrieve(self, expanded_query: str) -> List[Dict[str, Any]]:
        return self.bm25.search(
            expanded_query,
            n_results=self.bm25_top_n,
            apply_reference_penalty=self.bm25_apply_reference_penalty,
        )

    def _accumulate_channel(
        self,
        fused: Dict[str, Dict[str, Any]],
        results: List[Dict[str, Any]],
        channel: str,
        weight: float,
        apply_quality_penalty: bool,
    ) -> None:
        """Add one retrieval channel's reciprocal-rank contributions."""
        rank_key = f"{channel}_rank"
        score_key = f"{channel}_score"
        for result in results:
            chunk_id = normalize_chunk_id(result["id"])
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "id": chunk_id,
                    "document": result["document"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_score": None,
                    "bm25_score": None,
                    "chunk_role": result.get("chunk_role", "clinical"),
                }

            rank = result["rank"]
            fused[chunk_id][rank_key] = rank
            fused[chunk_id][score_key] = result["score"]
            quality = (
                result.get("reference_penalty", 1.0)
                if apply_quality_penalty
                else 1.0
            )
            fused[chunk_id]["rrf_score"] += weight * quality / (self.rrf_k + rank)

    def _rrf_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        apply_quality_penalty: bool = True,
    ) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        self._accumulate_channel(
            fused, dense_results, "dense", self.dense_weight, apply_quality_penalty
        )
        self._accumulate_channel(
            fused, bm25_results, "bm25", self.bm25_weight, apply_quality_penalty
        )

        ranked = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
        for rank, item in enumerate(ranked, start=1):
            item["rrf_rank"] = rank
        return ranked

    def search(
        self,
        query: str,
        n_results: int = 10,
        debug_gold_ids: Set[str] | None = None,
    ) -> Dict[str, Any]:
        processed = process_query(query)
        dense_results = self._dense_retrieve(processed["rewritten_query"])
        bm25_results = self._bm25_retrieve(processed["expanded_query"])
        fused_results = self._rrf_fusion(
            dense_results,
            bm25_results,
            apply_quality_penalty=self.rrf_quality_penalty,
        )

        debug = None
        if debug_gold_ids:
            gold = {
                normalize_chunk_id(value)
                for value in debug_gold_ids
                if normalize_chunk_id(value)
            }
            dense_top_ids = {normalize_chunk_id(item.get("id", "")) for item in dense_results}
            bm25_top_ids = {normalize_chunk_id(item.get("id", "")) for item in bm25_results}
            rrf_ranks = {
                normalize_chunk_id(item.get("id", "")): rank
                for rank, item in enumerate(fused_results, start=1)
            }

            full_dense = self._dense_retrieve(
                processed["rewritten_query"],
                top_n=dense_count(),
            )
            dense_rank_map = {
                normalize_chunk_id(item["id"]): rank
                for rank, item in enumerate(full_dense, start=1)
            }
            dense_item_map = {
                normalize_chunk_id(item["id"]): item
                for item in full_dense
            }
            bm25_debug = self.bm25.debug_rank(
                processed["expanded_query"],
                gold,
                apply_reference_penalty=self.bm25_apply_reference_penalty,
            )

            gold_rows = []
            for chunk_id in sorted(gold):
                dense_item = dense_item_map.get(chunk_id)
                bm25_item = bm25_debug.get(chunk_id)
                gold_rows.append({
                    "ID": chunk_id,
                    "Dense Top N": chunk_id in dense_top_ids,
                    "Dense rank (full)": dense_rank_map.get(chunk_id),
                    "Dense distance": dense_item.get("score") if dense_item else None,
                    "BM25 Top N": chunk_id in bm25_top_ids,
                    "BM25 rank (full)": bm25_item.get("rank") if bm25_item else None,
                    "BM25 score": bm25_item.get("score") if bm25_item else None,
                    "BM25 raw score": bm25_item.get("raw_score") if bm25_item else None,
                    "BM25 role": bm25_item.get("chunk_role") if bm25_item else None,
                    "RRF rank": rrf_ranks.get(chunk_id),
                    "RRF rank (candidate pool)": rrf_ranks.get(chunk_id),
                    "In candidate pool": rrf_ranks.get(chunk_id, 10**9) <= n_results,
                })

            debug = {
                "gold": gold_rows,
                "dense_top": dense_results,
                "bm25_top": bm25_results,
                "rrf_top": fused_results[:n_results],
                "dense_index_count": len(full_dense),
                "bm25_index_count": len(self.bm25.ids),
                "config": {
                    "dense_top_n": self.dense_top_n,
                    "bm25_top_n": self.bm25_top_n,
                    "rrf_k": self.rrf_k,
                    "dense_weight": self.dense_weight,
                    "bm25_weight": self.bm25_weight,
                    "bm25_apply_reference_penalty": self.bm25_apply_reference_penalty,
                    "rrf_quality_penalty": self.rrf_quality_penalty,
                },
            }

        return {
            "query": query,
            "processed_query": processed,
            "dense_query": processed["rewritten_query"],
            "bm25_query": processed["expanded_query"],
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "results": fused_results[:n_results],
            "debug": debug,
        }
