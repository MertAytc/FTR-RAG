from __future__ import annotations

from typing import Dict, List, Any, Set

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding import embed_texts
from src.retrieval.vector_store import search as dense_search, count as dense_count
from src.retrieval.query_processor import process_query
from src.retrieval.retrieval_utils import annotate_result


class HybridRetriever:
    """Dense + BM25 hybrid retrieval with weighted, quality-aware RRF.

    Design goals:
    - Dense retrieval uses the English clinical rewrite.
    - BM25 uses the same rewrite plus validated terminology.
    - Bibliography/reference-heavy chunks are down-ranked rather than deleted.
    - Dense and lexical evidence can be weighted independently.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        dense_top_n: int = 40,
        bm25_top_n: int = 40,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 0.85,
    ):
        if dense_weight <= 0 or bm25_weight <= 0:
            raise ValueError("RRF ağırlıkları pozitif olmalıdır.")

        self.bm25 = bm25_retriever or BM25Retriever()
        self.dense_top_n = dense_top_n
        self.bm25_top_n = bm25_top_n
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def _dense_retrieve(
        self,
        query: str,
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = embed_texts([query])[0]
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
                "id": chunk_id,
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
        )

    def _rrf_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        apply_quality_penalty: bool = True,
    ) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}

        for result in dense_results:
            chunk_id = result["id"]
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

            fused[chunk_id]["dense_rank"] = result["rank"]
            fused[chunk_id]["dense_score"] = result["score"]

            quality = result.get("reference_penalty", 1.0) if apply_quality_penalty else 1.0
            fused[chunk_id]["rrf_score"] += (
                self.dense_weight
                * quality
                / (self.rrf_k + result["rank"])
            )

        for result in bm25_results:
            chunk_id = result["id"]
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

            fused[chunk_id]["bm25_rank"] = result["rank"]
            fused[chunk_id]["bm25_score"] = result["score"]

            quality = result.get("reference_penalty", 1.0) if apply_quality_penalty else 1.0
            fused[chunk_id]["rrf_score"] += (
                self.bm25_weight
                * quality
                / (self.rrf_k + result["rank"])
            )

        ranked = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        for rank, item in enumerate(ranked, start=1):
            item["rrf_rank"] = rank
        return ranked

    def search(
        self,
        query: str,
        n_results: int = 8,
        debug_gold_ids: Set[str] | None = None,
    ) -> Dict[str, Any]:
        processed = process_query(query)

        dense_results = self._dense_retrieve(
            processed["rewritten_query"]
        )
        bm25_results = self._bm25_retrieve(
            processed["expanded_query"]
        )
        fused_results = self._rrf_fusion(
            dense_results,
            bm25_results,
        )

        debug = None

        if debug_gold_ids:
            gold = {
                str(x).strip()
                for x in debug_gold_ids
                if str(x).strip()
            }

            dense_top_ids = {
                str(x.get("id", "")).strip()
                for x in dense_results
            }
            bm25_top_ids = {
                str(x.get("id", "")).strip()
                for x in bm25_results
            }
            rrf_ranks = {
                str(x.get("id", "")).strip(): i
                for i, x in enumerate(fused_results, start=1)
            }

            full_dense = self._dense_retrieve(
                processed["rewritten_query"],
                top_n=dense_count(),
            )
            dense_rank_map = {
                str(x["id"]).strip(): i
                for i, x in enumerate(full_dense, start=1)
            }
            dense_item_map = {
                str(x["id"]).strip(): x
                for x in full_dense
            }
            bm25_debug = self.bm25.debug_rank(
                processed["expanded_query"],
                gold,
            )

            gold_rows = []
            for gid in sorted(gold):
                d = dense_item_map.get(gid)
                b = bm25_debug.get(gid)

                gold_rows.append({
                    "ID": gid,
                    "Dense Top N": gid in dense_top_ids,
                    "Dense rank (full)": dense_rank_map.get(gid),
                    "Dense distance": d.get("score") if d else None,
                    "BM25 Top N": gid in bm25_top_ids,
                    "BM25 rank (full)": b.get("rank") if b else None,
                    "BM25 score": b.get("score") if b else None,
                    "BM25 raw score": b.get("raw_score") if b else None,
                    "BM25 role": b.get("chunk_role") if b else None,
                    # Keep both names for compatibility with old/new UI and logs.
                    "RRF rank": rrf_ranks.get(gid),
                    "RRF rank (candidate pool)": rrf_ranks.get(gid),
                    "In candidate pool": rrf_ranks.get(gid, 10**9) <= n_results,
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
                    "rrf_quality_penalty_default": True,
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
