"""Check Phase 1 retrieval invariants and compare ranking with the saved baseline.

When the constructor reports an existing dense/BM25 ID mismatch, the script
reports it and bypasses only that guard for diagnostic queries. It returns a
non-zero status for the mismatch after completing baseline comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retrieval_utils import normalize_chunk_id

BASELINE_PATH = PROJECT_ROOT / "tests" / "baselines" / "phase1_top10_baseline.json"


def _diagnostic_retriever(bm25: BM25Retriever) -> HybridRetriever:
    """Build an unguarded instance solely to compare pre/post ranking output."""
    retriever = object.__new__(HybridRetriever)
    retriever.bm25 = bm25
    retriever.dense_top_n = 40
    retriever.bm25_top_n = 40
    retriever.rrf_k = 60
    retriever.dense_weight = 1.0
    retriever.bm25_weight = 0.85
    retriever.bm25_apply_reference_penalty = True
    retriever.rrf_quality_penalty = True
    return retriever


def main() -> int:
    assert normalize_chunk_id("  chunk-1  ") == "chunk-1"
    bm25 = BM25Retriever()
    assert len(bm25.ids) == len(set(bm25.ids)), "BM25 chunk IDs are not unique."
    assert all(chunk_id == normalize_chunk_id(chunk_id) for chunk_id in bm25.ids)
    print(f"BM25 ID check passed: {len(bm25.ids)} unique normalized IDs")

    mismatch = None
    try:
        retriever = HybridRetriever(bm25_retriever=bm25)
        print("Hybrid dense/BM25 consistency check passed.")
    except RuntimeError as exc:
        mismatch = str(exc)
        print(f"Hybrid consistency check failed: {mismatch}")
        retriever = _diagnostic_retriever(bm25)
        print("Continuing with diagnostic queries; constructor guard remains enforced.")

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Saved ranking baseline is missing: {BASELINE_PATH}")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    current_rows = []
    for row in baseline["questions"]:
        query = row["query"]
        result = retriever.search(query, n_results=10)
        dense_top5 = [item["id"] for item in result["dense_results"][:5]]
        bm25_top5 = [item["id"] for item in result["bm25_results"][:5]]
        rrf_top10 = [item["id"] for item in result["results"]]
        current_rows.append({
            "question_id": row.get("question_id"),
            "query": query,
            "dense_top5": dense_top5,
            "bm25_top5": bm25_top5,
            "rrf_top10": rrf_top10,
        })
        print(f"\n{row.get('question_id')} dense top-5: {dense_top5}")
        print(f"{row.get('question_id')} BM25 top-5: {bm25_top5}")
        print(f"{row.get('question_id')} RRF top-5: {rrf_top10[:5]}")

    expected = [row["rrf_top10"] for row in baseline["questions"]]
    actual = [row["rrf_top10"] for row in current_rows]
    if actual != expected:
        print("RRF baseline comparison failed.")
        return 1
    print("\nRRF top-10 IDs exactly match the pre-edit baseline.")
    if mismatch:
        print("Note: the existing dense/BM25 index mismatch still prevents normal HybridRetriever construction.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
