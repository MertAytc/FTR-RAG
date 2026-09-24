from __future__ import annotations

"""
FTR-RAG — 20 Single-Gold Retrieval Benchmark

Run from the FTR-RAG project root:

    python run_retrieval_benchmark_20_single_gold.py

Tests:
    1. BM25-only
    2. Dense-only
    3. Dense + BM25 + weighted RRF
    4. RRF + BGE reranker

No answer generation is performed.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import process_query
from src.retrieval.embedding import EMBEDDING_MODEL
from src.retrieval.reranker import CrossEncoderReranker, RERANKER_MODEL


ROOT = Path(__file__).resolve().parent

BENCHMARK_PATH = ROOT / "benchmark_questions_20_single_gold.json"
OUTPUT_CSV = ROOT / "retrieval_benchmark_20_single_gold_results.csv"
OUTPUT_JSON = ROOT / "retrieval_benchmark_20_single_gold_results.json"

DENSE_TOP_N = 40
BM25_TOP_N = 40
RRF_K = 60
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 0.85

RRF_CANDIDATE_K = 30
RERANK_CANDIDATE_K = 30
RERANK_FINAL_K = 10
RERANK_BATCH_SIZE = 8
RERANK_MAX_LENGTH = 512


def normalize_id(value):
    return str(value).strip()


def load_benchmark():
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    if not isinstance(data, list) or len(data) != 20:
        raise ValueError(f"20 soru bekleniyordu, bulunan: {len(data)}")

    for q in data:
        if len(q.get("gold_chunk_ids", [])) != 1:
            raise ValueError(
                f"{q.get('id')} single-gold değil: {q.get('gold_chunk_ids')}"
            )

    return data


def metrics(results, gold_id):
    ids = [
        normalize_id(x.get("id", ""))
        for x in results
        if normalize_id(x.get("id", ""))
    ]

    first_rank = next(
        (i for i, x in enumerate(ids, 1) if x == gold_id),
        None
    )

    def recall(k):
        return 1.0 if gold_id in ids[:k] else 0.0

    def ndcg(k):
        if gold_id not in ids[:k]:
            return 0.0

        rank = ids.index(gold_id) + 1
        return 1.0 / math.log2(rank + 1)

    return {
        "Recall@1": recall(1),
        "Recall@3": recall(3),
        "Recall@5": recall(5),
        "Recall@10": recall(10),
        "MRR": 1.0 / first_rank if first_rank else 0.0,
        "nDCG@5": ndcg(5),
        "nDCG@10": ndcg(10),
        "First Relevant Rank": first_rank,
    }


def make_row(method, results, q, processed, elapsed_ms):
    gold = normalize_id(q["gold_chunk_ids"][0])
    m = metrics(results, gold)

    return {
        "Question ID": q["id"],
        "Category": q.get("category", ""),
        "Method": method,
        "Gold ID": gold,
        "First Relevant Rank": m["First Relevant Rank"],
        "Recall@1": round(m["Recall@1"], 4),
        "Recall@3": round(m["Recall@3"], 4),
        "Recall@5": round(m["Recall@5"], 4),
        "Recall@10": round(m["Recall@10"], 4),
        "MRR": round(m["MRR"], 4),
        "nDCG@5": round(m["nDCG@5"], 4),
        "nDCG@10": round(m["nDCG@10"], 4),
        "Time ms": round(elapsed_ms, 1),
        "Gold Found": bool(m["First Relevant Rank"]),
        "Top 10 IDs": "; ".join(
            normalize_id(x.get("id", "")) for x in results[:10]
        ),
        "Question": q["question"],
        "Rewritten Query": processed.get("rewritten_query", ""),
        "BM25 Query": processed.get("expanded_query", ""),
    }


def print_result(row):
    print(
        f"  {row['Method']:<18} "
        f"R@1={row['Recall@1']:.4f}  "
        f"R@3={row['Recall@3']:.4f}  "
        f"R@5={row['Recall@5']:.4f}  "
        f"R@10={row['Recall@10']:.4f}  "
        f"MRR={row['MRR']:.4f}  "
        f"nDCG5={row['nDCG@5']:.4f}  "
        f"nDCG10={row['nDCG@10']:.4f}  "
        f"GoldRank={str(row['First Relevant Rank']):<4} "
        f"{row['Time ms']:.0f} ms"
    )


def print_summary(rows):
    methods = [
        "BM25-only",
        "Dense-only",
        "Dense+BM25 RRF",
        "RRF+Reranker",
    ]

    print("\n" + "=" * 125)
    print("20 SORULUK FTR-RAG BENCHMARK — ORTALAMA SONUÇLAR")
    print("=" * 125)

    print(
        f"{'Method':<18}"
        f"{'R@1':>8}"
        f"{'R@3':>8}"
        f"{'R@5':>8}"
        f"{'R@10':>8}"
        f"{'MRR':>8}"
        f"{'nDCG@5':>10}"
        f"{'nDCG@10':>11}"
        f"{'Avg ms':>10}"
    )
    print("-" * 125)

    for method in methods:
        subset = [r for r in rows if r["Method"] == method]
        if not subset:
            continue

        print(
            f"{method:<18}"
            f"{mean(r['Recall@1'] for r in subset):>8.4f}"
            f"{mean(r['Recall@3'] for r in subset):>8.4f}"
            f"{mean(r['Recall@5'] for r in subset):>8.4f}"
            f"{mean(r['Recall@10'] for r in subset):>8.4f}"
            f"{mean(r['MRR'] for r in subset):>8.4f}"
            f"{mean(r['nDCG@5'] for r in subset):>10.4f}"
            f"{mean(r['nDCG@10'] for r in subset):>11.4f}"
            f"{mean(r['Time ms'] for r in subset):>10.1f}"
        )

    print("-" * 125)

    print("\nGold coverage:")
    for method in methods:
        subset = [r for r in rows if r["Method"] == method]
        if not subset:
            continue

        found = sum(r["Gold Found"] for r in subset)
        top1 = sum(r["First Relevant Rank"] == 1 for r in subset)
        top10 = sum(
            r["First Relevant Rank"] is not None
            and r["First Relevant Rank"] <= 10
            for r in subset
        )

        print(
            f"  {method:<18} "
            f"Gold={found}/20  "
            f"Top1={top1}/20  "
            f"Top10={top10}/20"
        )


def save(rows, total_seconds):
    fields = list(rows[0].keys())

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "question_count": 20,
                "embedding_model": EMBEDDING_MODEL,
                "reranker_model": RERANKER_MODEL,
                "dense_top_n": DENSE_TOP_N,
                "bm25_top_n": BM25_TOP_N,
                "rrf_k": RRF_K,
                "dense_weight": DENSE_WEIGHT,
                "bm25_weight": BM25_WEIGHT,
                "rrf_candidate_k": RRF_CANDIDATE_K,
                "rerank_candidate_k": RERANK_CANDIDATE_K,
                "rerank_final_k": RERANK_FINAL_K,
                "generation": False,
                "total_runtime_seconds": round(total_seconds, 2),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    print("=" * 125)
    print("FTR-RAG — 20 SINGLE-GOLD RETRIEVAL BENCHMARK")
    print("=" * 125)
    print(f"Embedding : {EMBEDDING_MODEL}")
    print(f"Reranker  : {RERANKER_MODEL}")
    print("Pipeline  : BM25 -> Dense -> RRF -> RRF+Reranker")
    print("Generation: KAPALI")
    print(
        f"Settings  : Dense Top-{DENSE_TOP_N}, "
        f"BM25 Top-{BM25_TOP_N}, "
        f"RRF k={RRF_K}, weights={DENSE_WEIGHT}/{BM25_WEIGHT}, "
        f"RRF Top-{RRF_CANDIDATE_K}, "
        f"Reranker {RERANK_CANDIDATE_K}->{RERANK_FINAL_K}"
    )

    benchmark = load_benchmark()

    print("\nBM25 index yükleniyor...")
    bm25 = BM25Retriever()
    corpus_ids = {
        normalize_id(x) for x in bm25.ids if normalize_id(x)
    }
    print(f"Corpus: {len(corpus_ids)} chunk")

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_top_n=DENSE_TOP_N,
        bm25_top_n=BM25_TOP_N,
        rrf_k=RRF_K,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
    )

    reranker = CrossEncoderReranker(
        batch_size=RERANK_BATCH_SIZE,
        max_length=RERANK_MAX_LENGTH,
    )

    rows = []
    all_start = time.perf_counter()

    for i, q in enumerate(benchmark, 1):
        print("\n" + "-" * 125)
        print(f"SORU {i}/20 | {q['id']} | {q.get('category', '')}")
        print(f"{q['question']}")
        print(f"GOLD: {q['gold_chunk_ids'][0]}")

        # Same query processing for every retrieval method.
        t = time.perf_counter()
        processed = process_query(q["question"])
        rewrite_ms = (time.perf_counter() - t) * 1000
        print(f"Query rewrite: {rewrite_ms:.0f} ms")

        # ------------------------------------------------------------
        # 1. BM25
        # ------------------------------------------------------------
        t = time.perf_counter()
        bm25_results = bm25.search(
            processed["expanded_query"],
            n_results=BM25_TOP_N,
            apply_reference_penalty=False,
        )
        elapsed = (time.perf_counter() - t) * 1000

        row = make_row("BM25-only", bm25_results, q, processed, elapsed)
        rows.append(row)
        print_result(row)

        # ------------------------------------------------------------
        # 2. Dense
        # ------------------------------------------------------------
        t = time.perf_counter()
        dense_results = hybrid._dense_retrieve(
            processed["rewritten_query"],
            top_n=DENSE_TOP_N,
        )
        elapsed = (time.perf_counter() - t) * 1000

        row = make_row("Dense-only", dense_results, q, processed, elapsed)
        rows.append(row)
        print_result(row)

        # ------------------------------------------------------------
        # 3. RRF
        # ------------------------------------------------------------
        t = time.perf_counter()
        rrf_results = hybrid._rrf_fusion(
            dense_results,
            bm25_results,
            apply_quality_penalty=False,
        )
        rrf_results = rrf_results[:RRF_CANDIDATE_K]
        elapsed = (time.perf_counter() - t) * 1000

        row = make_row(
            "Dense+BM25 RRF",
            rrf_results,
            q,
            processed,
            elapsed,
        )
        rows.append(row)
        print_result(row)

        # ------------------------------------------------------------
        # 4. RRF + Reranker
        # ------------------------------------------------------------
        t = time.perf_counter()
        reranked_results = reranker.rerank(
            processed["rewritten_query"],
            rrf_results[:RERANK_CANDIDATE_K],
            top_k=RERANK_FINAL_K,
        )
        elapsed = (time.perf_counter() - t) * 1000

        row = make_row(
            "RRF+Reranker",
            reranked_results,
            q,
            processed,
            elapsed,
        )
        rows.append(row)
        print_result(row)

        # Useful diagnostic.
        if row["First Relevant Rank"] is None:
            print("  -> Reranker Top-10 içinde GOLD YOK")
        else:
            print(
                f"  -> Reranker Gold rank: "
                f"{row['First Relevant Rank']}"
            )

    total = time.perf_counter() - all_start

    print_summary(rows)
    save(rows, total)

    print("\n" + "=" * 125)
    print("TEST TAMAMLANDI")
    print("=" * 125)
    print(f"Toplam süre : {total:.1f} saniye")
    print(f"CSV         : {OUTPUT_CSV}")
    print(f"JSON        : {OUTPUT_JSON}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest durduruldu.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nBENCHMARK HATASI: {type(exc).__name__}: {exc}")
        raise
