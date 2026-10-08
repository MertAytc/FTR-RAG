from __future__ import annotations

"""
FTR-RAG — 33 Single-Gold Retrieval Benchmark
BGE vs Jev Reranker Comparison

Run from the FTR-RAG project root:

    python run_retrieval_benchmark_33_bge_vs_jev.py

Tests:
    1. BM25-only
    2. Dense-only
    3. Dense + BM25 + weighted RRF
    4. RRF + BGE reranker
    5. RRF + Jev reranker

BGE and Jev receive the exact same RRF Top-30 candidate pool.

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
from src.retrieval.jev_reranker import JevReranker, JEV_MODEL


ROOT = Path(__file__).resolve().parent

BENCHMARK_PATH = ROOT / "ftr_rag_33_new_single_gold_benchmark.json"

# Yeni sonuç dosyaları.
# Eski benchmark sonuçlarının üzerine yazmaz.
OUTPUT_CSV = ROOT / "retrieval_benchmark_33_bge_vs_jev_results.csv"
OUTPUT_JSON = ROOT / "retrieval_benchmark_33_bge_vs_jev_results.json"


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

DENSE_TOP_N = 40
BM25_TOP_N = 20

RRF_K = 60
DENSE_WEIGHT = 1.10
BM25_WEIGHT = 0.85

# RRF sonrası iki reranker da aynı Top-30'u alacak.
RRF_CANDIDATE_K = 30

# BGE
RERANK_CANDIDATE_K = 30
RERANK_FINAL_K = 10
RERANK_BATCH_SIZE = 16
RERANK_MAX_LENGTH = 512

# Jev
JEV_RERANK_CANDIDATE_K = 30


def normalize_id(value):
    return str(value).strip()


# ============================================================
# BENCHMARK LOAD
# ============================================================

def load_benchmark():
    data = json.loads(
        BENCHMARK_PATH.read_text(encoding="utf-8")
    )

    if isinstance(data, dict):
        questions = data.get("questions")
    else:
        questions = data

    if not isinstance(questions, list) or len(questions) != 33:
        count = len(questions) if isinstance(questions, list) else 0
        raise ValueError(
            f"33 soru bekleniyordu, bulunan: {count}"
        )

    normalized = []

    for q in questions:
        q = dict(q)

        gold_ids = q.get("gold_chunk_ids", [])

        # Backward compatibility:
        # Eğer tek gold_chunk_id kullanılmışsa listeye çevir.
        if not gold_ids and q.get("gold_chunk_id"):
            gold_ids = [q["gold_chunk_id"]]

        gold_ids = [
            normalize_id(x)
            for x in gold_ids
            if normalize_id(x)
        ]

        if not gold_ids:
            raise ValueError(
                f"{q.get('id')} için en az bir gold_chunk_id gerekli."
            )

        q["gold_chunk_ids"] = gold_ids
        normalized.append(q)

    return normalized


# ============================================================
# METRICS
# ============================================================

def metrics(results, gold_ids):

    ids = [
        normalize_id(x.get("id", ""))
        for x in results
        if normalize_id(x.get("id", ""))
    ]

    gold_set = {
        normalize_id(x)
        for x in gold_ids
    }

    # Retrieved gold chunk'ların rank'leri
    gold_ranks = [
        i
        for i, chunk_id in enumerate(ids, 1)
        if chunk_id in gold_set
    ]

    first_rank = (
        min(gold_ranks)
        if gold_ranks
        else None
    )

    def hit_at_1():
        return (
            1.0
            if ids and ids[0] in gold_set
            else 0.0
        )

    def recall(k):
        return (
            1.0
            if any(
                chunk_id in gold_set
                for chunk_id in ids[:k]
            )
            else 0.0
        )

    def ndcg(k):

        retrieved = ids[:k]

        dcg = 0.0

        for rank, chunk_id in enumerate(
            retrieved,
            1
        ):
            if chunk_id in gold_set:
                dcg += (
                    1.0 /
                    math.log2(rank + 1)
                )

        ideal_relevant_count = min(
            len(gold_set),
            k
        )

        idcg = sum(
            1.0 /
            math.log2(rank + 1)
            for rank in range(
                1,
                ideal_relevant_count + 1
            )
        )

        return (
            dcg / idcg
            if idcg > 0
            else 0.0
        )

    return {
        "Hit@1": hit_at_1(),
        "Recall@1": recall(1),
        "Recall@3": recall(3),
        "Recall@5": recall(5),
        "Recall@10": recall(10),
        "MRR": (
            1.0 / first_rank
            if first_rank
            else 0.0
        ),
        "nDCG@5": ndcg(5),
        "nDCG@10": ndcg(10),
        "First Relevant Rank": first_rank,
        "Gold Ranks": gold_ranks,
        "Gold Count": len(gold_set),
    }


# ============================================================
# RESULT ROW
# ============================================================

def make_row(
    method,
    results,
    q,
    processed,
    elapsed_ms,
):

    gold_ids = [
        normalize_id(x)
        for x in q["gold_chunk_ids"]
    ]

    m = metrics(
        results,
        gold_ids
    )

    return {
        "Question ID": q["id"],
        "Category": q.get(
            "category",
            ""
        ),
        "Method": method,

        "Gold IDs": "; ".join(
            gold_ids
        ),

        "Gold Count": m[
            "Gold Count"
        ],

        "Gold Ranks": "; ".join(
            str(x)
            for x in m["Gold Ranks"]
        ),

        "First Relevant Rank":
            m["First Relevant Rank"],

        "Hit@1": round(
            m["Hit@1"],
            4
        ),

        "Recall@1": round(
            m["Recall@1"],
            4
        ),

        "Recall@3": round(
            m["Recall@3"],
            4
        ),

        "Recall@5": round(
            m["Recall@5"],
            4
        ),

        "Recall@10": round(
            m["Recall@10"],
            4
        ),

        "MRR": round(
            m["MRR"],
            4
        ),

        "nDCG@5": round(
            m["nDCG@5"],
            4
        ),

        "nDCG@10": round(
            m["nDCG@10"],
            4
        ),

        "Time ms": round(
            elapsed_ms,
            1
        ),

        "Gold Found": bool(
            m["First Relevant Rank"]
        ),

        "Top 10 IDs": "; ".join(
            normalize_id(
                x.get("id", "")
            )
            for x in results[:10]
        ),

        "Question":
            q["question"],

        "Rewritten Query":
            processed.get(
                "rewritten_query",
                ""
            ),

        "BM25 Query":
            processed.get(
                "expanded_query",
                ""
            ),
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(row):

    print(
        f"  {row['Method']:<18} "
        f"Hit@1={row['Hit@1']:.4f}  "
        f"R@1={row['Recall@1']:.4f}  "
        f"R@3={row['Recall@3']:.4f}  "
        f"R@5={row['Recall@5']:.4f}  "
        f"R@10={row['Recall@10']:.4f}  "
        f"MRR={row['MRR']:.4f}  "
        f"nDCG5={row['nDCG@5']:.4f}  "
        f"nDCG10={row['nDCG@10']:.4f}  "
        f"GoldRank="
        f"{str(row['First Relevant Rank']):<4} "
        f"{row['Time ms']:.0f} ms"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(rows):

    methods = [
        "BM25-only",
        "Dense-only",
        "Dense+BM25 RRF",
        "RRF+BGE",
        "RRF+Jev",
    ]

    print(
        "\n" + "=" * 150
    )

    print(
        "33 SORULUK FTR-RAG BENCHMARK — "
        "ORTALAMA SONUÇLAR"
    )

    print(
        "=" * 150
    )

    print(
        f"{'Method':<18}"
        f"{'Hit@1':>8}"
        f"{'R@1':>8}"
        f"{'R@3':>8}"
        f"{'R@5':>8}"
        f"{'R@10':>8}"
        f"{'MRR':>8}"
        f"{'nDCG@5':>10}"
        f"{'nDCG@10':>11}"
        f"{'Avg ms':>10}"
    )

    print(
        "-" * 150
    )

    for method in methods:

        subset = [
            r
            for r in rows
            if r["Method"] == method
        ]

        if not subset:
            continue

        print(
            f"{method:<18}"
            f"{mean(r['Hit@1'] for r in subset):>8.4f}"
            f"{mean(r['Recall@1'] for r in subset):>8.4f}"
            f"{mean(r['Recall@3'] for r in subset):>8.4f}"
            f"{mean(r['Recall@5'] for r in subset):>8.4f}"
            f"{mean(r['Recall@10'] for r in subset):>8.4f}"
            f"{mean(r['MRR'] for r in subset):>8.4f}"
            f"{mean(r['nDCG@5'] for r in subset):>10.4f}"
            f"{mean(r['nDCG@10'] for r in subset):>11.4f}"
            f"{mean(r['Time ms'] for r in subset):>10.1f}"
        )

    print(
        "-" * 150
    )

    print("\nGold coverage:")

    for method in methods:

        subset = [
            r
            for r in rows
            if r["Method"] == method
        ]

        if not subset:
            continue

        found = sum(
            r["Gold Found"]
            for r in subset
        )

        hit1 = sum(
            r["Hit@1"] == 1.0
            for r in subset
        )

        top3 = sum(
            r["First Relevant Rank"]
            is not None
            and
            r["First Relevant Rank"] <= 3
            for r in subset
        )

        top5 = sum(
            r["First Relevant Rank"]
            is not None
            and
            r["First Relevant Rank"] <= 5
            for r in subset
        )

        top10 = sum(
            r["First Relevant Rank"]
            is not None
            and
            r["First Relevant Rank"] <= 10
            for r in subset
        )

        print(
            f"  {method:<18} "
            f"AnyGold={found}/33  "
            f"Hit@1={hit1}/33  "
            f"Top3={top3}/33  "
            f"Top5={top5}/33  "
            f"Top10={top10}/33"
        )


# ============================================================
# SAVE
# ============================================================

def save(
    rows,
    total_seconds
):

    fields = list(
        rows[0].keys()
    )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "question_count": 33,

                "benchmark_type":
                    "single-gold",

                "embedding_model":
                    EMBEDDING_MODEL,

                "bge_reranker_model":
                    RERANKER_MODEL,

                "jev_reranker_model":
                    JEV_MODEL,

                "dense_top_n":
                    DENSE_TOP_N,

                "bm25_top_n":
                    BM25_TOP_N,

                "rrf_k":
                    RRF_K,

                "dense_weight":
                    DENSE_WEIGHT,

                "bm25_weight":
                    BM25_WEIGHT,

                "rrf_candidate_k":
                    RRF_CANDIDATE_K,

                "bge_rerank_candidate_k":
                    RERANK_CANDIDATE_K,

                "jev_rerank_candidate_k":
                    JEV_RERANK_CANDIDATE_K,

                "rerank_final_k":
                    RERANK_FINAL_K,

                "generation":
                    False,

                "total_runtime_seconds":
                    round(
                        total_seconds,
                        2
                    ),

                "results":
                    rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 125)

    print(
        "FTR-RAG — "
        "33 SINGLE-GOLD RETRIEVAL BENCHMARK"
    )

    print("=" * 125)

    print(
        f"Embedding : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"BGE       : "
        f"{RERANKER_MODEL}"
    )

    print(
        f"Jev       : "
        f"{JEV_MODEL}"
    )

    print(
        "Pipeline  : "
        "BM25 -> Dense -> RRF -> BGE / Jev"
    )

    print(
        "Generation: KAPALI"
    )

    print(
        f"Settings  : "
        f"Dense Top-{DENSE_TOP_N}, "
        f"BM25 Top-{BM25_TOP_N}, "
        f"RRF k={RRF_K}, "
        f"weights="
        f"{DENSE_WEIGHT}/{BM25_WEIGHT}, "
        f"RRF Top-{RRF_CANDIDATE_K}, "
        f"BGE "
        f"{RERANK_CANDIDATE_K}"
        f"->{RERANK_FINAL_K}, "
        f"Jev "
        f"{JEV_RERANK_CANDIDATE_K}"
        f"->{RERANK_FINAL_K}"
    )

    benchmark = load_benchmark()

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print(
        "\nBM25 index yükleniyor..."
    )

    bm25 = BM25Retriever()

    corpus_ids = {
        normalize_id(x)
        for x in bm25.ids
        if normalize_id(x)
    }

    print(
        f"Corpus: "
        f"{len(corpus_ids)} chunk"
    )

    # --------------------------------------------------------
    # GOLD CHECK
    # --------------------------------------------------------

    missing_gold = []
    total_gold = 0

    for q in benchmark:

        for gold_id in q[
            "gold_chunk_ids"
        ]:

            total_gold += 1

            if gold_id not in corpus_ids:
                missing_gold.append(
                    (
                        q["id"],
                        gold_id
                    )
                )

    if missing_gold:

        print(
            "\n[UYARI] Index içinde "
            "bulunamayan v3.4 gold chunk'ları:"
        )

        for qid, gold_id in missing_gold:

            print(
                f"  Soru {qid}: "
                f"{gold_id}"
            )

        raise ValueError(
            f"{len(missing_gold)} "
            "gold chunk indexte bulunamadı. "
            "Önce v3.4 index oluşturulmalı."
        )

    print(
        f"Gold kontrolü: "
        f"33/33 soru ve "
        f"{total_gold} gold chunk "
        "index içinde bulundu."
    )

    # --------------------------------------------------------
    # HYBRID RETRIEVER
    # --------------------------------------------------------

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_top_n=DENSE_TOP_N,
        bm25_top_n=BM25_TOP_N,
        rrf_k=RRF_K,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
    )

    # --------------------------------------------------------
    # BGE
    # --------------------------------------------------------

    bge_reranker = (
        CrossEncoderReranker(
            batch_size=
                RERANK_BATCH_SIZE,
            max_length=
                RERANK_MAX_LENGTH,
        )
    )

    # --------------------------------------------------------
    # JEV
    # --------------------------------------------------------

    # JEV_AI_API_KEY ortam değişkeninden okunur.
    jev_reranker = JevReranker()

    rows = []

    all_start = (
        time.perf_counter()
    )

    # ========================================================
    # QUESTIONS
    # ========================================================

    for i, q in enumerate(
        benchmark,
        1
    ):

        print(
            "\n" + "-" * 125
        )

        print(
            f"SORU {i}/33 | "
            f"{q['id']} | "
            f"{q.get('category', '')}"
        )

        print(
            f"{q['question']}"
        )

        print(
            f"GOLD "
            f"({len(q['gold_chunk_ids'])}): "
            f"{'; '.join(q['gold_chunk_ids'])}"
        )

        # ----------------------------------------------------
        # QUERY PROCESSING
        # ----------------------------------------------------

        t = time.perf_counter()

        processed = process_query(
            q["question"]
        )

        rewrite_ms = (
            time.perf_counter() - t
        ) * 1000

        print(
            f"Query rewrite: "
            f"{rewrite_ms:.0f} ms"
        )

        # ====================================================
        # 1. BM25
        # ====================================================

        t = time.perf_counter()

        bm25_results = bm25.search(
            processed[
                "expanded_query"
            ],
            n_results=BM25_TOP_N,
            apply_reference_penalty=False,
        )

        elapsed = (
            time.perf_counter() - t
        ) * 1000

        row = make_row(
            "BM25-only",
            bm25_results,
            q,
            processed,
            elapsed,
        )

        rows.append(row)

        print_result(row)

        # ====================================================
        # 2. DENSE
        # ====================================================

        t = time.perf_counter()

        dense_results = (
            hybrid._dense_retrieve(
                processed[
                    "rewritten_query"
                ],
                top_n=DENSE_TOP_N,
            )
        )

        elapsed = (
            time.perf_counter() - t
        ) * 1000

        row = make_row(
            "Dense-only",
            dense_results,
            q,
            processed,
            elapsed,
        )

        rows.append(row)

        print_result(row)

        # ====================================================
        # 3. RRF
        # ====================================================

        t = time.perf_counter()

        rrf_results = (
            hybrid._rrf_fusion(
                dense_results,
                bm25_results,
                apply_quality_penalty=False,
            )
        )

        # KRİTİK:
        # Hem BGE hem Jev aynı Top-30'u kullanacak.
        rrf_results = (
            rrf_results[
                :RRF_CANDIDATE_K
            ]
        )

        elapsed = (
            time.perf_counter() - t
        ) * 1000

        row = make_row(
            "Dense+BM25 RRF",
            rrf_results,
            q,
            processed,
            elapsed,
        )

        rows.append(row)

        print_result(row)

        # ====================================================
        # 4. RRF + BGE
        # ====================================================

        t = time.perf_counter()

        bge_results = (
            bge_reranker.rerank(
                processed[
                    "rewritten_query"
                ],
                rrf_results[
                    :RERANK_CANDIDATE_K
                ],
                top_k=RERANK_FINAL_K,
            )
        )

        elapsed = (
            time.perf_counter() - t
        ) * 1000

        bge_row = make_row(
            "RRF+BGE",
            bge_results,
            q,
            processed,
            elapsed,
        )

        rows.append(bge_row)

        print_result(bge_row)

        # ====================================================
        # 5. RRF + JEV
        # ====================================================

        t = time.perf_counter()

        jev_results = (
            jev_reranker.rerank(
                processed[
                    "rewritten_query"
                ],
                rrf_results[
                    :JEV_RERANK_CANDIDATE_K
                ],
                top_k=RERANK_FINAL_K,
            )
        )

        elapsed = (
            time.perf_counter() - t
        ) * 1000

        jev_row = make_row(
            "RRF+Jev",
            jev_results,
            q,
            processed,
            elapsed,
        )

        rows.append(jev_row)

        print_result(jev_row)

        # ----------------------------------------------------
        # DIAGNOSTIC
        # ----------------------------------------------------

        if (
            bge_row[
                "First Relevant Rank"
            ]
            is None
        ):

            print(
                "  -> BGE Top-10 içinde "
                "hiçbir GOLD yok"
            )

        else:

            print(
                "  -> BGE ilk GOLD rank: "
                f"{bge_row['First Relevant Rank']}"
            )

        if (
            jev_row[
                "First Relevant Rank"
            ]
            is None
        ):

            print(
                "  -> Jev Top-10 içinde "
                "hiçbir GOLD yok"
            )

        else:

            print(
                "  -> Jev ilk GOLD rank: "
                f"{jev_row['First Relevant Rank']}"
            )

    # ========================================================
    # FINISH
    # ========================================================

    total = (
        time.perf_counter()
        - all_start
    )

    print_summary(rows)

    save(
        rows,
        total
    )

    print(
        "\n" + "=" * 125
    )

    print(
        "TEST TAMAMLANDI"
    )

    print(
        "=" * 125
    )

    print(
        f"Toplam süre : "
        f"{total:.1f} saniye"
    )

    print(
        f"CSV         : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"JSON        : "
        f"{OUTPUT_JSON}"
    )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nTest durduruldu."
        )

        sys.exit(130)

    except Exception as exc:

        print(
            "\nBENCHMARK HATASI: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise